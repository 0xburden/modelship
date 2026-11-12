package modelship

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

//go:embed assets/*
var assets embed.FS
var containerName = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_-]*$`)

const managedLabel = "io.modelship.managed=true"

func docker(ctx context.Context, args ...string) error {
	cmd := exec.CommandContext(ctx, "docker", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	return cmd.Run()
}
func output(ctx context.Context, args ...string) (string, error) {
	b, err := exec.CommandContext(ctx, "docker", args...).Output()
	return strings.TrimSpace(string(b)), err
}
func BuildArgs(p Plan) []string {
	args := []string{"run", "--detach", "--name", p.Name, "--label", managedLabel,
		"--label", "io.modelship.model=" + p.Model, "--label", "io.modelship.revision=" + p.Revision,
		"--label", "io.modelship.resident-bytes=" + strconv.FormatInt(p.WeightResidentBytes, 10),
		"--label", "io.modelship.gpu-uuid=" + p.GPU.UUID,
		"--restart", "unless-stopped",
		"--publish", fmt.Sprintf("127.0.0.1:%d:8000", p.Port),
		"--security-opt", "no-new-privileges", "--cap-drop", "ALL",
		"--shm-size", "2g", "--mount", "type=volume,source=modelship-hf-cache,target=/cache/huggingface",
		"--env", "HF_HOME=/cache/huggingface", "--env", "HF_HUB_DISABLE_TELEMETRY=1", "--env", "USE_HUB_KERNELS=NO"}
	args = append(args, GPUArgs(p)...)
	if os.Getenv("HF_TOKEN") != "" {
		args = append(args, "--env", "HF_TOKEN")
	}
	if p.Runtime == "vllm" {
		// The vLLM image owns its entrypoint. No remote model code or shell interpolation.
		args = append(args, p.Image, "--model", p.Model, "--revision", p.Revision,
			"--tokenizer-revision", p.Revision, "--host", "0.0.0.0", "--port", "8000",
			"--served-model-name", p.Model, "--dtype", "half", "--load-format", "safetensors",
			"--max-model-len", fmt.Sprint(p.Context), "--max-num-seqs", "1",
			"--gpu-memory-utilization", fmt.Sprintf("%.4f", p.GPUMemoryFraction), "--enforce-eager")
	} else {
		args = append(args, "--env", "MODEL_ID="+p.Model, "--env", "MODEL_REVISION="+p.Revision,
			"--env", "MODEL_TASK="+p.Task, "--env", "MODEL_PIPELINE="+p.Pipeline, p.Image)
	}
	return args
}
func buildImages(ctx context.Context) error {
	dir, err := os.MkdirTemp("", "modelship-build-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	for _, name := range []string{"Dockerfile", "server.py"} {
		b, e := assets.ReadFile("assets/" + name)
		if e != nil {
			return e
		}
		if e = os.WriteFile(filepath.Join(dir, name), b, 0600); e != nil {
			return e
		}
	}
	return docker(ctx, "build", "--tag", ImageRuntime, dir)
}
func Deploy(ctx context.Context, p Plan, wait time.Duration) error {
	if p.Backend == "native" {
		return DeployNative(ctx, p, wait)
	}
	if len(p.Blockers) > 0 {
		return fmt.Errorf("deployment blocked: %s", strings.Join(p.Blockers, "; "))
	}
	if err := verifyWeightFitNow(ctx, p); err != nil {
		return err
	}
	// Both metadata discovery and Docker must refer to this local server.
	endpoint, err := output(ctx, "context", "inspect", "--format", "{{.Endpoints.docker.Host}}")
	if err != nil {
		return fmt.Errorf("cannot inspect Docker context")
	}
	if host := os.Getenv("DOCKER_HOST"); host != "" {
		endpoint = host
	}
	if !strings.HasPrefix(endpoint, "unix://") {
		return fmt.Errorf("remote Docker endpoints are unsupported; use a local Unix socket")
	}
	// Fail before builds or downloads if this name already exists.
	if _, err = output(ctx, "inspect", p.Name); err == nil {
		return fmt.Errorf("container %s already exists; use a new --name, or explicitly uninstall it", p.Name)
	}
	// Resolve provider-specific access once; use the same mode for preflight and serving.
	p, err = GPUPreflight(ctx, p)
	if err != nil {
		return err
	}
	if p.Runtime == "diffusers" {
		if err = buildImages(ctx); err != nil {
			return fmt.Errorf("build image runtime: %w", err)
		}
	}
	if err = verifyWeightFitNow(ctx, p); err != nil {
		return err
	}
	// Separate volumes avoid ownership conflicts between vLLM root and Diffusers uid 10001.
	args := BuildArgs(p)
	if p.Runtime == "diffusers" {
		for i := range args {
			args[i] = strings.ReplaceAll(args[i], "source=modelship-hf-cache,", "source=modelship-diffusers-cache,")
		}
	}
	if err = docker(ctx, args...); err != nil {
		return fmt.Errorf("start deployment: %w", err)
	}
	fmt.Printf("Container %s started. Downloads resume through the persistent Hugging Face cache.\n", p.Name)
	if wait <= 0 {
		return nil
	}
	if err = WaitReady(ctx, p.Name, p.Port, p.Runtime, wait); err != nil {
		return err
	}
	fmt.Printf("Ready: http://127.0.0.1:%d (model revision %s)\n", p.Port, p.Revision)
	return nil
}
func WaitReady(ctx context.Context, name string, port int, engine string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	path := "/health"
	if engine == "vllm" {
		path = "/v1/models"
	}
	client := &http.Client{Timeout: 5 * time.Second}
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		status, err := output(ctx, "inspect", "--format", "{{.State.Status}}", name)
		if err != nil {
			return fmt.Errorf("cannot inspect deployment; use modelship logs --name %s", name)
		}
		if status == "exited" || status == "dead" || status == "restarting" {
			return fmt.Errorf("container is %s; use modelship logs --name %s", status, name)
		}
		if status == "running" {
			req, _ := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("http://127.0.0.1:%d%s", port, path), nil)
			resp, e := client.Do(req)
			if e == nil {
				io.Copy(io.Discard, io.LimitReader(resp.Body, 1024*1024))
				resp.Body.Close()
				if resp.StatusCode == 200 {
					return nil
				}
			}
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("readiness timed out or interrupted; container is retained; use modelship logs --name %s", name)
		case <-ticker.C:
		}
	}
}
func requireManaged(ctx context.Context, name string) error {
	if len(name) > 63 || !containerName.MatchString(name) {
		return fmt.Errorf("invalid container name")
	}
	value, err := output(ctx, "inspect", "--format", `{{index .Config.Labels "io.modelship.managed"}}`, name)
	if err != nil || value != "true" {
		return fmt.Errorf("container %s is absent or not managed by modelship", name)
	}
	return nil
}
func Manage(ctx context.Context, action, name string) error {
	if err := requireManaged(ctx, name); err != nil {
		return err
	}
	switch action {
	case "logs":
		return docker(ctx, "logs", "--tail", "200", "--follow", name)
	case "stop":
		return docker(ctx, "stop", name)
	case "start":
		// Read only labels/running state, never Config.Env (which may hold tokens).
		state, err := output(ctx, "inspect", "--format", `{"running":{{.State.Running}},"labels":{{json .Config.Labels}}}`, name)
		if err != nil {
			return fmt.Errorf("cannot inspect deployment memory metadata")
		}
		var info struct {
			Running bool              `json:"running"`
			Labels  map[string]string `json:"labels"`
		}
		if err = json.Unmarshal([]byte(state), &info); err != nil {
			return fmt.Errorf("invalid deployment memory metadata")
		}
		if !info.Running && info.Labels["io.modelship.resident-bytes"] != "" {
			resident, err := strconv.ParseInt(info.Labels["io.modelship.resident-bytes"], 10, 64)
			if err != nil || resident <= 0 {
				return fmt.Errorf("invalid deployment weight budget; re-plan and redeploy")
			}
			p := Plan{WeightResidentBytes: resident, GPU: GPU{UUID: info.Labels["io.modelship.gpu-uuid"]}}
			if err = verifyWeightFitNow(ctx, p); err != nil {
				return err
			}
		}
		return docker(ctx, "start", name)
	case "uninstall":
		return docker(ctx, "rm", "--force", name)
	case "status":
		// Do not print inspect's Config.Env: it can contain HF_TOKEN.
		return docker(ctx, "inspect", "--format", `{{.Name}} {{.State.Status}} {{json .NetworkSettings.Ports}}`, name)
	}
	return fmt.Errorf("unknown management action")
}
func Smoke(ctx context.Context, name string) error {
	if err := requireManaged(ctx, name); err != nil {
		return err
	}
	model, err := output(ctx, "inspect", "--format", `{{index .Config.Labels "io.modelship.model"}}`, name)
	if err != nil {
		return err
	}
	ports, err := output(ctx, "inspect", "--format", "{{json .NetworkSettings.Ports}}", name)
	if err != nil {
		return err
	}
	var mapping map[string][]struct {
		HostIP   string
		HostPort string
	}
	if err = json.Unmarshal([]byte(ports), &mapping); err != nil {
		return err
	}
	bindings := mapping["8000/tcp"]
	if len(bindings) != 1 || bindings[0].HostIP != "127.0.0.1" {
		return fmt.Errorf("expected one loopback port binding")
	}
	runtimeImage, err := output(ctx, "inspect", "--format", "{{.Config.Image}}", name)
	if err != nil {
		return err
	}
	if runtimeImage != VLLMImage {
		return fmt.Errorf("image smoke tests require a prompt/input; use the image API examples in README.md")
	}
	body, _ := json.Marshal(map[string]any{"model": model, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
	req, err := http.NewRequestWithContext(ctx, "POST", "http://127.0.0.1:"+bindings[0].HostPort+"/v1/completions", strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 2 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("smoke test returned HTTP %d; inspect container logs", resp.StatusCode)
	}
	_, err = io.Copy(os.Stdout, io.LimitReader(resp.Body, 64*1024))
	fmt.Println()
	return err
}
