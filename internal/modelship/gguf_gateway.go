package modelship

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

func llamaEnv(cfg NativeConfig, dir string) []string {
	libs := []string{}
	for _, path := range cfg.RuntimeLibraries {
		libs = append(libs, filepath.Join(dir, path))
	}
	// No HF/API secrets, proxy variables, LLAMA_ARG_* overrides, host preload or
	// broad driver/library path manipulation are inherited by the native engine.
	return []string{"PATH=/usr/local/bin:/usr/bin:/bin", "HOME=" + dir, "TMPDIR=" + dir, "CUDA_VISIBLE_DEVICES=" + cfg.Plan.GPU.UUID, "LD_LIBRARY_PATH=" + strings.Join(libs, ":")}
}
func llamaArgs(p Plan, port int, keyPath string) []string {
	args := []string{"--model", ggufModelPath(p), "--alias", p.Model, "--host", "127.0.0.1", "--port", strconv.Itoa(port), "--api-key-file", keyPath,
		"--device", "CUDA0", "--gpu-layers", "all", "--split-mode", "none", "--main-gpu", "0", "--fit", "off", "--ctx-size", strconv.Itoa(p.Context), "--no-context-shift", "--parallel", "1",
		"--batch-size", "128", "--ubatch-size", "128", "--threads", strconv.Itoa(min(8, max(1, runtime.NumCPU()))), "--cache-type-k", "f16", "--cache-type-v", "f16",
		"--jinja", "--reasoning", "off", "--no-reasoning-preserve", "--spec-type", "none", "--no-mmproj", "--no-webui", "--no-agent", "--no-ui-mcp-proxy"}
	if p.GGUF != nil && p.GGUF.Vision != nil {
		filtered := args[:0]
		for _, v := range args {
			if v != "--no-mmproj" {
				filtered = append(filtered, v)
			}
		}
		args = filtered
		// Both files are verified local paths; no HF repository/URL or implicit
		// projector download options are passed to the private server.
		args = append(args, "--mmproj", ggufArtifactPath(p, p.GGUF.Vision.File.Name), "--mmproj-offload", "--mmproj-device", "CUDA0", "--image-max-tokens", strconv.Itoa(visionImageMaxTokens), "--mtmd-batch-max-tokens", "256")
	}
	return args
}

type ggufMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}
type ggufTextRequest struct {
	Model       string        `json:"model,omitempty"`
	Prompt      *string       `json:"prompt,omitempty"`
	Messages    []ggufMessage `json:"messages,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
	Seed        *int64        `json:"seed,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}
type ggufGateway struct {
	plan                        Plan
	runID, upstream, backendKey string
	ownerDigest                 [32]byte
	authenticated               bool
	ready                       atomic.Bool
	failed                      atomic.Bool
	busy                        chan struct{}
	client                      *http.Client
}

func newGGUFGateway(p Plan, runID, upstream, backendKey string) (*ggufGateway, error) {
	if e := validateNativeAccess(p, true); e != nil {
		return nil, e
	}
	g := &ggufGateway{plan: p, runID: runID, upstream: upstream, backendKey: backendKey, busy: make(chan struct{}, 1), client: nativeClient(15 * time.Minute)}
	if p.APIKeyFile != "" {
		key, e := readAPIKey(p.APIKeyFile)
		if e != nil {
			return nil, e
		}
		g.ownerDigest = sha256.Sum256([]byte(key))
		g.authenticated = true
	}
	return g, nil
}
func (g *ggufGateway) json(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
func (g *ggufGateway) fail(w http.ResponseWriter, status int, message string) {
	g.json(w, status, map[string]string{"detail": message})
}
func (g *ggufGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if g.authenticated {
		headers := r.Header.Values("Authorization")
		valid := false
		if len(headers) == 1 && len(headers[0]) <= 512 {
			scheme, key, ok := strings.Cut(headers[0], " ")
			if ok && strings.EqualFold(scheme, "Bearer") && apiKeyPattern.MatchString(key) {
				digest := sha256.Sum256([]byte(key))
				valid = subtle.ConstantTimeCompare(digest[:], g.ownerDigest[:]) == 1
			}
		}
		if !valid {
			w.Header().Set("WWW-Authenticate", `Bearer realm="modelship"`)
			g.fail(w, 401, "Authentication required")
			return
		}
	}
	switch r.URL.Path {
	case "/health":
		if r.Method != "GET" {
			g.fail(w, 405, "method not allowed")
			return
		}
		status := 200
		if !g.ready.Load() {
			status = 503
		}
		g.json(w, status, map[string]any{"ready": g.ready.Load(), "model": g.plan.Model, "revision": g.plan.Revision, "task": "text-generation", "recipe": g.plan.RecipeID, "runtime": "llama.cpp", "run_id": g.runID, "input_modalities": g.inputModalities()})
	case "/v1/capabilities":
		if r.Method != "GET" {
			g.fail(w, 405, "method not allowed")
			return
		}
		g.json(w, 200, g.capabilities())
	case "/v1/models":
		if r.Method != "GET" {
			g.fail(w, 405, "method not allowed")
			return
		}
		g.json(w, 200, map[string]any{"object": "list", "data": []map[string]any{{"id": g.plan.Model, "object": "model", "owned_by": "modelship", "input_modalities": g.inputModalities()}}})
	case "/v1/chat/completions", "/v1/completions":
		if r.Method != "POST" {
			g.fail(w, 405, "method not allowed")
			return
		}
		g.generate(w, r, r.URL.Path == "/v1/chat/completions")
	default:
		g.fail(w, 404, "route not available on this GGUF chat adapter")
	}
}
func (g *ggufGateway) generate(w http.ResponseWriter, r *http.Request, chat bool) {
	if !g.ready.Load() {
		g.fail(w, 503, "model is not ready")
		return
	}
	select {
	case g.busy <- struct{}{}:
		defer func() { <-g.busy }()
	default:
		g.fail(w, 503, "GPU busy; retry after the current request finishes")
		return
	}
	limit := int64(1 << 20)
	if g.visionEnabled() {
		limit = 8 << 20
	}
	data, e := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if e != nil || int64(len(data)) > limit {
		g.fail(w, 413, "request exceeds the deployment body limit")
		return
	}
	var input ggufTextRequest
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if e = decoder.Decode(&input); e != nil {
		g.fail(w, 422, "invalid text request fields")
		return
	}
	if decoder.Decode(new(any)) != io.EOF {
		g.fail(w, 422, "invalid text request body")
		return
	}
	if input.Model != "" && input.Model != g.plan.Model {
		g.fail(w, 404, "requested model is not loaded")
		return
	}
	tokens, temp, top := 128, 0.0, 1.0
	if input.MaxTokens != nil {
		tokens = *input.MaxTokens
	}
	if input.Temperature != nil {
		temp = *input.Temperature
	}
	if input.TopP != nil {
		top = *input.TopP
	}
	if input.Stream || tokens < 1 || tokens > 1024 || temp < 0 || temp > 2 || top <= 0 || top > 1 || input.Seed != nil && (*input.Seed < 0 || *input.Seed >= 4294967295) {
		g.fail(w, 422, "unsupported streaming or text parameter bounds")
		return
	}
	messages := input.Messages
	if chat {
		if input.Prompt != nil || len(messages) == 0 || len(messages) > 128 {
			g.fail(w, 422, "chat requires bounded text/image messages")
			return
		}
	} else {
		if len(input.Messages) != 0 || input.Prompt == nil || strings.TrimSpace(*input.Prompt) == "" || len(*input.Prompt) > 65536 {
			g.fail(w, 422, "completion requires a text prompt only")
			return
		}
		content, _ := json.Marshal(*input.Prompt)
		messages = []ggufMessage{{Role: "user", Content: content}}
	}
	var normalized []map[string]any
	if chat {
		normalized, e = validateGGUFMessages(messages, g.visionEnabled())
		if e != nil {
			g.fail(w, 422, e.Error())
			return
		}
	} else {
		normalized = []map[string]any{{"role": "user", "content": *input.Prompt}}
	}
	body := map[string]any{"model": g.plan.Model, "messages": normalized, "max_tokens": tokens, "temperature": temp, "top_p": top, "stream": false, "n": 1, "top_k": 0, "min_p": 0, "chat_template_kwargs": map[string]bool{"enable_thinking": false}}
	if input.Seed != nil {
		body["seed"] = *input.Seed
	}
	encoded, _ := json.Marshal(body)
	inferenceCtx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	req, _ := http.NewRequestWithContext(inferenceCtx, "POST", g.upstream+"/v1/chat/completions", strings.NewReader(string(encoded)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+g.backendKey)
	resp, e := g.client.Do(req)
	if e != nil {
		g.failed.Store(true)
		g.ready.Store(false)
		g.fail(w, 502, "private inference backend state is uncertain; restart the service before retrying")
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		status := 502
		if resp.StatusCode == 400 || resp.StatusCode == 422 {
			status = 422
		}
		if resp.StatusCode == 503 {
			g.failed.Store(true)
			g.ready.Store(false)
			status = 503
		}
		g.fail(w, status, "private backend rejected generation; check context/parameters or service logs")
		return
	}
	data, e = io.ReadAll(io.LimitReader(resp.Body, (2<<20)+1))
	if e != nil || len(data) > 2<<20 {
		g.failed.Store(true)
		g.ready.Store(false)
		g.fail(w, 502, "invalid/oversized inference response")
		return
	}
	var value struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if json.Unmarshal(data, &value) != nil || len(value.Choices) != 1 || strings.TrimSpace(value.Choices[0].Message.Content) == "" {
		g.fail(w, 502, "backend returned no usable text")
		return
	}
	choice := map[string]any{"index": 0, "finish_reason": value.Choices[0].FinishReason}
	kind := "text_completion"
	if chat {
		kind = "chat.completion"
		choice["message"] = map[string]string{"role": "assistant", "content": value.Choices[0].Message.Content}
	} else {
		choice["text"] = value.Choices[0].Message.Content
	}
	g.json(w, 200, map[string]any{"id": value.ID, "object": kind, "created": time.Now().Unix(), "model": g.plan.Model, "choices": []any{choice}, "usage": value.Usage})
}
func runGGUFWorker(ctx context.Context, cfg NativeConfig, dir, runID string) error {
	if e := verifyGGUFInstall(cfg, dir); e != nil {
		return e
	}
	if e := validateNativeAccess(cfg.Plan, true); e != nil {
		return e
	}
	if e := verifyWeightFitNow(ctx, cfg.Plan); e != nil {
		return e
	}
	return serveGGUFWorker(ctx, cfg, dir, runID)
}

// serveGGUFWorker is the process/HTTP adapter; the public worker entrypoint must
// complete immutable artifact/access/fresh-GPU preflights above before calling it.
func serveGGUFWorker(ctx context.Context, cfg NativeConfig, dir, runID string) error {
	listener, e := net.Listen("tcp4", fmt.Sprintf("%s:%d", nativeListen(cfg.Plan), cfg.Plan.Port))
	if e != nil {
		return fmt.Errorf("GGUF gateway listen failed")
	}
	defer listener.Close()
	private, e := net.Listen("tcp4", "127.0.0.1:0")
	if e != nil {
		return e
	}
	port := private.Addr().(*net.TCPAddr).Port
	private.Close()
	secret := make([]byte, 32)
	if _, e = rand.Read(secret); e != nil {
		return e
	}
	backendKey := base64.RawURLEncoding.EncodeToString(secret)
	f, e := os.CreateTemp(dir, ".backend-key-")
	if e != nil {
		return e
	}
	keyPath := f.Name()
	defer os.Remove(keyPath)
	if _, e = f.WriteString(backendKey + "\n"); e != nil {
		f.Close()
		return e
	}
	if e = f.Close(); e != nil {
		return e
	}
	binary, e := fileWithin(dir, cfg.RuntimeBinary)
	if e != nil {
		return e
	}
	cmd := exec.Command(binary, llamaArgs(cfg.Plan, port, keyPath)...)
	cmd.Env = llamaEnv(cfg, dir)
	cmd.Dir = dir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if e = cmd.Start(); e != nil {
		return fmt.Errorf("verified llama.cpp runtime failed to start")
	}
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	gateway, e := newGGUFGateway(cfg.Plan, runID, fmt.Sprintf("http://127.0.0.1:%d", port), backendKey)
	if e != nil {
		_ = cmd.Process.Kill()
		<-exited
		return e
	}
	defer gateway.client.CloseIdleConnections()
	server := &http.Server{Handler: gateway, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16384}
	serveError := make(chan error, 1)
	go func() { serveError <- server.Serve(listener) }()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	var result error
	childDone := false
loop:
	for {
		select {
		case <-ctx.Done():
			break loop
		case <-exited:
			childDone = true
			result = fmt.Errorf("private llama.cpp worker exited; inspect private service logs")
			break loop
		case e := <-serveError:
			result = e
			break loop
		case <-ticker.C:
			if !gateway.ready.Load() && !gateway.failed.Load() {
				req, _ := http.NewRequestWithContext(ctx, "GET", gateway.upstream+"/health", nil)
				req.Header.Set("Authorization", "Bearer "+backendKey)
				client := nativeClient(2 * time.Second)
				resp, e := client.Do(req)
				if e == nil {
					if resp.StatusCode == 200 {
						gateway.ready.Store(true)
					}
					resp.Body.Close()
				}
				client.CloseIdleConnections()
			}
		}
	}
	gateway.ready.Store(false)
	shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	_ = server.Shutdown(shutdown)
	cancel()
	_ = server.Close()
	if !childDone {
		_ = cmd.Process.Signal(os.Interrupt)
		select {
		case <-exited:
		case <-time.After(5 * time.Second):
			_ = cmd.Process.Kill()
			<-exited
		}
	}
	return result
}
