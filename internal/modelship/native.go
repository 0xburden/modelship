package modelship

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type TorchInfo struct {
	Version        string `json:"torch_version"`
	PackageVersion string `json:"torch_package_version"`
	File           string `json:"torch_file"`
	CUDA           string `json:"cuda_version"`
	GPU            string `json:"gpu"`
	PythonVersion  string `json:"python_version,omitempty"`
}

// Schema 2 records recipe-specific execution. Older CLIs reject it rather
// than silently starting a modern model with legacy loader/dtype defaults.
const nativeSchemaVersion = 2

type NativeConfig struct {
	Schema int       `json:"schema"`
	Plan   Plan      `json:"plan"`
	Torch  TorchInfo `json:"torch"`
}
type NativeRun struct {
	PID        int    `json:"pid"`
	StartTicks string `json:"start_ticks"`
	RunID      string `json:"run_id"`
}

var runIDPattern = regexp.MustCompile(`^[a-f0-9]{32}$`)
var packageVersionPattern = regexp.MustCompile(`^[A-Za-z0-9.!+_-]+$`)

func nativeDir(name string) (string, error) {
	if len(name) > 63 || !containerName.MatchString(name) {
		return "", fmt.Errorf("invalid deployment name")
	}
	base := os.Getenv("XDG_STATE_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		base = filepath.Join(home, ".local", "state")
	}
	if !filepath.IsAbs(base) {
		return "", fmt.Errorf("XDG_STATE_HOME must be absolute")
	}
	return filepath.Join(base, "modelship", "native", name), nil
}
func privateDir(path string) error {
	if err := os.MkdirAll(path, 0700); err != nil {
		return err
	}
	st, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !st.IsDir() || st.Mode()&os.ModeSymlink != 0 || st.Mode().Perm()&0077 != 0 {
		return fmt.Errorf("state directory must be a real, private directory (0700): %s", path)
	}
	return nil
}
func writeJSONFile(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".state-")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err = f.Write(b); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
func readJSONFile(path string, v any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewDecoder(io.LimitReader(f, 1024*1024)).Decode(v)
}
func nativeConfig(name string) (NativeConfig, string, error) {
	var cfg NativeConfig
	dir, err := nativeDir(name)
	if err != nil {
		return cfg, "", err
	}
	st, err := os.Lstat(dir)
	if err != nil || !st.IsDir() || st.Mode().Perm()&0077 != 0 {
		return cfg, dir, fmt.Errorf("native deployment %s is absent or its state directory is not private", name)
	}
	if err = readJSONFile(filepath.Join(dir, "config.json"), &cfg); err != nil {
		return cfg, dir, fmt.Errorf("read native deployment %s: %w", name, err)
	}
	if (cfg.Schema != 1 && cfg.Schema != nativeSchemaVersion) || cfg.Plan.Backend != "native" || cfg.Plan.Name != name || !commitPattern.MatchString(cfg.Plan.Revision) || cfg.Plan.Port < 1024 || cfg.Plan.Port > 65535 {
		return cfg, dir, fmt.Errorf("invalid native deployment metadata")
	}
	return cfg, dir, nil
}
func nativeLock(dir string) (func(), error) {
	f, err := os.OpenFile(filepath.Join(dir, "operation.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err = syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		return nil, fmt.Errorf("another lifecycle operation is active for this deployment")
	}
	return func() { syscall.Flock(int(f.Fd()), syscall.LOCK_UN); f.Close() }, nil
}
func nativeEnv(p Plan, runID string) []string {
	dtype, loader := p.DType, p.NativeLoader
	if dtype == "" {
		dtype = "float16"
	}
	if loader == "" {
		loader = "causal-lm"
	}
	overrides := map[string]string{
		"MODEL_DTYPE": dtype, "MODEL_NATIVE_LOADER": loader, "MODEL_RECIPE": p.RecipeID,
		"MODEL_TEMPLATE_COMPLETIONS": strconv.FormatBool(p.TemplateCompletions), "MODEL_DISABLE_THINKING": strconv.FormatBool(p.DisableThinking),
		"CUDA_VISIBLE_DEVICES": p.GPU.UUID, "MODEL_ID": p.Model, "MODEL_REVISION": p.Revision,
		"MODEL_TASK": p.Task, "MODEL_PIPELINE": p.Pipeline, "MODEL_CONTEXT": strconv.Itoa(p.Context),
		"MODEL_PORT": strconv.Itoa(p.Port), "MODEL_RUN_ID": runID, "PYTHONUNBUFFERED": "1",
		"HF_HUB_DISABLE_TELEMETRY": "1",
	}
	var env []string
	for _, item := range os.Environ() {
		key, _, _ := strings.Cut(item, "=")
		if _, ok := overrides[key]; !ok {
			env = append(env, item)
		}
	}
	for k, v := range overrides {
		env = append(env, k+"="+v)
	}
	return env
}
func probeTorch(ctx context.Context, python string, p Plan) (TorchInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	var info TorchInfo
	script, err := assets.ReadFile("assets/cuda_probe.py")
	if err != nil {
		return info, err
	}
	cmd := exec.CommandContext(ctx, python, "-c", string(script))
	cmd.Env = nativeEnv(p, "")
	cmd.Stderr = os.Stderr
	b, err := cmd.Output()
	if err != nil {
		return info, fmt.Errorf("native PyTorch CUDA preflight failed using %s; choose --python pointing to your working CUDA-enabled environment: %w", python, err)
	}
	if err = json.Unmarshal(b, &info); err != nil {
		return info, fmt.Errorf("invalid CUDA preflight response: %w", err)
	}
	if info.File == "" || !packageVersionPattern.MatchString(info.PackageVersion) {
		return info, fmt.Errorf("invalid PyTorch package metadata")
	}
	return info, nil
}
func DeployNative(ctx context.Context, p Plan, wait time.Duration) error {
	if len(p.Blockers) > 0 {
		return fmt.Errorf("deployment blocked: %s", strings.Join(p.Blockers, "; "))
	}
	python, err := exec.LookPath(p.Python)
	if err != nil {
		return fmt.Errorf("base Python not found: %w", err)
	}
	python, err = filepath.Abs(python)
	if err != nil {
		return err
	}
	// Preserve a chosen venv path rather than resolving its interpreter symlink.
	p.Python = python
	dir, err := nativeDir(p.Name)
	if err != nil {
		return err
	}
	if err = privateDir(filepath.Dir(dir)); err != nil {
		return err
	}
	if err = os.Mkdir(dir, 0700); err != nil {
		return fmt.Errorf("native deployment %s already exists or cannot be created; use a new --name or uninstall explicitly: %w", p.Name, err)
	}
	// Keep failed installations for inspection/uninstall; never delete user caches.
	unlock, err := nativeLock(dir)
	if err != nil {
		return err
	}
	defer unlock()
	cfg := NativeConfig{Schema: nativeSchemaVersion, Plan: p}
	if err = writeJSONFile(filepath.Join(dir, "config.json"), cfg); err != nil {
		return err
	}
	before, err := probeTorch(ctx, python, p)
	if err != nil {
		return err
	}
	if p.MinPython != "" {
		var major, minor int
		if _, err := fmt.Sscanf(p.MinPython, "%d.%d", &major, &minor); err != nil || !driverAtLeast(before.PythonVersion, major, minor) {
			return fmt.Errorf("recipe %s requires Python >= %s; selected Python reports %q", p.RecipeID, p.MinPython, before.PythonVersion)
		}
	}
	fmt.Printf("Native PyTorch: %s · CUDA %s · %s\n", before.Version, before.CUDA, before.GPU)
	cfg.Torch = before
	if err = writeJSONFile(filepath.Join(dir, "config.json"), cfg); err != nil {
		return err
	}
	for _, name := range []string{"native_entry.py", "server.py", "text_server.py"} {
		b, e := assets.ReadFile("assets/" + name)
		if e != nil {
			return e
		}
		if e = os.WriteFile(filepath.Join(dir, name), b, 0600); e != nil {
			return e
		}
	}
	venv := filepath.Join(dir, "venv")
	if err = runNativeInstall(ctx, python, "-m", "venv", "--system-site-packages", venv); err != nil {
		return fmt.Errorf("create native venv (your OS may need python3-venv): %w", err)
	}
	// Verify the new venv can already see the exact working PyTorch. Do this
	// before pip so a missing inherited torch can never trigger a CUDA wheel install.
	venvPython := filepath.Join(venv, "bin", "python")
	inherited, err := probeTorch(ctx, venvPython, p)
	if err != nil {
		return fmt.Errorf("new system-site-packages venv cannot use the base PyTorch; select a system Python with working CUDA PyTorch (a parent venv-only torch is not inherited): %w", err)
	}
	if inherited.File != before.File || inherited.Version != before.Version || inherited.PackageVersion != before.PackageVersion {
		return fmt.Errorf("new venv did not inherit the selected base PyTorch; refusing dependency installation")
	}
	constraint := filepath.Join(dir, "constraints.txt")
	if err = os.WriteFile(constraint, []byte("torch==="+before.PackageVersion+"\n"), 0600); err != nil {
		return err
	}
	requirements, err := nativeRequirements(p)
	if err != nil {
		return err
	}
	requirementsPath := filepath.Join(dir, "requirements.txt")
	if err = os.WriteFile(requirementsPath, []byte(strings.Join(requirements, "\n")+"\n"), 0600); err != nil {
		return err
	}
	args := []string{"-m", "pip", "install", "--disable-pip-version-check", "--constraint", constraint, "--requirement", requirementsPath}
	if err = runNativeInstall(ctx, venvPython, args...); err != nil {
		return fmt.Errorf("install native service dependencies (base PyTorch is constrained): %w", err)
	}
	after, err := probeTorch(ctx, venvPython, p)
	if err != nil {
		return err
	}
	if after.File != before.File || after.Version != before.Version || after.PackageVersion != before.PackageVersion {
		return fmt.Errorf("native dependency installation changed the PyTorch import; refusing to start")
	}
	return startNative(ctx, cfg, dir, wait)
}
func runNativeInstall(ctx context.Context, python string, args ...string) error {
	cmd := exec.CommandContext(ctx, python, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
func startNative(ctx context.Context, cfg NativeConfig, dir string, wait time.Duration) error {
	old, err := loadNativeRun(dir)
	if err == nil && nativeAlive(dir, old) {
		return fmt.Errorf("native deployment already running (PID %d)", old.PID)
	}
	listener, err := net.Listen("tcp4", fmt.Sprintf("127.0.0.1:%d", cfg.Plan.Port))
	if err != nil {
		return fmt.Errorf("native localhost port unavailable: %w", err)
	}
	listener.Close()
	id := make([]byte, 16)
	if _, err = rand.Read(id); err != nil {
		return err
	}
	runID := hex.EncodeToString(id)
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	log, err := os.OpenFile(filepath.Join(dir, "service.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer log.Close()
	cmd := exec.Command(executable, "__native-run", cfg.Plan.Name, runID)
	cmd.Env = os.Environ()
	cmd.Dir = dir
	cmd.Stdout = log
	cmd.Stderr = log
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err = cmd.Start(); err != nil {
		return err
	}
	pid := cmd.Process.Pid
	ticks, _, err := processIdentity(pid)
	if err != nil {
		cmd.Process.Kill()
		cmd.Wait()
		return fmt.Errorf("native worker exited during startup; see %s/service.log", dir)
	}
	run := NativeRun{PID: pid, StartTicks: ticks, RunID: runID}
	if err = writeJSONFile(filepath.Join(dir, "run.json"), run); err != nil {
		cmd.Process.Kill()
		cmd.Wait()
		return err
	}
	go func() { _ = cmd.Wait() }() // Reap if it exits while the CLI is alive; session survives CLI exit.
	fmt.Printf("Native deployment %s started (PID %d). Logs: %s/service.log\n", cfg.Plan.Name, pid, dir)
	if wait <= 0 {
		return nil
	}
	if err = waitNative(ctx, cfg, dir, run, wait); err != nil {
		return err
	}
	fmt.Printf("Ready: http://127.0.0.1:%d (native %s, revision %s)\n", cfg.Plan.Port, cfg.Plan.Runtime, cfg.Plan.Revision)
	return nil
}
func NativeExec(name, runID string) error {
	if !runIDPattern.MatchString(runID) {
		return fmt.Errorf("invalid native run ID")
	}
	cfg, dir, err := nativeConfig(name)
	if err != nil {
		return err
	}
	python := filepath.Join(dir, "venv", "bin", "python")
	return syscall.Exec(python, []string{python, "-u", filepath.Join(dir, "native_entry.py"), "--run-id", runID}, nativeEnv(cfg.Plan, runID))
}
func processIdentity(pid int) (string, string, error) {
	if pid <= 1 {
		return "", "", fmt.Errorf("invalid process PID")
	}
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return "", "", err
	}
	return parseProcessStat(string(b))
}
func parseProcessStat(stat string) (string, string, error) {
	end := strings.LastIndex(stat, ")")
	if end < 0 {
		return "", "", fmt.Errorf("malformed process stat")
	}
	fields := strings.Fields(stat[end+1:])
	if len(fields) < 20 {
		return "", "", fmt.Errorf("short process stat")
	}
	if _, err := strconv.ParseUint(fields[19], 10, 64); err != nil {
		return "", "", err
	}
	return fields[19], fields[0], nil
}
func loadNativeRun(dir string) (NativeRun, error) {
	var run NativeRun
	err := readJSONFile(filepath.Join(dir, "run.json"), &run)
	if err == nil && (!runIDPattern.MatchString(run.RunID) || run.PID <= 1 || run.StartTicks == "") {
		err = fmt.Errorf("invalid native process identity")
	}
	return run, err
}
func nativeAlive(dir string, run NativeRun) bool {
	ticks, state, err := processIdentity(run.PID)
	if err != nil || state == "Z" || state == "X" || ticks != run.StartTicks {
		return false
	}
	pgid, err := syscall.Getpgid(run.PID)
	if err != nil || pgid != run.PID {
		return false
	}
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", run.PID))
	if err != nil {
		return false
	}
	args := strings.Split(strings.TrimRight(string(b), "\x00"), "\x00")
	has := func(want string) bool {
		for _, arg := range args {
			if arg == want {
				return true
			}
		}
		return false
	}
	return has(run.RunID) && (has(filepath.Join(dir, "native_entry.py")) || has("__native-run"))
}
func waitNative(ctx context.Context, cfg NativeConfig, dir string, run NativeRun, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	client := &http.Client{Timeout: 3 * time.Second}
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		if !nativeAlive(dir, run) {
			return fmt.Errorf("native worker exited; use modelship logs --backend native --name %s", cfg.Plan.Name)
		}
		req, _ := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("http://127.0.0.1:%d/health", cfg.Plan.Port), nil)
		if resp, err := client.Do(req); err == nil {
			var health struct {
				Ready    bool   `json:"ready"`
				Model    string `json:"model"`
				Revision string `json:"revision"`
				RunID    string `json:"run_id"`
			}
			decodeErr := json.NewDecoder(io.LimitReader(resp.Body, 65536)).Decode(&health)
			resp.Body.Close()
			if resp.StatusCode == 200 && decodeErr == nil && health.Ready && health.Model == cfg.Plan.Model && health.Revision == cfg.Plan.Revision && health.RunID == run.RunID {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("native readiness timed out/interrupted; worker retained; use modelship logs --backend native --name %s", cfg.Plan.Name)
		case <-ticker.C:
		}
	}
}
func stopNative(ctx context.Context, dir string) error {
	run, err := loadNativeRun(dir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if !nativeAlive(dir, run) {
		return nil
	}
	// A fresh start-time and command-line check guards against stale/reused PIDs.
	if err = syscall.Kill(-run.PID, syscall.SIGTERM); err != nil && err != syscall.ESRCH {
		return err
	}
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	timeout := time.NewTimer(20 * time.Second)
	defer timeout.Stop()
	forced := false
	for nativeAlive(dir, run) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timeout.C:
			if !nativeAlive(dir, run) {
				return nil
			}
			if forced {
				return fmt.Errorf("native worker did not exit after SIGKILL; state retained for inspection")
			}
			if err = syscall.Kill(-run.PID, syscall.SIGKILL); err != nil && err != syscall.ESRCH {
				return err
			}
			forced = true
			timeout.Reset(5 * time.Second)
		case <-ticker.C:
		}
	}
	return nil
}
func ManageNative(ctx context.Context, action, name string) error {
	cfg, dir, err := nativeConfig(name)
	if err != nil {
		return err
	}
	if action == "status" {
		run, e := loadNativeRun(dir)
		state := "stopped"
		pid := 0
		if e == nil && nativeAlive(dir, run) {
			state = "running"
			pid = run.PID
		}
		return printJSON(map[string]any{"backend": "native", "name": name, "state": state, "pid": pid, "model": cfg.Plan.Model, "revision": cfg.Plan.Revision, "endpoint": fmt.Sprintf("http://127.0.0.1:%d", cfg.Plan.Port), "log": filepath.Join(dir, "service.log"), "torch": cfg.Torch})
	}
	if action == "logs" {
		cmd := exec.CommandContext(ctx, "tail", "-n", "200", "-f", filepath.Join(dir, "service.log"))
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		return cmd.Run()
	}
	unlock, err := nativeLock(dir)
	if err != nil {
		return err
	}
	defer unlock()
	switch action {
	case "start":
		return startNative(ctx, cfg, dir, 30*time.Minute)
	case "stop":
		return stopNative(ctx, dir)
	case "uninstall":
		if err = stopNative(ctx, dir); err != nil {
			return err
		}
		return os.RemoveAll(dir)
	default:
		return fmt.Errorf("unknown native lifecycle action")
	}
}
func SmokeNative(ctx context.Context, name string) error {
	cfg, dir, err := nativeConfig(name)
	if err != nil {
		return err
	}
	if cfg.Plan.Task != "text-generation" {
		return fmt.Errorf("image smoke tests require a prompt/input; use the README image API examples")
	}
	run, err := loadNativeRun(dir)
	if err != nil || !nativeAlive(dir, run) {
		return fmt.Errorf("native worker is not running")
	}
	body, _ := json.Marshal(map[string]any{"model": cfg.Plan.Model, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
	req, _ := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("http://127.0.0.1:%d/v1/completions", cfg.Plan.Port), strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 2 * time.Minute}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("native smoke returned HTTP %d; inspect service logs", resp.StatusCode)
	}
	_, err = io.Copy(os.Stdout, io.LimitReader(resp.Body, 65536))
	fmt.Println()
	return err
}
