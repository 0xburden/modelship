package modelship

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestNativePlanDoesNotRequireDockerOrPinnedDriver(t *testing.T) {
	h := hostFixture()
	h.Docker = false
	h.DockerEndpoint = "tcp://unrelated:2375"
	h.GPUs[0].Driver = "550.54.15"
	for _, m := range []Model{textFixture(), imageFixture()} {
		o := optsFixture()
		o.Backend = "native"
		o.Python = "/chosen/venv/bin/python"
		p, err := BuildPlan(m, h, o)
		if err != nil {
			t.Fatal(err)
		}
		if len(p.Blockers) > 0 || p.Image != "" || p.Backend != "native" || p.GPUMode != "native" || p.Python != o.Python {
			t.Fatalf("%+v", p)
		}
		if m.Index.Class == "" && p.Runtime != "transformers" {
			t.Fatal(p.Runtime)
		}
	}
	o := optsFixture()
	o.Backend = "native"
	o.GPUMode = "cdi"
	if _, err := BuildPlan(textFixture(), h, o); err == nil {
		t.Fatal("accepted Docker GPU mode for native backend")
	}
}
func TestNativeEnvReplacesRatherThanDuplicates(t *testing.T) {
	t.Setenv("CUDA_VISIBLE_DEVICES", "all")
	t.Setenv("MODEL_PORT", "9999")
	t.Setenv("HF_TOKEN", "private-token")
	o := optsFixture()
	o.Backend = "native"
	p, err := BuildPlan(textFixture(), hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	env := nativeEnv(p, "abc")
	joined := strings.Join(env, "\n")
	if strings.Count(joined, "CUDA_VISIBLE_DEVICES=") != 1 || !strings.Contains(joined, "CUDA_VISIBLE_DEVICES=GPU-test") || strings.Contains(joined, "MODEL_PORT=9999") {
		t.Fatal("duplicate/incorrect environment")
	}
	if !strings.Contains(joined, "HF_TOKEN=private-token") {
		t.Fatal("token not forwarded")
	}
}
func TestParseProcessStatWithParentheses(t *testing.T) {
	// After the command field: state is field 3 and start ticks are field 22.
	stat := "123 (name (with) spaces) S " + strings.Repeat("0 ", 18) + "456 0 0"
	ticks, state, err := parseProcessStat(stat)
	if err != nil || ticks != "456" || state != "S" {
		t.Fatalf("%q %q %v", ticks, state, err)
	}
	for _, s := range []string{"garbage", "123 (name) S 0", "123 (name) S " + strings.Repeat("0 ", 18) + "invalid"} {
		if _, _, err = parseProcessStat(s); err == nil {
			t.Fatal("accepted malformed stat")
		}
	}
}
func TestNativeStateAndLocks(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir, err := nativeDir("example")
	if err != nil {
		t.Fatal(err)
	}
	if err = privateDir(dir); err != nil {
		t.Fatal(err)
	}
	unlock, err := nativeLock(dir)
	if err != nil {
		t.Fatal(err)
	}
	if second, err := nativeLock(dir); err == nil {
		second()
		t.Fatal("concurrent operation accepted")
	}
	unlock()
	path := filepath.Join(dir, "secret.json")
	if err = writeJSONFile(path, map[string]string{"value": "hello"}); err != nil {
		t.Fatal(err)
	}
	st, _ := os.Stat(path)
	if st.Mode().Perm() != 0600 {
		t.Fatal(st.Mode())
	}
	var got map[string]string
	if err = readJSONFile(path, &got); err != nil || got["value"] != "hello" {
		t.Fatal(got, err)
	}
	if _, err = nativeDir("../unrelated"); err == nil {
		t.Fatal("accepted traversal")
	}
	t.Setenv("XDG_STATE_HOME", "relative")
	if _, err = nativeDir("example"); err == nil {
		t.Fatal("accepted relative state root")
	}
}
func TestNativeHelperProcess(t *testing.T) {
	if os.Getenv("MODELSHIP_NATIVE_HELPER") != "1" {
		return
	}
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGTERM, os.Interrupt)
	<-ch
	os.Exit(0)
}
func helperNativeRun(t *testing.T, dir string) NativeRun {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("native process identity uses Linux /proc")
	}
	runID := strings.Repeat("b", 32)
	cmd := exec.Command(os.Args[0], "-test.run=^TestNativeHelperProcess$", "--", filepath.Join(dir, "native_entry.py"), runID)
	cmd.Env = append(os.Environ(), "MODELSHIP_NATIVE_HELPER=1")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); _ = cmd.Wait() })
	ticks, _, err := processIdentity(cmd.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	run := NativeRun{PID: cmd.Process.Pid, StartTicks: ticks, RunID: runID}
	if err = writeJSONFile(filepath.Join(dir, "run.json"), run); err != nil {
		t.Fatal(err)
	}
	return run
}
func TestNativeProcessIdentityRejectsStalePIDAndStopsOwnedWorker(t *testing.T) {
	dir := t.TempDir()
	run := helperNativeRun(t, dir)
	if !nativeAlive(dir, run) {
		t.Fatal("owned helper not recognized")
	}
	stale := run
	stale.StartTicks = "0"
	if nativeAlive(dir, stale) {
		t.Fatal("accepted stale PID start time")
	}
	stale = run
	stale.RunID = strings.Repeat("a", 32)
	if nativeAlive(dir, stale) {
		t.Fatal("accepted unrelated process command line")
	}
	// Give helper a chance to install its handler; SIGTERM default termination is
	// also correct, so no readiness sleep is required.
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := stopNative(ctx, dir); err != nil {
		t.Fatal(err)
	}
	if nativeAlive(dir, run) {
		t.Fatal("worker remained alive")
	}
}
func TestNativeReadinessChecksDeploymentIdentity(t *testing.T) {
	dir := t.TempDir()
	run := helperNativeRun(t, dir)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Error(r.URL.Path)
		}
		fmt.Fprintf(w, `{"ready":true,"model":"org/model","revision":"%s","run_id":"%s"}`, strings.Repeat("a", 40), run.RunID)
	}))
	defer srv.Close()
	_, portText, _ := net.SplitHostPort(srv.Listener.Addr().String())
	port, _ := strconv.Atoi(portText)
	cfg := NativeConfig{Plan: Plan{Name: "test", Model: "org/model", Revision: strings.Repeat("a", 40), Port: port}}
	if err := waitNative(context.Background(), cfg, dir, run, time.Second); err != nil {
		t.Fatal(err)
	}
	cfg.Plan.Revision = strings.Repeat("c", 40)
	if err := waitNative(context.Background(), cfg, dir, run, 50*time.Millisecond); err == nil {
		t.Fatal("accepted another deployment's readiness")
	}
}
func TestNativeUninstallLeavesExternalCacheAlone(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	cache := t.TempDir()
	sentinel := filepath.Join(cache, "weights")
	os.WriteFile(sentinel, []byte("keep"), 0600)
	t.Setenv("HF_HOME", cache)
	o := optsFixture()
	o.Backend = "native"
	p, err := BuildPlan(textFixture(), hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	dir, _ := nativeDir(p.Name)
	if err = privateDir(dir); err != nil {
		t.Fatal(err)
	}
	if err = writeJSONFile(filepath.Join(dir, "config.json"), NativeConfig{Schema: 1, Plan: p}); err != nil {
		t.Fatal(err)
	}
	if err = ManageNative(context.Background(), "uninstall", p.Name); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("deployment directory retained", err)
	}
	if _, err = os.Stat(sentinel); err != nil {
		t.Fatal("external cache removed", err)
	}
}
