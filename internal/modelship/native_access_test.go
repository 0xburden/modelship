package modelship

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

func fixtureAPIKey(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "api.key")
	if err := os.WriteFile(path, []byte(strings.Repeat("a", 43)+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}
func TestAPIKeyFilesPrivateAndBounded(t *testing.T) {
	path := fixtureAPIKey(t)
	if _, err := readAPIKey(path); err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{"short", strings.Repeat("a", 129), strings.Repeat("a", 32) + "\n\n", strings.Repeat("a", 32) + " secret", strings.Repeat("a", 32) + "\x00"} {
		if err := os.WriteFile(path, []byte(value), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := readAPIKey(path); err == nil {
			t.Fatal("accepted invalid key content")
		}
	}
	if err := os.WriteFile(path, []byte(strings.Repeat("a", 43)+"\r\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := readAPIKey(path); err != nil {
		t.Fatal("CRLF compatibility", err)
	}
	if err := os.Chmod(path, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := readAPIKey(path); err == nil {
		t.Fatal("world-readable key accepted")
	}
	link := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	if _, err := readAPIKey(link); err == nil {
		t.Fatal("symlink accepted")
	}
	fifo := filepath.Join(t.TempDir(), "fifo")
	if err := syscall.Mkfifo(fifo, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := readAPIKey(fifo); err == nil {
		t.Fatal("FIFO accepted")
	}
}
func TestNativeAccessPolicy(t *testing.T) {
	for _, p := range []Plan{{Listen: "0.0.0.0"}, {Listen: "localhost"}, {Listen: "::"}, {Listen: "127.0.0.2"}, {APIKeyFile: "relative"}} {
		if validateNativeAccess(p, true) == nil {
			t.Fatal("unsafe access configuration accepted")
		}
	}
	if err := validateNativeAccess(Plan{}, true); err != nil {
		t.Fatal(err)
	}
	path := fixtureAPIKey(t)
	if err := validateNativeAccess(Plan{Listen: "0.0.0.0", APIKeyFile: path}, true); err != nil {
		t.Fatal(err)
	}
	o := optsFixture()
	o.Listen = "0.0.0.0"
	o.APIKeyFile = path
	if _, err := BuildPlan(textFixture(), hostFixture(), o); err == nil {
		t.Fatal("Docker auth flags accepted")
	}
	o.Backend = "native"
	p, err := BuildPlan(textFixture(), hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(p)
	if strings.Contains(string(data), strings.Repeat("a", 43)) {
		t.Fatal("key serialized in plan")
	}
	t.Setenv("MODEL_LISTEN", "0.0.0.0")
	t.Setenv("MODEL_API_KEY_FILE", "untrusted")
	env := strings.Join(nativeEnv(Plan{}, ""), "\n")
	if strings.Contains(env, "untrusted") || !strings.Contains(env, "MODEL_LISTEN=127.0.0.1") {
		t.Fatal("inherited access overrides were not replaced")
	}
}
func TestAuthKeygenNeverOverwrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "private", "key")
	if err := authKeygenCommand([]string{"--file", path}); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = readAPIKey(path); err != nil {
		t.Fatal(err)
	}
	if err = authKeygenCommand([]string{"--file", path}); err == nil {
		t.Fatal("existing key overwritten")
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatal("key changed")
	}
}
func TestAuthenticatedHealthAndRedirectRefusal(t *testing.T) {
	path := fixtureAPIKey(t)
	var sinkCalls atomic.Int32
	sink := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { sinkCalls.Add(1) }))
	defer sink.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, sink.URL, 302) }))
	defer redirect.Close()
	req, _ := http.NewRequest("GET", redirect.URL, nil)
	if err := nativeAuthorize(req, Plan{APIKeyFile: path}); err != nil {
		t.Fatal(err)
	}
	client := nativeClient(time.Second)
	defer client.CloseIdleConnections()
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 302 || sinkCalls.Load() != 0 {
		t.Fatal("credential-bearing redirect followed")
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+strings.Repeat("a", 43) {
			w.WriteHeader(401)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"ready":true,"model":"org/model","revision":"revision","run_id":"run"}`))
	}))
	defer server.Close()
	// Use the same loopback port through the production readiness checker.
	port := server.Listener.Addr().(*net.TCPAddr).Port
	cfg := NativeConfig{Plan: Plan{Port: port, Model: "org/model", Revision: "revision", APIKeyFile: path}}
	ready, msg := nativeHealth(context.Background(), cfg, NativeRun{RunID: "run"})
	if !ready || msg != "" {
		t.Fatal("authenticated readiness failed", msg)
	}
	cfg.Plan.APIKeyFile = ""
	if ready, _ = nativeHealth(context.Background(), cfg, NativeRun{RunID: "run"}); ready {
		t.Fatal("unauthenticated readiness succeeded")
	}
	cfg.Plan.APIKeyFile = path
	if ready, _ = nativeHealth(context.Background(), cfg, NativeRun{RunID: "wrong"}); ready {
		t.Fatal("wrong worker identity accepted")
	}
}
func TestConfigureUpgradesOnlyAccessAndRejectsLegacyPolicy(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir, err := nativeDir("access-test")
	if err != nil {
		t.Fatal(err)
	}
	if err = privateDir(dir); err != nil {
		t.Fatal(err)
	}
	cfg := NativeConfig{Schema: 3, Plan: Plan{Backend: "native", Name: "access-test", Model: "org/model", Revision: strings.Repeat("a", 40), Port: 18032}}
	if err = writeJSONFile(filepath.Join(dir, "config.json"), cfg); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "server.py"), []byte("preserved inference"), 0600); err != nil {
		t.Fatal(err)
	}
	listen, key := "0.0.0.0", fixtureAPIKey(t)
	if err = ConfigureNative(context.Background(), "access-test", &listen, &key); err != nil {
		t.Fatal(err)
	}
	got, _, err := nativeConfig("access-test")
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != 4 || got.Plan.Listen != listen || got.Plan.APIKeyFile != key {
		t.Fatal("access upgrade missing")
	}
	data, _ := os.ReadFile(filepath.Join(dir, "server.py"))
	if string(data) != "preserved inference" {
		t.Fatal("inference module modified")
	}
	for _, name := range []string{"native_entry.py", "api_auth.py"} {
		st, e := os.Stat(filepath.Join(dir, name))
		if e != nil || st.Mode().Perm() != 0600 {
			t.Fatal("missing/private access module", name)
		}
	}
	got.Schema = 3
	if err = writeJSONFile(filepath.Join(dir, "config.json"), got); err != nil {
		t.Fatal(err)
	}
	if _, _, err = nativeConfig("access-test"); err == nil {
		t.Fatal("legacy schema accepted new access policy")
	}
}
