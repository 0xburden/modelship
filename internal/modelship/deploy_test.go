package modelship

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func fakeDocker(t *testing.T, script string) string {
	t.Helper()
	fakeNvidiaSMI(t, "0, GPU-test, RTX 4090, 24576, 24000, 575.57.08, 8.9")
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	t.Setenv("DOCKER_LOG", log)
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("DOCKER_HOST", "")
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$DOCKER_LOG\"\n"+script), 0755); err != nil {
		t.Fatal(err)
	}
	return log
}
func TestDeployTextUsesPreflightAndPinnedRevision(t *testing.T) {
	log := fakeDocker(t, `case "$1" in
 context) echo unix:///var/run/docker.sock;;
 inspect) exit 1;;
 run) case "$*" in *--entrypoint*) echo GPU-test;; *) echo fixture-container;; esac;;
 *) exit 1;;
 esac
 `)
	p, err := BuildPlan(textFixture(), hostFixture(), optsFixture())
	if err != nil {
		t.Fatal(err)
	}
	if err = Deploy(context.Background(), p, 0); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if !strings.Contains(s, "--entrypoint nvidia-smi") || !strings.Contains(s, "--revision "+p.Revision) || !strings.Contains(s, "--detach --name modelship") {
		t.Fatal(s)
	}
}
func TestDeployRefusesExistingOrRemote(t *testing.T) {
	for _, tc := range []struct{ name, script string }{
		{"existing", `case "$1" in context) echo unix:///var/run/docker.sock;; inspect) echo existing;; *) exit 1;; esac`},
		{"remote", `echo tcp://remote:2375`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			log := fakeDocker(t, tc.script)
			p, err := BuildPlan(textFixture(), hostFixture(), optsFixture())
			if err != nil {
				t.Fatal(err)
			}
			if err = Deploy(context.Background(), p, 0); err == nil {
				t.Fatal("expected refusal")
			}
			b, _ := os.ReadFile(log)
			if strings.Contains(string(b), "run ") {
				t.Fatal("started a container before refusing")
			}
		})
	}
}
func TestManageRefusesUnlabeledContainers(t *testing.T) {
	log := fakeDocker(t, "echo false\n")
	if err := Manage(context.Background(), "uninstall", "unrelated"); err == nil {
		t.Fatal("accepted unrelated container")
	}
	b, _ := os.ReadFile(log)
	if strings.Contains(string(b), "rm ") {
		t.Fatal("removed unrelated container")
	}
	if err := Manage(context.Background(), "stop", "--privileged"); err == nil {
		t.Fatal("accepted flag injection")
	}
}
func TestManageExplicitRemovalRetainsVolumes(t *testing.T) {
	log := fakeDocker(t, "if [ \"$1\" = inspect ]; then echo true; fi\n")
	if err := Manage(context.Background(), "uninstall", "modelship"); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(log)
	s := string(b)
	if !strings.Contains(s, "rm --force modelship") || strings.Contains(s, "volume rm") {
		t.Fatal(s)
	}
}
func TestWaitReady(t *testing.T) {
	fakeDocker(t, "echo running\n")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Error(r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	_, portText, _ := net.SplitHostPort(srv.Listener.Addr().String())
	port, _ := strconv.Atoi(portText)
	if err := WaitReady(context.Background(), "modelship", port, "vllm", time.Second); err != nil {
		t.Fatal(err)
	}
}
func TestWaitReadyReportsCrash(t *testing.T) {
	fakeDocker(t, "echo restarting\n")
	err := WaitReady(context.Background(), "modelship", 8000, "vllm", time.Second)
	if err == nil || !strings.Contains(err.Error(), "restarting") {
		t.Fatal(err)
	}
}
func TestBudgetAccountsForExistingGPUUsage(t *testing.T) {
	h := hostFixture()
	h.GPUs[0].FreeMiB = 12000
	p, err := BuildPlan(textFixture(), h, optsFixture())
	if err != nil {
		t.Fatal(err)
	}
	if p.GPUMemoryFraction < 0.43 || p.GPUMemoryFraction > 0.45 {
		t.Fatal(p.GPUMemoryFraction)
	}
	args := strings.Join(BuildArgs(p), " ")
	if !strings.Contains(args, "--gpu-memory-utilization 0.4394") {
		t.Fatal(args)
	}
}
func TestCLIErrorsDoNotDeploy(t *testing.T) {
	for _, args := range [][]string{
		{"no-such-command"}, {"deploy"}, {"plan", "org/model", "--task", "video"},
		{"deploy", "org/model", "--json"}, {"deploy", "org/model", "--wait=-1s"},
		{"uninstall"}, {"status", "extra"},
	} {
		if err := Run(args, "test"); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
