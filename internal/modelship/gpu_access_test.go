package modelship

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestCDIArgsForBothRuntimes(t *testing.T) {
	for _, m := range []Model{textFixture(), imageFixture()} {
		o := optsFixture()
		o.GPUMode = "cdi"
		p, err := BuildPlan(m, hostFixture(), o)
		if err != nil {
			t.Fatal(err)
		}
		args := strings.Join(BuildArgs(p), " ")
		if !strings.Contains(args, "--device nvidia.com/gpu=all") || strings.Contains(args, "--gpus") || strings.Contains(args, "--runtime") {
			t.Fatal(args)
		}
	}
}
func TestCDIRefusesMultipleGPUs(t *testing.T) {
	h := hostFixture()
	h.GPUs = append(h.GPUs, GPU{Index: 1, UUID: "GPU-other"})
	o := optsFixture()
	o.GPUMode = "cdi"
	if _, err := BuildPlan(textFixture(), h, o); err == nil {
		t.Fatal("CDI plan accepted multiple GPUs")
	}
	p, err := BuildPlan(textFixture(), hostFixture(), optsFixture())
	if err != nil {
		t.Fatal(err)
	}
	p.GPUMode = "cdi"
	p.GPUCount = 2
	if _, err = GPUPreflight(context.Background(), p); err == nil {
		t.Fatal("CDI preflight accepted multiple GPUs")
	}
}
func TestThunderAutoFallbackPropagatesToDeployment(t *testing.T) {
	log := fakeDocker(t, `case "$1" in
 context) echo unix:///var/run/docker.sock;;
 inspect) exit 1;;
 run)
  case "$*" in
   *--gpus*) echo 'Thunder Compute uses a special version of docker. Use --device nvidia.com/gpu=all' >&2; exit 1;;
   *--entrypoint*) echo GPU-test;;
   *) echo fixture-container;;
  esac;;
 *) exit 1;;
 esac`)
	p, err := BuildPlan(textFixture(), hostFixture(), optsFixture())
	if err != nil {
		t.Fatal(err)
	}
	if err = Deploy(context.Background(), p, 0); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(log)
	calls := string(b)
	if strings.Count(calls, "--gpus") != 1 {
		t.Fatal("unexpected NVIDIA retries:", calls)
	}
	if strings.Count(calls, "--device nvidia.com/gpu=all") != 2 {
		t.Fatal("CDI not used for both probe and deployment:", calls)
	}
}
func TestExplicitCDINeverUsesGPUsFlag(t *testing.T) {
	log := fakeDocker(t, "echo GPU-test\n")
	o := optsFixture()
	o.GPUMode = "cdi"
	p, err := BuildPlan(textFixture(), hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	result, err := GPUPreflight(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(log)
	if result.GPUMode != "cdi" || strings.Contains(string(b), "--gpus") {
		t.Fatalf("%+v %s", result, b)
	}
}
func TestPreflightDoesNotFallbackForUnrelatedErrors(t *testing.T) {
	log := fakeDocker(t, "echo 'permission denied' >&2; exit 1\n")
	p, err := BuildPlan(textFixture(), hostFixture(), optsFixture())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = GPUPreflight(context.Background(), p); err == nil {
		t.Fatal("accepted failure")
	}
	b, _ := os.ReadFile(log)
	if strings.Contains(string(b), "--device") {
		t.Fatal("retried unrelated failure")
	}
}
func TestThunderDoesNotExposeMultipleGPUs(t *testing.T) {
	log := fakeDocker(t, "echo 'Thunder Compute: --device nvidia.com/gpu=all' >&2; exit 1\n")
	p, err := BuildPlan(textFixture(), hostFixture(), optsFixture())
	if err != nil {
		t.Fatal(err)
	}
	p.GPUCount = 2
	if _, err = GPUPreflight(context.Background(), p); err == nil {
		t.Fatal("exposed multiple GPUs")
	}
	b, _ := os.ReadFile(log)
	if strings.Contains(string(b), "--device") {
		t.Fatal("unsafe CDI retry")
	}
}
func TestPreflightRejectsUnexpectedVisibility(t *testing.T) {
	for _, s := range []string{"echo GPU-other\n", "printf 'GPU-test\\nGPU-other\\n'\n", "echo ''\n"} {
		fakeDocker(t, s)
		p, err := BuildPlan(textFixture(), hostFixture(), optsFixture())
		if err != nil {
			t.Fatal(err)
		}
		if _, err = GPUPreflight(context.Background(), p); err == nil {
			t.Fatal("accepted wrong GPU visibility")
		}
	}
}
func TestForcedNVIDIAModeDoesNotFallback(t *testing.T) {
	log := fakeDocker(t, "echo 'Thunder Compute: --device nvidia.com/gpu=all' >&2; exit 1\n")
	o := optsFixture()
	o.GPUMode = "nvidia"
	p, err := BuildPlan(textFixture(), hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = GPUPreflight(context.Background(), p); err == nil {
		t.Fatal("expected failure")
	}
	b, _ := os.ReadFile(log)
	if strings.Contains(string(b), "--device") {
		t.Fatal("overrode explicit mode")
	}
}
