package modelship

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == "__native-run" && os.Getenv("MODELSHIP_FAKE_NATIVE_CHILD") == "1" {
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// Fake Python implements only the install/probe command contract. No pip, CUDA,
// model downloads, or host Python modifications are performed by these tests.
func fakeNativePython(t *testing.T, mode string) (string, string) {
	t.Helper()
	fakeNvidiaSMI(t, "0, GPU-test, RTX 4090, 24576, 24000, 575.57.08, 8.9")
	dir := t.TempDir()
	python := filepath.Join(dir, "python")
	mark := filepath.Join(dir, "pip-ran")
	t.Setenv("MODELSHIP_FAKE_PIP_MARK", mark)
	t.Setenv("MODELSHIP_FAKE_NATIVE_CHILD", "1")
	t.Setenv("MODELSHIP_FAKE_TORCH_MODE", mode)
	script := `#!/bin/sh
case "$1" in
 -c)
  case "$0" in
   */venv/bin/python)
    if [ "$MODELSHIP_FAKE_TORCH_MODE" = missing-inherited ]; then exit 1; fi;;
  esac
  if [ "$MODELSHIP_FAKE_TORCH_MODE" = replaced ] && [ -f "$MODELSHIP_FAKE_PIP_MARK" ]; then
   printf '{"torch_version":"9.9","torch_package_version":"9.9","torch_file":"/different/torch.py","cuda_version":"13.0","gpu":"fixture","python_version":"3.12.13"}\n'
  else
   printf '{"torch_version":"2.12.1+cu130","torch_package_version":"2.12.1+cu130","torch_file":"/working/torch/__init__.py","cuda_version":"13.0","gpu":"fixture","python_version":"3.12.13"}\n'
  fi;;
 -m)
  case "$2" in
   venv) mkdir -p "$4/bin"; cp "$0" "$4/bin/python";;
   pip) printf '%s\n' "$*" > "$MODELSHIP_FAKE_PIP_MARK";;
   *) exit 1;;
  esac;;
 *) exit 1;;
esac
`
	if err := os.WriteFile(python, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	return python, mark
}
func installPlan(t *testing.T, python string) Plan {
	t.Helper()
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	o := optsFixture()
	o.Backend = "native"
	o.Port = unusedLoopbackPort(t)
	o.Python = python
	p, err := BuildPlan(textFixture(), hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func TestNativeInstallConstrainsAndPreservesTorch(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("native launch identity requires Linux /proc; exercised in Linux CI")
	}
	python, mark := fakeNativePython(t, "normal")
	p := installPlan(t, python)
	if err := DeployNative(context.Background(), p, 0); err != nil {
		t.Fatal(err)
	}
	cfg, dir, err := nativeConfig(p.Name)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Schema != nativeSchemaVersion || cfg.Torch.Version != "2.12.1+cu130" || cfg.Torch.File != "/working/torch/__init__.py" {
		t.Fatalf("%+v", cfg.Torch)
	}
	b, err := os.ReadFile(mark)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "--constraint "+filepath.Join(dir, "constraints.txt")) {
		t.Fatal(string(b))
	}
	b, _ = os.ReadFile(filepath.Join(dir, "constraints.txt"))
	if string(b) != "torch===2.12.1+cu130\n" {
		t.Fatal(string(b))
	}
	b, _ = os.ReadFile(filepath.Join(dir, "requirements.txt"))
	if strings.Contains(string(b), "torch==") {
		t.Fatal("explicit torch installation requested")
	}
	for _, name := range []string{"native_entry.py", "server.py", "text_server.py", "model_code.py", "video_server.py"} {
		st, err := os.Stat(filepath.Join(dir, name))
		if err != nil || st.Mode().Perm() != 0600 {
			t.Fatal(name, err)
		}
	}
}
func TestExperimentalProfileUsesOwnPinsWithoutChangingTorch(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("native launch identity requires Linux /proc; exercised in Linux CI")
	}
	python, _ := fakeNativePython(t, "normal")
	_ = installPlan(t, python) // Isolate native state.
	o := optsFixture()
	o.Backend = "native"
	o.Port = unusedLoopbackPort(t)
	o.Python = python
	o.Experimental = true
	p, err := BuildPlan(targetFixture(t, "google/gemma-4-E4B-it"), hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	if err = DeployNative(context.Background(), p, 0); err != nil {
		t.Fatal(err)
	}
	dir, _ := nativeDir(p.Name)
	b, _ := os.ReadFile(filepath.Join(dir, "requirements.txt"))
	if !strings.Contains(string(b), "transformers==5.22.2") || strings.Contains(string(b), "transformers==4.56.2") {
		t.Fatal(string(b))
	}
}
func TestExperimentalProfileChecksPythonBeforePip(t *testing.T) {
	python, mark := fakeNativePython(t, "normal")
	b, err := os.ReadFile(python)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(python, []byte(strings.ReplaceAll(string(b), "3.12.13", "3.10.12")), 0755); err != nil {
		t.Fatal(err)
	}
	_ = installPlan(t, python)
	o := optsFixture()
	o.Backend = "native"
	o.Port = unusedLoopbackPort(t)
	o.Python = python
	o.Experimental = true
	p, err := BuildPlan(targetFixture(t, "google/gemma-4-E4B-it"), hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	err = DeployNative(context.Background(), p, 0)
	if err == nil || !strings.Contains(err.Error(), "Python >= 3.11") {
		t.Fatal(err)
	}
	if _, err = os.Stat(mark); !os.IsNotExist(err) {
		t.Fatal("pip ran on an unsupported Python")
	}
}
func TestNativeMissingPythonDoesNotLeaveAnUnmanagedDirectory(t *testing.T) {
	p := installPlan(t, "/definitely/not/a/python")
	if err := DeployNative(context.Background(), p, 0); err == nil {
		t.Fatal("accepted missing interpreter")
	}
	dir, _ := nativeDir(p.Name)
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("left an unusable deployment directory", err)
	}
}
func TestNativeInstallRefusesTorchReplacement(t *testing.T) {
	python, _ := fakeNativePython(t, "replaced")
	p := installPlan(t, python)
	err := DeployNative(context.Background(), p, 0)
	if err == nil || !strings.Contains(err.Error(), "changed the PyTorch import") {
		t.Fatal(err)
	}
	dir, _ := nativeDir(p.Name)
	if _, err = os.Stat(filepath.Join(dir, "run.json")); !os.IsNotExist(err) {
		t.Fatal("started worker after torch replacement")
	}
}
func TestNativeInstallDoesNotAskPipForMissingInheritedTorch(t *testing.T) {
	python, mark := fakeNativePython(t, "missing-inherited")
	p := installPlan(t, python)
	err := DeployNative(context.Background(), p, 0)
	if err == nil || !strings.Contains(err.Error(), "cannot use the base PyTorch") {
		t.Fatal(err)
	}
	if _, err = os.Stat(mark); !os.IsNotExist(err) {
		t.Fatal("ran pip despite missing inherited PyTorch")
	}
}

func TestRecipeTorchMinimumFailsBeforePipWithoutReplacingHost(t *testing.T) {
	python, mark := fakeNativePython(t, "normal")
	p := installPlan(t, python)
	p.MinTorch = "99.0"
	err := DeployNative(context.Background(), p, 0)
	if err == nil || !strings.Contains(err.Error(), "PyTorch >= 99.0") || !strings.Contains(err.Error(), "will not replace") {
		t.Fatal(err)
	}
	if _, err := os.Stat(mark); !os.IsNotExist(err) {
		t.Fatal("pip ran despite incompatible host Torch")
	}
}

func unusedLoopbackPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return port
}
