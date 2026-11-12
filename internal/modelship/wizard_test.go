package modelship

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func testWizardDeps(t *testing.T, m Model) (wizardDeps, *int, *Plan) {
	t.Helper()
	calls := 0
	var got Plan
	d := wizardDeps{
		inspect: func(_ context.Context, id, revision string) (Model, error) { m.Hub.ID = id; return m, nil },
		detect:  func(context.Context, string) (Host, error) { return hostFixture(), nil },
		cache:   func(*Host, Model) {}, build: BuildPlan,
		probe: func(context.Context, string, Plan) (TorchInfo, error) {
			return TorchInfo{Version: "2.12.1+cu130", PythonVersion: "3.12.13", CUDA: "13.0"}, nil
		},
		createKey:  func(path string) error { _, err := generateAPIKey(path); return err },
		deploy:     func(_ context.Context, p Plan, _ time.Duration) error { calls++; got = p; return nil },
		executable: "/home/test/.local/bin/modelship",
	}
	return d, &calls, &got
}
func textWizardInput(final string) string {
	// model, revision, backend, name, port, context, python, access, key, wait, final.
	return strings.Join([]string{"https://huggingface.co/org/model", "", "", "", "", "", "", "", "", "", final}, "\n") + "\n"
}
func TestWizardDeploysExactlyReviewedPinnedPlan(t *testing.T) {
	d, calls, got := testWizardDeps(t, textFixture())
	var output bytes.Buffer
	t.Setenv("HF_TOKEN", "fixture-private-hub-credential")
	if err := runWizard(context.Background(), strings.NewReader(textWizardInput("y")), &output, d); err != nil {
		t.Fatal(err, output.String())
	}
	if *calls != 1 || got.Revision != strings.Repeat("a", 40) || got.Model != "org/model" || got.Listen != "127.0.0.1" {
		t.Fatal("wrong reviewed deployment")
	}
	for _, required := range []string{"REVIEW", "--revision " + strings.Repeat("a", 40), "--backend native", "--listen 127.0.0.1"} {
		if !strings.Contains(output.String(), required) {
			t.Fatal("review missing", required)
		}
	}
	if strings.Contains(output.String(), "fixture-private-hub-credential") {
		t.Fatal("Hub token leaked")
	}
}
func TestWizardDeclineAndEOFNeverDeploy(t *testing.T) {
	for _, input := range []string{textWizardInput("n"), "", strings.TrimSuffix(textWizardInput("y"), "y\n")} {
		d, calls, _ := testWizardDeps(t, textFixture())
		keys := 0
		d.createKey = func(string) error { keys++; return nil }
		var output bytes.Buffer
		if err := runWizard(context.Background(), strings.NewReader(input), &output, d); err == nil {
			t.Fatal("canceled input succeeded")
		}
		if *calls != 0 || keys != 0 {
			t.Fatal("side effect before approval")
		}
	}
}
func TestWizardNewNetworkKeyOnlyAfterFinalApproval(t *testing.T) {
	for _, approved := range []bool{false, true} {
		d, calls, got := testWizardDeps(t, textFixture())
		var output bytes.Buffer
		key := filepath.Join(t.TempDir(), "private", "api.key")
		final := "n"
		if approved {
			final = "y"
		}
		input := strings.Join([]string{"org/model", "", "", "", "", "", "", "https-ingress", "new", key, "", final}, "\n") + "\n"
		// The shared model planner sees loopback while a NEW key is still only a
		// future action, but the exact reviewed/deployed plan has authenticated bind.
		created := 0
		d.createKey = func(path string) error {
			if !strings.Contains(output.String(), "REVIEW") {
				t.Fatal("key before review")
			}
			created++
			_, err := generateAPIKey(path)
			return err
		}
		err := runWizard(context.Background(), strings.NewReader(input), &output, d)
		if !approved {
			if err == nil || *calls != 0 || created != 0 {
				t.Fatal("declined key/deployment created")
			}
			if _, e := os.Stat(filepath.Dir(key)); !os.IsNotExist(e) {
				t.Fatal("key directory created before confirmation")
			}
			continue
		}
		if err != nil {
			t.Fatal(err, output.String())
		}
		if *calls != 1 || created != 1 || got.Listen != "0.0.0.0" || got.APIKeyFile != key {
			t.Fatal("authenticated intent not preserved")
		}
		value, e := readAPIKey(key)
		if e != nil {
			t.Fatal(e)
		}
		if strings.Contains(output.String(), value) {
			t.Fatal("API key value leaked")
		}
	}
}
func TestWizardExistingKeyAndShellSafeReproduction(t *testing.T) {
	d, calls, got := testWizardDeps(t, textFixture())
	key := fixtureAPIKey(t)
	d.executable = "/tmp/space and 'quote/modelship"
	d.createKey = func(string) error { t.Fatal("existing key replaced"); return nil }
	input := strings.Join([]string{"org/model", "", "", "", "", "", "", "https-ingress", "existing", key, "", "y"}, "\n") + "\n"
	var out bytes.Buffer
	if err := runWizard(context.Background(), strings.NewReader(input), &out, d); err != nil {
		t.Fatal(err, out.String())
	}
	if *calls != 1 || got.APIKeyFile != key {
		t.Fatal("existing key not referenced")
	}
	if strings.Contains(out.String(), strings.Repeat("a", 43)) {
		t.Fatal("existing key value printed")
	}
	if !strings.Contains(out.String(), "'\\''") {
		t.Fatal("executable path not shell quoted")
	}
}
func TestWizardHardBlockCannotBeOverridden(t *testing.T) {
	d, calls, _ := testWizardDeps(t, textFixture())
	keys := 0
	d.createKey = func(string) error { keys++; return nil }
	d.build = func(m Model, h Host, o Options) (Plan, error) {
		p, e := BuildPlan(m, h, o)
		p.Blockers = append(p.Blockers, "weights exceed physical VRAM")
		return p, e
	}
	var out bytes.Buffer
	if err := runWizard(context.Background(), strings.NewReader(textWizardInput("y")), &out, d); err == nil {
		t.Fatal("hard block ignored")
	}
	if *calls != 0 || keys != 0 || strings.Contains(out.String(), "Override ONLY") {
		t.Fatal("hard block offered bypass or side effect")
	}
}
func TestWizardRuntimeOverrideRequiresSeparateConsent(t *testing.T) {
	for _, consent := range []string{"n", "y"} {
		d, calls, _ := testWizardDeps(t, textFixture())
		var out bytes.Buffer
		d.build = func(m Model, h Host, o Options) (Plan, error) {
			p, e := BuildPlan(m, h, o)
			if !o.AllowRisk {
				p.Blockers = append(p.Blockers, "runtime heuristic")
			}
			return p, e
		}
		input := strings.TrimSuffix(textWizardInput("y"), "y\n") + consent + "\ny\n"
		err := runWizard(context.Background(), strings.NewReader(input), &out, d)
		if consent == "n" {
			if err == nil || *calls != 0 {
				t.Fatal("risk consent skipped")
			}
		} else {
			if err != nil || *calls != 1 || !strings.Contains(out.String(), "--allow-risk") {
				t.Fatal("explicit override not reproduced", err, out.String())
			}
		}
	}
}
func TestWizardNeverSilentlySubstitutesWAN(t *testing.T) {
	d, calls, _ := testWizardDeps(t, textFixture())
	inspected := 0
	d.inspect = func(context.Context, string, string) (Model, error) { inspected++; return Model{}, nil }
	var out bytes.Buffer
	if err := runWizard(context.Background(), strings.NewReader("Wan-AI/Wan2.2-T2V-A14B\nn\n"), &out, d); err == nil {
		t.Fatal("conversion refusal ignored")
	}
	if inspected != 0 || *calls != 0 {
		t.Fatal("inspected/deployed substitute without consent")
	}
}
func TestWizardWANExplicitConversionAndExperimentalConsent(t *testing.T) {
	d, calls, got := testWizardDeps(t, targetFixture(t, "Wan-AI/Wan2.2-T2V-A14B-Diffusers"))
	var out bytes.Buffer
	d.inspect = func(_ context.Context, id, rev string) (Model, error) {
		if id != "Wan-AI/Wan2.2-T2V-A14B-Diffusers" {
			t.Fatal("wrong explicit conversion")
		}
		return targetFixture(t, id), nil
	}
	d.build = func(m Model, h Host, o Options) (Plan, error) {
		return Plan{Backend: o.Backend, Model: m.Hub.ID, Revision: m.Hub.SHA, Task: o.Task, Name: o.Name, Port: o.Port, GPU: h.GPUs[0]}, nil
	}
	// conversion, revision, native/task/GPU auto-selected, name, port, python,
	// local access, no key, explicit experimental consent, wait, final.
	input := "Wan-AI/Wan2.2-T2V-A14B\ny\n\n\n\n\n\n\ny\n\ny\n"
	if err := runWizard(context.Background(), strings.NewReader(input), &out, d); err != nil {
		t.Fatal(err, out.String())
	}
	if *calls != 1 || got.RequestedModel != "Wan-AI/Wan2.2-T2V-A14B" || !strings.Contains(out.String(), "--weights-repo Wan-AI/Wan2.2-T2V-A14B-Diffusers") || !strings.Contains(out.String(), "--experimental") {
		t.Fatal("conversion/consent absent from plan", out.String())
	}
}
func TestWizardCodeConsentCannotBeAssumed(t *testing.T) {
	d, calls, _ := testWizardDeps(t, targetFixture(t, "OpenMOSS-Team/MOSS-Image-SFT"))
	var out bytes.Buffer
	d.build = func(Model, Host, Options) (Plan, error) {
		t.Fatal("planned custom code without consent")
		return Plan{}, nil
	}
	// native/task/GPU auto; name/port/python/access/key; experimental yes;
	// separate repository-code consent explicitly declined.
	input := "OpenMOSS-Team/MOSS-Image-SFT\n\n\n\n\n\n\ny\nn\n"
	if err := runWizard(context.Background(), strings.NewReader(input), &out, d); err == nil {
		t.Fatal("missing custom-code consent")
	}
	if *calls != 0 {
		t.Fatal("custom code deployed")
	}
	if !strings.Contains(out.String(), "NOT in a sandbox") || !strings.Contains(out.String(), "SHA-256") {
		t.Fatal("reviewed-code boundary not shown", out.String())
	}
}
func TestWizardTorchFailureAndCancellationNoDeployment(t *testing.T) {
	d, calls, _ := testWizardDeps(t, textFixture())
	d.probe = func(context.Context, string, Plan) (TorchInfo, error) {
		return TorchInfo{}, errors.New("CUDA preflight failed")
	}
	var out bytes.Buffer
	if err := runWizard(context.Background(), strings.NewReader(textWizardInput("y")), &out, d); err == nil || *calls != 0 {
		t.Fatal("failed CUDA preflight deployed")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := runWizard(ctx, strings.NewReader("org/model\n"), io.Discard, d); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
func TestWizardRejectsLongLinesAndPrivatePathReuse(t *testing.T) {
	w := wizardUI{ctx: context.Background(), in: bufio.NewReaderSize(strings.NewReader(strings.Repeat("a", 9000)+"\n"), 8193), out: io.Discard}
	if _, err := w.ask("value", "", nil); err == nil {
		t.Fatal("oversized line accepted")
	}
	if err := newKeyPath(fixtureAPIKey(t)); err == nil {
		t.Fatal("existing key scheduled for overwrite")
	}
	f, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if terminalFile(f) {
		t.Fatal("/dev/null mistaken for terminal")
	}
}

func TestWizardReviewEscapesRemoteTerminalControlData(t *testing.T) {
	d, _, _ := testWizardDeps(t, textFixture())
	d.build = func(m Model, h Host, o Options) (Plan, error) {
		p, e := BuildPlan(m, h, o)
		p.Warnings = append(p.Warnings, "untrusted\x1b[2J\u202e")
		p.Files = append(p.Files, "untrusted\x1b[31m.safetensors")
		return p, e
	}
	var out bytes.Buffer
	_ = runWizard(context.Background(), strings.NewReader(textWizardInput("n")), &out, d)
	if strings.ContainsAny(out.String(), "\x1b\u202e") {
		t.Fatal("terminal control data emitted")
	}
}
