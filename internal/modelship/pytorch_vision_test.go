package modelship

import (
	"context"
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func gemmaImagePlan(t *testing.T) (Model, Host, Options, Plan) {
	t.Helper()
	m := targetFixture(t, gemmaVisionRepo)
	h := hostFixture()
	o := optsFixture()
	o.Backend = "native"
	o.Experimental = true
	o.Vision = true
	p, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	return m, h, o, p
}
func TestGemmaVisionFullWeightsAndSeparateActivationHeuristic(t *testing.T) {
	m, h, o, p := gemmaImagePlan(t)
	o.Vision = false
	text, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	if p.Vision == nil || p.Vision.MaxImageTokens != 280 || p.WeightResidentBytes != text.WeightResidentBytes || p.WeightBytes != text.WeightBytes || !reflect.DeepEqual(p.Files, text.Files) || p.EstimatedVRAMMiB != text.EstimatedVRAMMiB+2048 {
		t.Fatal("vision changed weight placement or omitted activation allowance", p.Vision)
	}
	if e = validateNativeVisionPlan(p); e != nil {
		t.Fatal(e)
	}
	b, _ := json.Marshal(p)
	var saved Plan
	json.Unmarshal(b, &saved)
	if e = validateNativeVisionPlan(saved); e != nil {
		t.Fatal("saved vision profile changed identity", e)
	}
	o.Vision = true
	o.AllowRisk = true
	h.GPUs[0].FreeMiB = 1024
	p, e = BuildPlan(m, h, o)
	if e != nil || p.WeightsFitFree || len(p.Blockers) == 0 {
		t.Fatal("vision bypassed hard weight gate", e, p.Blockers)
	}
}
func TestGemmaVisionCannotBroadenModelsOrMixRuntimes(t *testing.T) {
	for _, name := range []string{"revision", "repository", "architecture", "docker", "projector", "generation", "qwen"} {
		t.Run(name, func(t *testing.T) {
			m, h, o, _ := gemmaImagePlan(t)
			switch name {
			case "revision":
				m.Hub.SHA = strings.Repeat("a", 40)
			case "repository":
				m.Hub.ID = "community/gemma-variant"
			case "architecture":
				m.Config.Architectures = []string{"GemmaForCausalLM"}
			case "docker":
				o.Backend = "docker"
			case "projector":
				o.MMProjFile = ggufProjectorFile
			case "generation":
				o.Task = "text-to-image"
			case "qwen":
				m = targetFixture(t, "Qwen/Qwen3.8-27B")
			}
			if _, e := BuildPlan(m, h, o); e == nil {
				t.Fatal("unreviewed PyTorch image path enabled")
			}
		})
	}
	_, _, _, p := gemmaImagePlan(t)
	copy := *p.Vision
	p.Vision = &copy
	p.Vision.MaxImageTokens = 1120
	if e := validateNativeVisionPlan(p); e == nil {
		t.Fatal("mutated image-token profile accepted")
	}
	m, h, o := ggufPlanFixture(t)
	gp, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	gp.Vision = &copy
	if e = validateGGUFPlan(gp); e == nil {
		t.Fatal("PyTorch profile smuggled into GGUF")
	}
}
func TestGemmaVisionNativeEnvAndSchemaBoundary(t *testing.T) {
	_, _, _, p := gemmaImagePlan(t)
	t.Setenv("MODEL_VISION_PROFILE", "untrusted")
	env := strings.Join(nativeEnv(p, "test"), "\n")
	if !strings.Contains(env, "MODEL_VISION_PROFILE=gemma4-e4b-image-v1") {
		t.Fatal("missing explicit processor profile")
	}
	specFound := false
	for _, item := range nativeEnv(p, "test") {
		if strings.HasPrefix(item, "MODEL_VISION_PROFILE_SPEC=") {
			var profile NativeVisionProfile
			if e := json.Unmarshal([]byte(strings.TrimPrefix(item, "MODEL_VISION_PROFILE_SPEC=")), &profile); e != nil || !reflect.DeepEqual(profile, *p.Vision) {
				t.Fatal("worker profile binding differs", e)
			}
			specFound = true
		}
	}
	if !specFound {
		t.Fatal("missing complete worker profile binding")
	}

	text := p
	text.Vision = nil
	env = strings.Join(nativeEnv(text, "test"), "\n")
	if strings.Contains(env, "untrusted") || !contains(nativeEnv(text, "test"), "MODEL_VISION_PROFILE=") {
		t.Fatal("inherited vision environment was not cleared", env)
	}
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir, e := nativeDir(p.Name)
	if e != nil {
		t.Fatal(e)
	}
	if e = privateDir(dir); e != nil {
		t.Fatal(e)
	}
	// A legacy image plan predates per-request text controls.
	p.TextFeatures = nil
	for _, schema := range []int{5, 6, 7} {
		cfg := NativeConfig{Schema: schema, Plan: p}
		if e = writeJSONFile(filepath.Join(dir, "config.json"), cfg); e != nil {
			t.Fatal(e)
		}
		_, _, e = nativeConfig(p.Name)
		if (schema < 7) != (e != nil) {
			t.Fatal("PyTorch vision schema guard", schema, e)
		}
	}
}
func TestGemmaVisionAssetsAndWizardReproduction(t *testing.T) {
	for _, name := range []string{"gemma_vision.py", "vision_input.py", "pytorch_vision_profiles.json"} {
		if _, e := assets.ReadFile("assets/" + name); e != nil {
			t.Fatal(e)
		}
	}
	_, _, o, p := gemmaImagePlan(t)
	command := wizardCommand("modelship", p.Model, "", o, p, 0)
	if !strings.Contains(command, "--vision") || strings.Contains(command, "--mmproj-file") {
		t.Fatal(command)
	}
	d, calls, got := testWizardDeps(t, targetFixture(t, gemmaVisionRepo))
	d.detect = func(context.Context, string) (Host, error) {
		h := hostFixture()
		h.GPUs[0].TotalMiB = 49152
		h.GPUs[0].FreeMiB = 48000
		return h, nil
	}
	input := []string{gemmaVisionRepo, "", "2", "", "", "", "", "1", "", "y", "", "y"}
	var out strings.Builder
	if e := runWizard(context.Background(), strings.NewReader(strings.Join(input, "\n")+"\n"), &out, d); e != nil {
		t.Fatal(e, out.String())
	}
	if *calls != 1 || got.Vision == nil || !strings.Contains(out.String(), gemmaVisionRevision) || strings.Contains(out.String(), "--mmproj-file") {
		t.Fatal("Gemma image wizard/repro mismatched", out.String())
	}
}
