package modelship

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func targetFixture(t *testing.T, id string) Model {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "models", strings.ReplaceAll(id, "/", "__")+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Hub    HubModel    `json:"hub"`
		Config ModelConfig `json:"config"`
		Index  ModelIndex  `json:"index"`
		Source string      `json:"source_config"`
	}
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	return Model{Hub: fixture.Hub, Config: fixture.Config, Index: fixture.Index, SourceConfig: fixture.Source}
}
func TestRecipeRegistryLoadsAndPinsDependencies(t *testing.T) {
	catalog, err := LoadRecipes()
	if err != nil {
		t.Fatal(err)
	}
	if len(catalog.Recipes) != 11 {
		t.Fatal(len(catalog.Recipes))
	}
	for _, r := range catalog.Recipes {
		for _, pin := range r.NativeRequirements {
			if !dependencyPin.MatchString(pin) || strings.HasPrefix(pin, "torch==") {
				t.Fatal(r.ID, pin)
			}
		}
		if r.RequiresCustomCode && r.Status != "planned" && (r.CodeRevision == "" || len(r.CodeHashes) != 3 || r.NativeLoader != "moss-image") {
			t.Fatal("custom code recipe lacks reviewed immutable manifest")
		}
	}
}
func TestRequestedModelsAreRecognizedWithoutBeingSilentlyEnabled(t *testing.T) {
	cases := []struct {
		id, recipe, status string
		bytes              int64
	}{
		{"Qwen/Qwen3.8-27B", "qwen3.8-multimodal", "experimental", 55563006776},
		{"Qwen/Qwen-Image-2.1", "qwen-image21", "experimental", 33115613408},
		{"google/gemma-4-E4B-it", "gemma4-text", "experimental", 15992595884},
		{"OpenMOSS-Team/MOSS-Image-SFT", "moss-image", "experimental", 27199664580},
		{"Wan-AI/Wan2.2-T2V-A14B", "wan22-t2v-native", "planned", 126177685050},
		{"Wan-AI/Wan2.2-T2V-A14B-Diffusers", "wan22-t2v-diffusers", "experimental", 126177598620},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			m := targetFixture(t, tc.id)
			report, err := inspectModel(m)
			if err != nil {
				t.Fatal(err)
			}
			if report.Recipe == nil || report.Recipe.ID != tc.recipe || report.RecipeStatus != tc.status || report.WeightArtifactBytes != tc.bytes {
				t.Fatalf("%+v", report)
			}
			if tc.status == "planned" {
				o := optsFixture()
				o.Backend = "native"
				o.Experimental = true
				if _, err = BuildPlan(m, hostFixture(), o); err == nil || !strings.Contains(err.Error(), "not deployable") {
					t.Fatal("planned recipe enabled", err)
				}
			}
		})
	}
}
func TestGemma4RequiresOptInAndUsesNewIsolatedProfile(t *testing.T) {
	m := targetFixture(t, "google/gemma-4-E4B-it")
	o := optsFixture()
	o.Backend = "native"
	p, err := BuildPlan(m, hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Blockers) != 1 || !strings.Contains(p.Blockers[0], "--experimental") {
		t.Fatalf("%+v", p)
	}
	o.Experimental = true
	p, err = BuildPlan(m, hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Blockers) > 0 || p.DType != "bfloat16" || p.NativeLoader != "multimodal-lm" || p.MinPython != "3.11" || !p.TemplateCompletions || !p.DisableThinking {
		t.Fatalf("%+v", p)
	}
	if !contains(p.NativeRequirements, "transformers==5.22.2") || contains(p.NativeRequirements, "transformers==4.56.2") {
		t.Fatal(p.NativeRequirements)
	}
	env := strings.Join(nativeEnv(p, "test"), "\n")
	for _, want := range []string{"MODEL_DTYPE=bfloat16", "MODEL_NATIVE_LOADER=multimodal-lm", "MODEL_RECIPE=gemma4-text", "MODEL_TEMPLATE_COMPLETIONS=true", "MODEL_DISABLE_THINKING=true"} {
		if !strings.Contains(env, want) {
			t.Fatal(env)
		}
	}
	o.Backend = "docker"
	if _, err = BuildPlan(m, hostFixture(), o); err == nil {
		t.Fatal("new recipe leaked into old Docker runtime")
	}
}
func TestGemmaNestedTextConfigAndComputeValidation(t *testing.T) {
	m := targetFixture(t, "google/gemma-4-E4B-it")
	o := optsFixture()
	o.Backend = "native"
	o.Experimental = true
	m.Config.TextConfig.MaxPosition = 2048
	if _, err := BuildPlan(m, hostFixture(), o); err == nil {
		t.Fatal("ignored nested max context")
	}
	m = targetFixture(t, "google/gemma-4-E4B-it")
	h := hostFixture()
	h.GPUs[0].Compute = "7.5"
	p, err := BuildPlan(m, h, o)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Blockers) != 1 || !strings.Contains(p.Blockers[0], "8.0") {
		t.Fatalf("%+v", p)
	}
	m.Config.Architectures = []string{"UnknownClass"}
	if _, err = BuildPlan(m, hostFixture(), o); err == nil {
		t.Fatal("exact repo bypassed architecture validation")
	}
}
func TestInspectionCountsUnknownSizesWithoutInventingMemory(t *testing.T) {
	m := textFixture()
	m.SourceConfig = "config.json"
	m.Hub.Siblings = append(m.Hub.Siblings, ModelFile{Name: "alternate/model.gguf"})
	r, err := inspectModel(m)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.WeightGroups) != 2 || r.WeightGroups[1].UnknownSizes != 1 {
		t.Fatalf("%+v", r.WeightGroups)
	}
}
func TestPartialInspectionReportsGatedConfiguration(t *testing.T) {
	sha := strings.Repeat("a", 40)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			fmt.Fprintf(w, `{"sha":"%s","pipeline_tag":"any-to-any","gated":"manual","config":{"architectures":["Gemma4ForConditionalGeneration"]},"siblings":[{"rfilename":"config.json"}]}`, sha)
		} else {
			w.WriteHeader(403)
		}
	}))
	defer server.Close()
	h := &Hub{BaseURL: server.URL, Client: server.Client()}
	m, err := h.InspectPartial(context.Background(), "google/gemma-4-E4B-it", "main")
	if err != nil {
		t.Fatal(err)
	}
	r, err := inspectModel(m)
	if err != nil {
		t.Fatal(err)
	}
	if r.ConfigurationReadable || len(r.Warnings) < 3 || r.Recipe == nil || r.Recipe.ID != "gemma4-text" {
		t.Fatalf("%+v", r)
	}
	if _, err = h.Inspect(context.Background(), "google/gemma-4-E4B-it", "main"); err == nil {
		t.Fatal("deployment inspection ignored missing authorization")
	}
}
func TestLegacySavedNativePlanStillHasDependencyFallback(t *testing.T) {
	for _, p := range []Plan{{Runtime: "transformers"}, {Runtime: "diffusers"}} {
		requirements, err := nativeRequirements(p)
		if err != nil {
			t.Fatal(err)
		}
		if !contains(requirements, "transformers==4.56.2") {
			t.Fatal(requirements)
		}
	}
}

func TestResearchAndCustomCodeRequireSeparateAcknowledgments(t *testing.T) {
	h := hostFixture()
	h.GPUs[0].TotalMiB = 163840
	h.GPUs[0].FreeMiB = 160000
	o := optsFixture()
	o.Backend = "native"
	o.Experimental = true
	image := targetFixture(t, "Qwen/Qwen-Image-2.1")
	p, err := BuildPlan(image, h, o)
	if err != nil || !strings.Contains(strings.Join(p.Blockers, ";"), "--research") {
		t.Fatalf("%+v %v", p, err)
	}
	o.Research = true
	p, err = BuildPlan(image, h, o)
	if err != nil || len(p.Blockers) > 0 || !p.Research || p.ComponentDTypes["vae"] != "float32" {
		t.Fatalf("%+v %v", p, err)
	}
	moss := targetFixture(t, "OpenMOSS-Team/MOSS-Image-SFT")
	p, err = BuildPlan(moss, h, o)
	if err != nil || !strings.Contains(strings.Join(p.Blockers, ";"), "--trust-model-code") {
		t.Fatalf("research implicitly trusted code: %+v %v", p, err)
	}
	o.TrustModelCode = true
	p, err = BuildPlan(moss, h, o)
	if err != nil || len(p.Blockers) > 0 || !p.TrustModelCode || p.CodeRevision != moss.Hub.SHA || len(p.CodeHashes) != 3 {
		t.Fatalf("%+v %v", p, err)
	}
	moss.Hub.SHA = strings.Repeat("a", 40)
	p, err = BuildPlan(moss, h, o)
	if err != nil || !strings.Contains(strings.Join(p.Blockers, ";"), "reviewed immutable revision") {
		t.Fatalf("unreviewed code revision accepted: %+v %v", p, err)
	}
}
func TestWANConvertedRepositoryIsOnlyExplicitlySelected(t *testing.T) {
	original := "Wan-AI/Wan2.2-T2V-A14B"
	converted := original + "-Diffusers"
	if actual, err := resolveWeightRepo(original, ""); err != nil || actual != original {
		t.Fatal("implicit substitution", actual, err)
	}
	if actual, err := resolveWeightRepo(original, converted); err != nil || actual != converted {
		t.Fatal(actual, err)
	}
	for _, pair := range [][2]string{{original, "other/model"}, {"Qwen/Qwen3.8-27B", converted}} {
		if _, err := resolveWeightRepo(pair[0], pair[1]); err == nil {
			t.Fatal("arbitrary model substitution permitted")
		}
	}
}

func TestGemma4VariantUsesExistingFamilyMatcherWithoutRepositoryWhitelist(t *testing.T) {
	// Synthetic metadata copied from the known Gemma fixture. This checks only
	// family matching, not this community repository's actual files or inference.
	model := targetFixture(t, "google/gemma-4-E4B-it")
	model.Hub.ID = "llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic"
	model.Hub.SHA = strings.Repeat("c", 40)
	recipe, known, err := findRecipe(model)
	if err != nil || !known || recipe.ID != "gemma4-text" || recipe.Status != "experimental" {
		t.Fatal("compatible family matching regressed", err)
	}
	if contains(recipe.Repositories, model.Hub.ID) {
		t.Fatal("test must exercise architecture matching, not a new allowlist")
	}
	if err = validateRecipeShape(recipe, model); err != nil {
		t.Fatal(err)
	}
}
