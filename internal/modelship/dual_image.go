package modelship

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
)

const dualImageTask = "image-generate-and-edit"
const dualImageAllowanceMiB = 2048

type dualImageProfile struct {
	Model           string            `json:"model"`
	Revision        string            `json:"revision"`
	Recipe          string            `json:"recipe"`
	Pipeline        string            `json:"pipeline"`
	Loader          string            `json:"loader"`
	Diffusers       string            `json:"diffusers"`
	DType           string            `json:"dtype"`
	ComponentDTypes map[string]string `json:"component_dtypes"`
	Research        bool              `json:"research"`
}

func dualImageProfileFor(model, revision, pipeline, recipe string) (dualImageProfile, bool) {
	var profiles []dualImageProfile
	data, err := assets.ReadFile("assets/dual_image_profiles.json")
	if err != nil || json.Unmarshal(data, &profiles) != nil {
		return dualImageProfile{}, false
	}
	for _, p := range profiles {
		if p.Model == model && p.Revision == revision && p.Pipeline == pipeline && p.Recipe == recipe {
			return p, true
		}
	}
	return dualImageProfile{}, false
}

func imageTaskChoices(r Recipe, m Model, backend string) []string {
	if _, ok := dualImageProfileFor(m.Hub.ID, m.Hub.SHA, m.Index.Class, r.ID); ok && backend == "native" {
		return append(append([]string{}, r.Tasks...), dualImageTask)
	}
	return r.Tasks
}

func validateDualImagePlan(p Plan) error {
	if p.Task != dualImageTask {
		return nil
	}
	profile, ok := dualImageProfileFor(p.Model, p.Revision, p.Pipeline, p.RecipeID)
	if !ok || p.Backend != "native" || p.Runtime != "diffusers" || p.Vision != nil || p.GGUF != nil || p.TextFeatures != nil || p.TrustModelCode || p.CodeRevision != "" || len(p.CodeHashes) > 0 || p.NativeLoader != profile.Loader || p.DType != profile.DType || !maps.Equal(p.ComponentDTypes, profile.ComponentDTypes) || !contains(p.NativeRequirements, "diffusers=="+profile.Diffusers) {
		return fmt.Errorf("%s requires an exact reviewed native Diffusers image recipe; see IMAGE_GENERATE_EDIT.md", dualImageTask)
	}
	catalog, err := LoadRecipes()
	if err != nil {
		return err
	}
	for _, recipe := range catalog.Recipes {
		if recipe.ID == p.RecipeID && slices.Equal(p.NativeRequirements, recipe.NativeRequirements) && p.MinPython == recipe.MinPython && p.MinTorch == recipe.MinTorch {
			return nil
		}
	}
	return fmt.Errorf("dual image runtime requirements must match the reviewed recipe")
}

func validateDualImageExecution(p Plan) error {
	if err := validateDualImagePlan(p); err != nil {
		return err
	}
	if p.Task == dualImageTask {
		if p.WeightResidentBytes <= 0 || strings.TrimSpace(p.GPU.UUID) == "" {
			return fmt.Errorf("dual image execution requires a resident-weight budget and selected GPU identity; re-plan the deployment")
		}
		profile, _ := dualImageProfileFor(p.Model, p.Revision, p.Pipeline, p.RecipeID)
		if profile.Research && !p.Research {
			return fmt.Errorf("dual-image Qwen requires --research acknowledgment of non-commercial use")
		}
	}
	return nil
}

func imageHealthMatches(task string, imageTasks []string) bool {
	return task != dualImageTask || len(imageTasks) == 2 && imageTasks[0] == "text-to-image" && imageTasks[1] == "image-to-image"
}
