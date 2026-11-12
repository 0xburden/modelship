package modelship

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"regexp"
	"strings"
)

type Recipe struct {
	ID                  string            `json:"id"`
	Name                string            `json:"name"`
	Kind                string            `json:"kind"`
	Status              string            `json:"status"`
	Repositories        []string          `json:"repositories,omitempty"`
	Architectures       []string          `json:"architectures,omitempty"`
	Pipelines           []string          `json:"pipelines,omitempty"`
	Backends            []string          `json:"backends"`
	Tasks               []string          `json:"tasks"`
	Runtime             string            `json:"runtime"`
	NativeLoader        string            `json:"native_loader,omitempty"`
	DType               string            `json:"dtype,omitempty"`
	MinPython           string            `json:"min_python,omitempty"`
	MinTorch            string            `json:"min_torch,omitempty"`
	ComponentDTypes     map[string]string `json:"component_dtypes,omitempty"`
	RequiresResearch    bool              `json:"requires_research,omitempty"`
	CodeRevision        string            `json:"code_revision,omitempty"`
	CodeHashes          map[string]string `json:"code_hashes,omitempty"`
	MinCompute          string            `json:"min_compute,omitempty"`
	MemoryFloorMiB      int64             `json:"memory_floor_mib,omitempty"`
	TextOnly            bool              `json:"text_only,omitempty"`
	TemplateCompletions bool              `json:"template_completions,omitempty"`
	DisableThinking     bool              `json:"disable_thinking,omitempty"`
	RequiresCustomCode  bool              `json:"requires_custom_code,omitempty"`
	NativeRequirements  []string          `json:"native_requirements,omitempty"`
	Blockers            []string          `json:"blockers,omitempty"`
	Notes               []string          `json:"notes,omitempty"`
}
type RecipeCatalog struct {
	Schema  int      `json:"schema"`
	Recipes []Recipe `json:"recipes"`
}

var dependencyPin = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9.+_-]*$`)

func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}
func LoadRecipes() (RecipeCatalog, error) {
	var catalog RecipeCatalog
	data, err := assets.ReadFile("assets/recipes.json")
	if err != nil {
		return catalog, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err = decoder.Decode(&catalog); err != nil {
		return catalog, err
	}
	if catalog.Schema != 1 || len(catalog.Recipes) == 0 {
		return catalog, fmt.Errorf("invalid recipe catalog schema")
	}
	seen := map[string]bool{}
	for _, r := range catalog.Recipes {
		if r.ID == "" || seen[r.ID] || !contains([]string{"text", "image", "video"}, r.Kind) || !contains([]string{"supported", "experimental", "planned"}, r.Status) {
			return catalog, fmt.Errorf("invalid/duplicate recipe %q", r.ID)
		}
		seen[r.ID] = true
		if len(r.Repositories)+len(r.Architectures)+len(r.Pipelines) == 0 || len(r.Tasks) == 0 || len(r.Backends) == 0 {
			return catalog, fmt.Errorf("recipe %s has no match/task/backend", r.ID)
		}
		if r.Status != "planned" {
			if r.NativeLoader == "" || len(r.NativeRequirements) == 0 {
				return catalog, fmt.Errorf("executable recipe %s lacks safe native configuration", r.ID)
			}
			if !contains([]string{"causal-lm", "multimodal-lm", "auto-pipeline", "qwen-image21", "moss-image", "wan-pipeline"}, r.NativeLoader) || !contains([]string{"float16", "bfloat16"}, r.DType) {
				return catalog, fmt.Errorf("invalid native loader/dtype in %s", r.ID)
			}
		}
		for component, dtype := range r.ComponentDTypes {
			if !repoPart.MatchString(component) || !contains([]string{"float16", "bfloat16", "float32"}, dtype) {
				return catalog, fmt.Errorf("invalid component dtype in %s", r.ID)
			}
		}
		if r.RequiresCustomCode && r.Status != "planned" {
			if r.NativeLoader != "moss-image" || !commitPattern.MatchString(r.CodeRevision) || len(r.CodeHashes) != 3 || len(r.Repositories) != 1 || r.Repositories[0] != "OpenMOSS-Team/MOSS-Image-SFT" {
				return catalog, fmt.Errorf("recipe %s lacks a reviewed code manifest", r.ID)
			}
			for _, path := range []string{"pipeline_moss_image.py", "scheduler/pipeline_moss_image.py", "transformer/pipeline_moss_image.py"} {
				hash := r.CodeHashes[path]
				if !fs.ValidPath(path) || len(hash) != 64 || strings.Trim(hash, "0123456789abcdef") != "" {
					return catalog, fmt.Errorf("recipe %s has an invalid code hash", r.ID)
				}
			}
		}
		for _, pin := range r.NativeRequirements {
			pkg, _, _ := strings.Cut(pin, "==")
			if !dependencyPin.MatchString(pin) || strings.EqualFold(pkg, "torch") {
				return catalog, fmt.Errorf("recipe %s contains an unpinned, unsafe, or torch dependency", r.ID)
			}
		}
	}
	return catalog, nil
}
func findRecipe(m Model) (Recipe, bool, error) {
	catalog, err := LoadRecipes()
	if err != nil {
		return Recipe{}, false, err
	}
	// Exact repositories also identify unusual native layouts such as WAN.
	for _, r := range catalog.Recipes {
		if contains(r.Repositories, m.Hub.ID) {
			return r, true, nil
		}
	}
	for _, r := range catalog.Recipes {
		// These execution paths deliberately accept only the reviewed official
		// repository, not arbitrary pipelines with the same class name.
		if r.NativeLoader == "moss-image" || r.NativeLoader == "wan-pipeline" {
			continue
		}
		if m.Index.Class != "" && contains(r.Pipelines, m.Index.Class) {
			return r, true, nil
		}
		if m.Index.Class == "" && len(m.Config.Architectures) == 1 && contains(r.Architectures, m.Config.Architectures[0]) {
			return r, true, nil
		}
	}
	return Recipe{}, false, nil
}
func validateRecipeShape(r Recipe, m Model) error {
	if len(r.Pipelines) > 0 && !contains(r.Pipelines, m.Index.Class) {
		return fmt.Errorf("repository pipeline %q does not match recipe %s", m.Index.Class, r.ID)
	}
	if len(r.Architectures) > 0 && (len(m.Config.Architectures) != 1 || !contains(r.Architectures, m.Config.Architectures[0])) {
		return fmt.Errorf("repository architectures %v do not match recipe %s", m.Config.Architectures, r.ID)
	}
	return nil
}
func nativeRequirements(p Plan) ([]string, error) {
	if len(p.NativeRequirements) > 0 {
		return append([]string{}, p.NativeRequirements...), nil
	}
	// Backward compatibility for saved v0.2 plans with no recipe snapshot.
	catalog, err := LoadRecipes()
	if err != nil {
		return nil, err
	}
	id := "transformers-text"
	if p.Runtime == "diffusers" {
		id = "sdxl"
	}
	for _, r := range catalog.Recipes {
		if r.ID == id {
			return append([]string{}, r.NativeRequirements...), nil
		}
	}
	return nil, fmt.Errorf("missing legacy native recipe")
}
