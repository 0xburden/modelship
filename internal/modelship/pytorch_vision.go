package modelship

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// Profiles are bundled reviewed adapters, not user-supplied execution recipes.
// They do not imply arbitrary multimodal architectures accept image input.
type NativeVisionProfile struct {
	Images                 *ImagePolicy `json:"images,omitempty"`
	ID                     string       `json:"id"`
	Repo                   string       `json:"repo"`
	Revision               string       `json:"revision"`
	Recipe                 string       `json:"recipe"`
	Architecture           string       `json:"architecture"`
	Processor              ArtifactPin  `json:"processor"`
	MaxImageTokens         int          `json:"max_image_tokens"`
	ActivationAllowanceMiB int64        `json:"activation_allowance_mib"`
}

const gemmaVisionRepo = "google/gemma-4-E4B-it"
const gemmaVisionRevision = "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"
const qwenVisionRepo = "Qwen/Qwen3.8-27B"
const qwenVisionRevision = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"

func nativeVisionProfiles() ([]NativeVisionProfile, error) {
	var c struct {
		Schema   int                   `json:"schema"`
		Profiles []NativeVisionProfile `json:"profiles"`
	}
	b, e := assets.ReadFile("assets/pytorch_vision_profiles.json")
	if e != nil {
		return nil, e
	}
	if json.Unmarshal(b, &c) != nil || c.Schema != 1 || len(c.Profiles) != 3 {
		return nil, fmt.Errorf("invalid reviewed PyTorch vision profiles")
	}
	seen := map[string]bool{}
	for _, p := range c.Profiles {
		if seen[p.ID] {
			return nil, fmt.Errorf("duplicate reviewed vision profile")
		}
		seen[p.ID] = true
		if p.ID == "qwen38-27b-images-v1" {
			if p.Repo != qwenVisionRepo || p.Revision != qwenVisionRevision || p.Recipe != "qwen3.8-multimodal" || p.Architecture != "Qwen3_5ForConditionalGeneration" || p.Processor != (ArtifactPin{Name: "preprocessor_config.json", Size: 390, SHA256: "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"}) || p.MaxImageTokens != 1024 || p.ActivationAllowanceMiB != 4096 || p.Images == nil || validateImagePolicy(p.Images) != nil {
				return nil, fmt.Errorf("invalid reviewed Qwen image profile")
			}
			continue
		}
		if (p.ID != "gemma4-e4b-image-v1" && p.ID != "gemma4-e4b-images-v2") || p.Repo != gemmaVisionRepo || p.Revision != gemmaVisionRevision || p.Recipe != "gemma4-text" || p.Architecture != "Gemma4ForConditionalGeneration" || p.Processor != (ArtifactPin{Name: "processor_config.json", Size: 1689, SHA256: "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c"}) || p.MaxImageTokens != 280 || (p.ID == "gemma4-e4b-image-v1" && (p.ActivationAllowanceMiB != 2048 || p.Images != nil) || p.ID == "gemma4-e4b-images-v2" && (p.ActivationAllowanceMiB != 4096 || p.Images == nil || validateImagePolicy(p.Images) != nil)) {
			return nil, fmt.Errorf("invalid reviewed Gemma image profile")
		}
	}
	return c.Profiles, nil
}
func nativeVisionFor(m Model) (*NativeVisionProfile, error) {
	profiles, e := nativeVisionProfiles()
	if e != nil {
		return nil, e
	}
	for _, p := range profiles {
		if p.Images != nil && m.Hub.ID == p.Repo && m.Hub.SHA == p.Revision && len(m.Config.Architectures) == 1 && m.Config.Architectures[0] == p.Architecture {
			return &p, nil
		}
	}
	return nil, fmt.Errorf("PyTorch image understanding requires reviewed %s @ %s or %s @ %s; no model/revision substitution", gemmaVisionRepo, gemmaVisionRevision, qwenVisionRepo, qwenVisionRevision)
}
func validateNativeVisionPlan(p Plan) error {
	if p.Vision == nil {
		return nil
	}
	profiles, e := nativeVisionProfiles()
	if e != nil {
		return e
	}
	match := false
	for _, profile := range profiles {
		if reflect.DeepEqual(*p.Vision, profile) {
			match = true
		}
	}
	if !match || p.GGUF != nil || p.Backend != "native" || p.Runtime != "transformers" || p.NativeLoader != "multimodal-lm" || p.RecipeID != p.Vision.Recipe || p.Model != p.Vision.Repo || p.Revision != p.Vision.Revision || p.Task != "text-generation" || p.DType != "bfloat16" || !p.TemplateCompletions || !p.DisableThinking || !contains(p.NativeRequirements, "transformers==5.22.2") || !contains(p.NativeRequirements, "pillow==11.3.0") || p.Vision.Images != nil && !contains(p.NativeRequirements, "numpy==2.3.5") {
		return fmt.Errorf("saved PyTorch vision constraints differ from the reviewed profile")
	}
	return nil
}
