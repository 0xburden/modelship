package modelship

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// Profiles are bundled reviewed adapters, not user-supplied execution recipes.
// They do not imply arbitrary multimodal architectures accept image input.
type NativeVisionProfile struct {
	ID                     string      `json:"id"`
	Repo                   string      `json:"repo"`
	Revision               string      `json:"revision"`
	Recipe                 string      `json:"recipe"`
	Architecture           string      `json:"architecture"`
	Processor              ArtifactPin `json:"processor"`
	MaxImageTokens         int         `json:"max_image_tokens"`
	ActivationAllowanceMiB int64       `json:"activation_allowance_mib"`
}

const gemmaVisionRepo = "google/gemma-4-E4B-it"
const gemmaVisionRevision = "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"

func nativeVisionProfiles() ([]NativeVisionProfile, error) {
	var c struct {
		Schema   int                   `json:"schema"`
		Profiles []NativeVisionProfile `json:"profiles"`
	}
	b, e := assets.ReadFile("assets/pytorch_vision_profiles.json")
	if e != nil {
		return nil, e
	}
	if json.Unmarshal(b, &c) != nil || c.Schema != 1 || len(c.Profiles) != 1 {
		return nil, fmt.Errorf("invalid reviewed PyTorch vision profiles")
	}
	p := c.Profiles[0]
	if p.ID != "gemma4-e4b-image-v1" || p.Repo != gemmaVisionRepo || p.Revision != gemmaVisionRevision || p.Recipe != "gemma4-text" || p.Architecture != "Gemma4ForConditionalGeneration" || p.Processor.Name != "processor_config.json" || p.Processor.Size <= 0 || p.Processor.Size > 1<<20 || p.Processor.URL != "" || !artifactDigestPattern.MatchString(p.Processor.SHA256) || p.MaxImageTokens != 280 || p.ActivationAllowanceMiB != 2048 {
		return nil, fmt.Errorf("invalid reviewed Gemma image profile")
	}
	return c.Profiles, nil
}
func nativeVisionFor(m Model) (*NativeVisionProfile, error) {
	profiles, e := nativeVisionProfiles()
	if e != nil {
		return nil, e
	}
	for _, p := range profiles {
		if m.Hub.ID == p.Repo && m.Hub.SHA == p.Revision && len(m.Config.Architectures) == 1 && m.Config.Architectures[0] == p.Architecture {
			return &p, nil
		}
	}
	return nil, fmt.Errorf("PyTorch image understanding initially requires reviewed %s @ %s; no model/revision substitution", gemmaVisionRepo, gemmaVisionRevision)
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
	if !match || p.GGUF != nil || p.Backend != "native" || p.Runtime != "transformers" || p.NativeLoader != "multimodal-lm" || p.RecipeID != p.Vision.Recipe || p.Model != p.Vision.Repo || p.Revision != p.Vision.Revision || p.Task != "text-generation" || p.DType != "bfloat16" || !p.TemplateCompletions || !p.DisableThinking || !contains(p.NativeRequirements, "transformers==5.22.2") || !contains(p.NativeRequirements, "pillow==11.3.0") {
		return fmt.Errorf("saved PyTorch vision constraints differ from the reviewed profile")
	}
	return nil
}
