package modelship

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// An immutable execution contract, not arbitrary template/generation options.
type NativeTextProfile struct {
	ID                    string `json:"id"`
	Repo                  string `json:"repo"`
	Revision              string `json:"revision"`
	Recipe                string `json:"recipe"`
	Architecture          string `json:"architecture"`
	MaxOutputTokens       int    `json:"max_output_tokens"`
	DefaultEnableThinking bool   `json:"default_enable_thinking"`
	ChannelStartID        int    `json:"channel_start_id"`
	ChannelEndID          int    `json:"channel_end_id"`
	ThinkTokenID          int    `json:"think_token_id"`
}

func nativeTextProfiles() ([]NativeTextProfile, error) {
	var catalog struct {
		Schema   int                 `json:"schema"`
		Profiles []NativeTextProfile `json:"profiles"`
	}
	b, e := assets.ReadFile("assets/native_text_profiles.json")
	if e != nil {
		return nil, e
	}
	if json.Unmarshal(b, &catalog) != nil || catalog.Schema != 1 || len(catalog.Profiles) != 1 {
		return nil, fmt.Errorf("invalid native text controls catalog")
	}
	p := catalog.Profiles[0]
	if p.ID != "gemma4-e4b-text-controls-v1" || p.Repo != gemmaVisionRepo || p.Revision != gemmaVisionRevision || p.Recipe != "gemma4-text" || p.Architecture != "Gemma4ForConditionalGeneration" || p.MaxOutputTokens != 1024 || p.DefaultEnableThinking || p.ChannelStartID != 100 || p.ChannelEndID != 101 || p.ThinkTokenID != 98 {
		return nil, fmt.Errorf("invalid reviewed native text controls profile")
	}
	return catalog.Profiles, nil
}
func nativeTextFor(m Model) (*NativeTextProfile, error) {
	profiles, e := nativeTextProfiles()
	if e != nil {
		return nil, e
	}
	for _, p := range profiles {
		if m.Hub.ID == p.Repo && m.Hub.SHA == p.Revision && len(m.Config.Architectures) == 1 && m.Config.Architectures[0] == p.Architecture {
			return &p, nil
		}
	}
	return nil, nil
}
func validateNativeTextPlan(p Plan) error {
	if p.TextFeatures == nil {
		return nil
	}
	profiles, e := nativeTextProfiles()
	if e != nil {
		return e
	}
	match := false
	for _, profile := range profiles {
		if reflect.DeepEqual(*p.TextFeatures, profile) {
			match = true
		}
	}
	f := p.TextFeatures
	if !match || p.GGUF != nil || p.Backend != "native" || p.Runtime != "transformers" || p.NativeLoader != "multimodal-lm" || p.Model != f.Repo || p.Revision != f.Revision || p.RecipeID != f.Recipe || p.Task != "text-generation" || p.DType != "bfloat16" || !p.TemplateCompletions || !p.DisableThinking || !contains(p.NativeRequirements, "transformers==5.22.2") {
		return fmt.Errorf("saved native text controls differ from the reviewed profile")
	}
	return nil
}
