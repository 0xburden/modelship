package modelship

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestNativeGemmaTextProfileConstraintsAndLegacyBoundary(t *testing.T) {
	_, _, _, p := gemmaImagePlan(t)
	if p.TextFeatures == nil || p.TextFeatures.ID != "gemma4-e4b-text-controls-v1" || !p.DisableThinking || p.TextFeatures.DefaultEnableThinking || p.TextFeatures.MaxOutputTokens != 1024 {
		t.Fatal("missing bounded opt-in text contract", p)
	}
	if e := validateNativeTextPlan(p); e != nil {
		t.Fatal(e)
	}
	for _, field := range []string{"revision", "repo", "dtype", "thinking-default", "loader", "limit", "token", "docker"} {
		q := p
		f := *p.TextFeatures
		q.TextFeatures = &f
		switch field {
		case "revision":
			q.Revision = strings.Repeat("b", 40)
		case "repo":
			q.Model = "community/variant"
		case "dtype":
			q.DType = "float16"
		case "thinking-default":
			f.DefaultEnableThinking = true
		case "loader":
			q.NativeLoader = "causal-lm"
		case "limit":
			f.MaxOutputTokens = 4096
		case "token":
			f.ChannelStartID++
		case "docker":
			q.Backend = "docker"
		}
		if validateNativeTextPlan(q) == nil {
			t.Fatal("unreviewed execution policy accepted", field)
		}
	}
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir, e := nativeDir(p.Name)
	if e != nil {
		t.Fatal(e)
	}
	if e = privateDir(dir); e != nil {
		t.Fatal(e)
	}
	for _, schema := range []int{6, 7, 8} {
		cfg := NativeConfig{Schema: schema, Plan: p}
		if e = writeJSONFile(filepath.Join(dir, "config.json"), cfg); e != nil {
			t.Fatal(e)
		}
		_, _, e = nativeConfig(p.Name)
		if (schema < 8) != (e != nil) {
			t.Fatal("text schema boundary", schema, e)
		}
	}
	// Old image plans stay readable and do NOT acquire controls implicitly.
	p.TextFeatures = nil
	if e = writeJSONFile(filepath.Join(dir, "config.json"), NativeConfig{Schema: 7, Plan: p}); e != nil {
		t.Fatal(e)
	}
	legacy, _, e := nativeConfig(p.Name)
	if e != nil || legacy.Plan.TextFeatures != nil {
		t.Fatal("legacy execution was migrated", e)
	}
}

func TestNativeTextProfileCannotBroadenOtherCheckpointsOrInheritEnvironment(t *testing.T) {
	for _, id := range []string{"Qwen/Qwen3.8-27B", "google/gemma-4-E4B-it"} {
		m := targetFixture(t, id)
		if id == gemmaVisionRepo {
			m.Hub.SHA = strings.Repeat("b", 40)
		}
		f, e := nativeTextFor(m)
		if e != nil || f != nil {
			t.Fatal("unreviewed model gained controls", id, e)
		}
	}
	_, _, _, p := gemmaImagePlan(t)
	t.Setenv("MODEL_TEXT_PROFILE", "inherited-secret")
	t.Setenv("MODEL_TEXT_PROFILE_SPEC", "inherited-secret")
	env := nativeEnv(p, "a")
	found := false
	for _, item := range env {
		if strings.HasPrefix(item, "MODEL_TEXT_PROFILE_SPEC=") {
			var f NativeTextProfile
			e := json.Unmarshal([]byte(strings.TrimPrefix(item, "MODEL_TEXT_PROFILE_SPEC=")), &f)
			if e != nil || !reflect.DeepEqual(f, *p.TextFeatures) {
				t.Fatal("worker execution spec differs", e)
			}
			found = true
		}
	}
	if !found || strings.Contains(strings.Join(env, "\n"), "inherited-secret") {
		t.Fatal("profile env not replaced")
	}
	p.TextFeatures = nil
	env = nativeEnv(p, "a")
	if !contains(env, "MODEL_TEXT_PROFILE=") || !contains(env, "MODEL_TEXT_PROFILE_SPEC=null") {
		t.Fatal("legacy controls environment not cleared")
	}
}
