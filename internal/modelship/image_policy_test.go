package modelship

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestNativeMultiImageSavedContractAndLegacyBoundary(t *testing.T) {
	_, _, _, p := gemmaImagePlan(t)
	if p.Vision.Images == nil || *p.Vision.Images != reviewedImagePolicy() {
		t.Fatal("missing explicit image policy")
	}
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir, e := nativeDir(p.Name)
	if e != nil {
		t.Fatal(e)
	}
	if e = privateDir(dir); e != nil {
		t.Fatal(e)
	}
	for schema := 1; schema <= 9; schema++ {
		if e = writeJSONFile(filepath.Join(dir, "config.json"), NativeConfig{Schema: schema, Plan: p}); e != nil {
			t.Fatal(e)
		}
		_, _, e = nativeConfig(p.Name)
		if (schema < 9) != (e != nil) {
			t.Fatal(schema, e)
		}
	}
	changed := *p.Vision
	policy := *changed.Images
	policy.MaxImages = 5
	changed.Images = &policy
	p.Vision = &changed
	if validateNativeVisionPlan(p) == nil {
		t.Fatal("unreviewed count accepted")
	}
}
func TestReviewedNativeQwenVisionPlanAndHardPlacement(t *testing.T) {
	m := targetFixture(t, qwenVisionRepo)
	h := hostFixture()
	o := optsFixture()
	o.Backend = "native"
	o.Experimental = true
	o.Vision = true
	p, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	if p.Vision == nil || p.Vision.ID != "qwen38-27b-images-v1" || p.Vision.MaxImageTokens != 1024 || p.TextFeatures != nil || !contains(p.NativeRequirements, "numpy==2.3.5") {
		t.Fatal(p.Vision, p.NativeRequirements)
	}
	if e = validateNativeVisionPlan(p); e != nil {
		t.Fatal(e)
	}
	// Use a known full-checkpoint weight budget, not the tiny fixture parameter count.
	p.WeightResidentBytes = 55563006776
	h.GPUs[0].TotalMiB = 49140
	h.GPUs[0].FreeMiB = 48000
	if verifyWeightFit(p, h.GPUs) == nil {
		t.Fatal("full checkpoint placement accepted on A6000")
	}
	o.AllowRisk = true
	m.Hub.SHA = strings.Repeat("b", 40)
	if _, e = BuildPlan(m, h, o); e == nil {
		t.Fatal("unreviewed revision enabled vision")
	}
	profiles, _ := nativeVisionProfiles()
	for _, v := range profiles {
		if v.Images != nil && (validateImagePolicy(v.Images) != nil || v.ActivationAllowanceMiB != 4096) {
			t.Fatal("unsafe profile", v)
		}
	}
}
