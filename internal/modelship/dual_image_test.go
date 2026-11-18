package modelship

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestDualImagePlanAndPolicy(t *testing.T) {
	for _, pair := range [][3]string{
		{"stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b", "StableDiffusionXLPipeline"},
		{"Qwen/Qwen-Image-2.1", "d26bb61231c349cf6b7896fa83353113880e1ba3", "QwenImage21Pipeline"},
	} {
		t.Run(pair[2], func(t *testing.T) {
			m := imageFixture()
			m.Hub.ID, m.Hub.SHA, m.Index.Class = pair[0], pair[1], pair[2]
			h := hostFixture()
			h.GPUs[0].FreeMiB, h.GPUs[0].TotalMiB = 81920, 81920
			o := optsFixture()
			o.Backend, o.Experimental, o.Research = "native", true, true
			single, err := BuildPlan(m, h, o)
			if err != nil || single.Task != "text-to-image" || len(single.Blockers) > 0 {
				t.Fatalf("changed auto task: %+v / %v", single, err)
			}
			o.Task = dualImageTask
			dual, err := BuildPlan(m, h, o)
			if err != nil || dual.Task != dualImageTask || len(dual.Blockers) > 0 || dual.EstimatedVRAMMiB != single.EstimatedVRAMMiB+dualImageAllowanceMiB || dual.WeightResidentBytes != single.WeightResidentBytes || dual.WeightBytes != single.WeightBytes || !reflect.DeepEqual(single.Files, dual.Files) {
				t.Fatalf("changed checkpoint or incorrect dual budget: %+v / %v", dual, err)
			}
			r, _, _ := findRecipe(m)
			if got := imageTaskChoices(r, m, "native"); len(got) != len(r.Tasks)+1 || got[len(got)-1] != dualImageTask || contains(imageTaskChoices(r, m, "docker"), dualImageTask) {
				t.Fatalf("wrong task selector: %v", got)
			}
			if strings.HasPrefix(pair[0], "Qwen/") {
				missing := o
				missing.Experimental, missing.Research, missing.AllowRisk = false, false, true
				p, e := BuildPlan(m, h, missing)
				if e != nil || !strings.Contains(strings.Join(p.Blockers, " "), "--experimental") || !strings.Contains(strings.Join(p.Blockers, " "), "--research") {
					t.Fatalf("dual bypassed policy: %+v / %v", p, e)
				}
			}
			old := m.Hub.SHA
			m.Hub.SHA = strings.Repeat("a", 40)
			if _, e := BuildPlan(m, h, o); e == nil {
				t.Fatal("unreviewed revision accepted")
			}
			m.Hub.SHA = old
			o.Backend = "docker"
			if _, e := BuildPlan(m, h, o); e == nil {
				t.Fatal("unreviewed Docker support accepted")
			}
			o.Backend, o.AllowRisk = "native", true
			h.GPUs[0].FreeMiB = 32
			p, e := BuildPlan(m, h, o)
			if e != nil || !strings.Contains(strings.Join(p.Blockers, " "), "weights alone") {
				t.Fatalf("physical weight gate bypassed: %+v / %v", p, e)
			}
			t.Setenv("XDG_STATE_HOME", t.TempDir())
			dir, e := nativeDir(dual.Name)
			if e != nil {
				t.Fatal(e)
			}
			if e = privateDir(dir); e != nil {
				t.Fatal(e)
			}
			for _, schema := range []int{9, 10} {
				if e = writeJSONFile(filepath.Join(dir, "config.json"), NativeConfig{Schema: schema, Plan: dual}); e != nil {
					t.Fatal(e)
				}
				if _, _, e = nativeConfig(dual.Name); (e != nil) != (schema == 9) {
					t.Fatalf("dual schema %d: %v", schema, e)
				}
			}
			dual.NativeRequirements = nil
			if validateDualImagePlan(dual) == nil {
				t.Fatal("unversioned dual runtime accepted")
			}
		})
	}
}

func TestDualImageRequiresBothRoutesForReadiness(t *testing.T) {
	if imageHealthMatches(dualImageTask, nil) || imageHealthMatches(dualImageTask, []string{"image-to-image"}) || !imageHealthMatches(dualImageTask, []string{"text-to-image", "image-to-image"}) || !imageHealthMatches("image-to-image", nil) {
		t.Fatal("dual readiness must require both advertised routes; unchanged legacy")
	}
}
