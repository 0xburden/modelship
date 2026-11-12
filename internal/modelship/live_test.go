package modelship

import (
	"context"
	"os"
	"testing"
	"time"
)

// Explicit opt-in: downloads only Hub metadata/configuration, never model weights.
func TestLiveHubMetadata(t *testing.T) {
	if os.Getenv("MODELSHIP_LIVE_TEST") != "1" {
		t.Skip("set MODELSHIP_LIVE_TEST=1 for public Hugging Face metadata checks")
	}
	h := hostFixture()
	h.GPUs[0].Name = "Simulated A100 80GB (planning only)"
	h.GPUs[0].TotalMiB = 81920
	h.GPUs[0].FreeMiB = 80000
	h.GPUs[0].Compute = "8.0"
	for _, tc := range []struct{ id, runtime string }{
		{"Qwen/Qwen2.5-3B-Instruct", "vllm"},
		{"stabilityai/stable-diffusion-xl-base-1.0", "diffusers"},
	} {
		t.Run(tc.id, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			hub := NewHub()
			model, err := hub.Inspect(ctx, tc.id, "main")
			if err != nil {
				t.Fatal(err)
			}
			plan, err := BuildPlan(model, h, optsFixture())
			if err != nil {
				t.Fatal(err)
			}
			if plan.Runtime != tc.runtime || len(plan.Files) == 0 || len(plan.Blockers) > 0 {
				t.Fatalf("%+v", plan)
			}
			t.Logf("%s @ %s: %s, %d weights, %d bytes, %d MiB estimated VRAM (not GPU-validated)", plan.Model, plan.Revision, plan.Runtime, len(plan.Files), plan.WeightBytes, plan.EstimatedVRAMMiB)
		})
	}
}
