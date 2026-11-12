package modelship

import (
	"context"
	"fmt"
	"time"
)

// Recheck immediately before launch: dependency installation and operator
// confirmation can take minutes, during which other workloads may consume VRAM.
// This is a check, not a reservation or an OOM guarantee.
func verifyWeightFitNow(ctx context.Context, p Plan) error {
	if p.WeightResidentBytes <= 0 {
		return nil
	} // Legacy saved deployments lack a budget.
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	h, err := DetectForBackend(ctx, "native")
	if err != nil {
		return fmt.Errorf("cannot refresh selected GPU VRAM before launch: %w", err)
	}
	return verifyWeightFit(p, h.GPUs)
}

func verifyWeightFit(p Plan, gpus []GPU) error {
	if p.WeightResidentBytes <= 0 {
		return nil
	}
	for _, gpu := range gpus {
		if gpu.UUID != p.GPU.UUID {
			continue
		}
		need := p.WeightResidentBytes / (1024 * 1024)
		if p.WeightResidentBytes%(1024*1024) != 0 {
			need++
		}
		if need > gpu.TotalMiB {
			return fmt.Errorf("weights alone require at least %d MiB, exceeding selected GPU total VRAM (%d MiB); re-plan on a larger GPU; --allow-risk cannot override", need, gpu.TotalMiB)
		}
		if need > gpu.FreeMiB {
			return fmt.Errorf("weights alone require at least %d MiB, exceeding currently free selected GPU VRAM (%d MiB); free memory or re-plan; --allow-risk cannot override", need, gpu.FreeMiB)
		}
		return nil
	}
	return fmt.Errorf("selected GPU identity is no longer present; re-plan and redeploy instead of changing devices implicitly")
}
