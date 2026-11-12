package modelship

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
)

// GPUArgs is shared by the diagnostic container and both inference runtimes.
// CDI's all alias is deliberately limited to one visible GPU by the planner
// and preflight, so it cannot silently grant access to unselected GPUs.
func GPUArgs(p Plan) []string {
	if p.GPUMode == "cdi" {
		return []string{"--device", "nvidia.com/gpu=all"}
	}
	return []string{"--gpus", "device=" + p.GPU.UUID}
}
func gpuProbe(ctx context.Context, p Plan) (string, string, error) {
	args := append([]string{"run", "--rm"}, GPUArgs(p)...)
	args = append(args, "--entrypoint", "nvidia-smi", "nvidia/cuda:12.8.1-base-ubuntu22.04", "--query-gpu=uuid", "--format=csv,noheader")
	cmd := exec.CommandContext(ctx, "docker", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = io.MultiWriter(os.Stdout, &stdout)
	cmd.Stderr = io.MultiWriter(os.Stderr, &stderr)
	err := cmd.Run()
	return stdout.String(), stderr.String(), err
}
func GPUPreflight(ctx context.Context, p Plan) (Plan, error) {
	if p.GPUMode == "" {
		p.GPUMode = "auto"
	}
	if p.GPUMode != "auto" && p.GPUMode != "nvidia" && p.GPUMode != "cdi" {
		return p, fmt.Errorf("invalid GPU access mode")
	}
	if p.GPUMode == "cdi" && p.GPUCount != 1 {
		return p, fmt.Errorf("CDI all-device mode requires exactly one visible GPU; refusing to expose unselected GPUs")
	}
	stdout, stderr, err := gpuProbe(ctx, p)
	if err != nil && ctx.Err() == nil && p.GPUMode == "auto" && strings.Contains(strings.ToLower(stderr), "thunder compute") && strings.Contains(stderr, "--device nvidia.com/gpu=all") {
		if p.GPUCount != 1 {
			return p, fmt.Errorf("Thunder Compute requires CDI all-device access, but %d GPUs are visible; refusing to expose unselected GPUs", p.GPUCount)
		}
		fmt.Println("Thunder Compute detected: retrying with --device nvidia.com/gpu=all (single-GPU host).")
		p.GPUMode = "cdi"
		stdout, _, err = gpuProbe(ctx, p)
	}
	if err != nil {
		return p, fmt.Errorf("GPU container preflight failed in %s mode: %w; check the output above and your provider's GPU container requirements (Thunder Compute: --gpu-mode cdi, not runtime reconfiguration)", p.GPUMode, err)
	}
	// Validate actual visibility, not just a successful docker exit. This also
	// catches local socket proxies pointing at a different GPU host.
	if strings.TrimSpace(stdout) != p.GPU.UUID {
		return p, fmt.Errorf("GPU preflight did not expose exactly selected GPU %s; refusing deployment", p.GPU.UUID)
	}
	if p.GPUMode == "auto" {
		p.GPUMode = "nvidia"
	}
	return p, nil
}
