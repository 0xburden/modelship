package modelship

import (
	"fmt"
	"path/filepath"
	"strings"
)

const ggufRepo = "unsloth/Qwen3.8-27B-GGUF"
const ggufRevision = "4af51b3542f775b0a8e4b310bfb3f17e01c306c9"
const ggufFilename = "Qwen3.8-27B-UD-Q4_K_M.gguf"

type ArtifactPin struct {
	Name   string `json:"name"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
	URL    string `json:"url,omitempty"`
}
type GGUFSpec struct {
	File     ArtifactPin  `json:"file"`
	Metadata GGUFMetadata `json:"metadata"`
	Runtime  string       `json:"runtime"`
}

func ggufTarget(repo, revision, file string) (ArtifactPin, error) {
	if repo != ggufRepo || revision != ggufRevision || file != ggufFilename {
		return ArtifactPin{}, fmt.Errorf("experimental GGUF supports only %s @ %s with explicit --gguf-file %s", ggufRepo, ggufRevision, ggufFilename)
	}
	return ArtifactPin{Name: ggufFilename, Size: 16564259744, SHA256: "1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529"}, nil
}
func ggufOnly(m Model) bool {
	gguf, safe := false, false
	for _, f := range m.Hub.Siblings {
		gguf = gguf || strings.HasSuffix(f.Name, ".gguf")
		safe = safe || strings.HasSuffix(f.Name, ".safetensors")
	}
	return gguf && !safe
}
func ggufModelPath(p Plan) string {
	return filepath.Join(p.ModelCacheDir, "modelship-gguf", strings.ReplaceAll(p.Model, "/", "--"), p.Revision, p.GGUF.File.Name)
}
func validateGGUFPlan(p Plan) error {
	if p.GGUF == nil {
		return fmt.Errorf("missing GGUF deployment specification")
	}
	pin, e := ggufTarget(p.Model, p.Revision, p.GGUF.File.Name)
	if e != nil {
		return e
	}
	if p.Backend != "native" || p.Runtime != "llama.cpp" || p.RecipeID != "qwen35-gguf-text" || !p.DisableThinking || !p.TemplateCompletions || len(p.Files) != 1 || p.Files[0] != pin.Name || p.WeightBytes != pin.Size || p.Task != "text-generation" || p.GGUF.Runtime != "llama-b12110-cuda12.8" || p.GGUF.File.Size != pin.Size || p.GGUF.File.SHA256 != pin.SHA256 || p.GGUF.Metadata.Architecture != "qwen35" || p.GGUF.Metadata.TensorBytes <= 0 || p.GGUF.Metadata.TensorBytes > pin.Size || p.WeightResidentBytes != p.GGUF.Metadata.TensorBytes || p.Context < 128 || p.Context > 32768 || !filepath.IsAbs(p.ModelCacheDir) {
		return fmt.Errorf("invalid GGUF deployment policy/budgets")
	}
	return nil
}
func buildGGUFPlan(m Model, h Host, o Options) (Plan, error) {
	p := Plan{Schema: 1, Backend: o.Backend, Model: m.Hub.ID, Revision: m.Hub.SHA, License: m.Hub.CardData.License, Name: o.Name, Port: o.Port, Context: o.Context, Task: "text-generation", Runtime: "llama.cpp", NativeLoader: "llama.cpp", RecipeID: "qwen35-gguf-text", RecipeStatus: "experimental", DisableThinking: true, TemplateCompletions: true, DType: "GGUF UD-Q4_K_M", GGUF: m.GGUF, GPUMode: "native", GPUCount: len(h.GPUs), Listen: o.Listen, APIKeyFile: o.APIKeyFile, ModelCacheDir: h.ModelCacheDir}
	if o.Backend != "native" || o.GPUMode != "" && o.GPUMode != "auto" {
		return p, fmt.Errorf("GGUF is native-only; Docker GPU options are not supported")
	}
	if o.Research || o.TrustModelCode {
		return p, fmt.Errorf("GGUF does not execute repository Python; research/code flags do not apply to this recipe")
	}
	if o.Task != "auto" && o.Task != "" && o.Task != "text-generation" {
		return p, fmt.Errorf("first GGUF adapter is text-only; vision/projector/MTP selection is not enabled")
	}
	if o.GGUFFile == "" || m.GGUF == nil {
		return p, fmt.Errorf("select --gguf-file %s; a bounded GGUF header probe is required", ggufFilename)
	}
	if _, e := ggufTarget(m.Hub.ID, m.Hub.SHA, o.GGUFFile); e != nil {
		return p, e
	}
	if len(p.Name) > 63 || !containerName.MatchString(p.Name) || p.Port < 1024 || p.Port > 65535 || p.Context < 128 || p.Context > 32768 || int64(p.Context) > m.GGUF.Metadata.Context {
		return p, fmt.Errorf("invalid GGUF name/port/context (context range 128–32768, also bounded by model metadata)")
	}
	if !o.Experimental {
		p.Blockers = append(p.Blockers, "GGUF requires explicit --experimental opt-in")
	}
	if h.OS != "linux" || h.Arch != "amd64" {
		p.Blockers = append(p.Blockers, "GGUF runtime currently requires Linux amd64")
	}
	found := false
	for _, g := range h.GPUs {
		if g.Index == o.GPU {
			p.GPU = g
			found = true
		}
	}
	if !found {
		return p, fmt.Errorf("selected GPU index is absent")
	}
	if !driverAtLeast(p.GPU.Driver, 570, 26) || !computeAtLeast(p.GPU.Compute, 8, 0) {
		p.Blockers = append(p.Blockers, "pinned GGUF CUDA12.8 runtime requires driver>=570.26 and compute capability>=8.0")
	}
	p.Files = []string{m.GGUF.File.Name}
	p.WeightBytes = m.GGUF.File.Size
	p.WeightResidentBytes = m.GGUF.Metadata.TensorBytes
	p.WeightMemoryBasis = "GGUF encoded tensor-byte full-GPU placement budget; conservative for fixed CPU tensors, not measured VRAM or a peak guarantee"
	need := (p.WeightResidentBytes + (1 << 20) - 1) / (1 << 20)
	p.WeightsFitTotal = need <= p.GPU.TotalMiB
	p.WeightsFitFree = need <= p.GPU.FreeMiB
	if !p.WeightsFitTotal || !p.WeightsFitFree {
		p.Blockers = append(p.Blockers, "GGUF weight-placement budget exceeds selected GPU total/free VRAM; --allow-risk cannot override")
	}
	meta := m.GGUF.Metadata
	// Conservatively budget all advertised blocks as attention plus an 8GiB
	// runtime/recurrent-state/activation allowance. This is not a fit guarantee.
	kv := meta.Layers * int64(p.Context) * meta.KVHeads * (meta.KeyLength + meta.ValueLength) * 2
	p.EstimatedVRAMMiB = need + (kv+(1<<20)-1)/(1<<20) + 8192
	if p.EstimatedVRAMMiB > p.GPU.FreeMiB*90/100 {
		if o.AllowRisk {
			p.Warnings = append(p.Warnings, "OVERRIDDEN: conservative GGUF runtime/context budget")
		} else {
			p.Blockers = append(p.Blockers, "GGUF runtime/context estimate exceeds 90% of free VRAM; --allow-risk overrides only this heuristic")
		}
	}
	if !h.ModelCacheInspected {
		p.Blockers = append(p.Blockers, "GGUF cache filesystem/disk budget could not be verified")
	} else {
		missing := max(int64(0), p.WeightBytes-h.CachedWeightBytes)
		required := (missing+missing/20+(1<<20)-1)/(1<<20) + 12*1024
		if h.ModelCacheFreeMiB < required {
			p.Blockers = append(p.Blockers, "GGUF missing weights plus runtime/install/headroom budget exceed cache disk; --allow-risk cannot override")
		}
	}
	p.Warnings = append(p.Warnings, h.Warnings...)
	p.Warnings = append(p.Warnings, "Experimental text-only GGUF: no independent live GPU/model inference validation yet; no implicit mmproj, MTP, calibration, quant conversion or CPU fallback", "llama.cpp b12110 CUDA12.8 is separately checksum-pinned; no Python/PyTorch/driver installation or replacement", "Full GPU layer placement, device CUDA0, one slot, fixed context and --fit off are explicit; fixed backend CPU metadata/tensors can remain on host", "Source file bytes, encoded tensor-byte budget and runtime/host memory are distinct; context/recurrent state/temporary allocation may still OOM")
	if e := validateGGUFPlan(p); e != nil {
		return p, e
	}
	return p, nil
}
