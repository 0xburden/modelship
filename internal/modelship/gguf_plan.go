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
type GGUFVisionSpec struct {
	File     ArtifactPin  `json:"file"`
	Metadata GGUFMetadata `json:"metadata"`
}

type GGUFSpec struct {
	Vision   *GGUFVisionSpec `json:"vision,omitempty"`
	File     ArtifactPin     `json:"file"`
	Metadata GGUFMetadata    `json:"metadata"`
	Runtime  string          `json:"runtime"`
}

func ggufTarget(repo, revision, file string) (ArtifactPin, error) {
	return ggufArtifact(repo, revision, file, false)
}
func ggufOnly(m Model) bool {
	gguf, safe := false, false
	for _, f := range m.Hub.Siblings {
		gguf = gguf || strings.HasSuffix(f.Name, ".gguf")
		safe = safe || strings.HasSuffix(f.Name, ".safetensors")
	}
	return gguf && !safe
}
func ggufModelPath(p Plan) string { return ggufArtifactPath(p, p.GGUF.File.Name) }
func validateGGUFPlan(p Plan) error {
	if p.GGUF == nil {
		return fmt.Errorf("missing GGUF deployment specification")
	}
	pin, e := ggufTarget(p.Model, p.Revision, p.GGUF.File.Name)
	if e != nil {
		return e
	}
	if p.Backend != "native" || p.Runtime != "llama.cpp" || p.RecipeID != "qwen35-gguf-text" || !p.DisableThinking || !p.TemplateCompletions || len(p.Files) < 1 || p.Files[0] != pin.Name || p.Task != "text-generation" || p.GGUF.File.Size != pin.Size || p.GGUF.File.SHA256 != pin.SHA256 || p.GGUF.Metadata.Architecture != "qwen35" || p.GGUF.Metadata.TensorBytes <= 0 || p.GGUF.Metadata.TensorBytes > pin.Size || p.Context < 128 || p.Context > 32768 || !filepath.IsAbs(p.ModelCacheDir) {
		return fmt.Errorf("invalid GGUF deployment policy/budgets")
	}
	bytes, resident := ggufBudgets(p.GGUF)
	selected := ggufSelectedArtifacts(p.GGUF)
	if p.WeightBytes != bytes || p.WeightResidentBytes != resident || len(p.Files) != len(selected) {
		return fmt.Errorf("GGUF selected artifact budgets do not match saved metadata")
	}
	for i, pin := range selected {
		if p.Files[i] != pin.Name {
			return fmt.Errorf("GGUF artifact list mismatch")
		}
	}
	if p.GGUF.Vision != nil {
		if p.GGUF.Runtime != llamaJammyRuntime {
			return fmt.Errorf("vision requires the reviewed Ubuntu22.04 runtime profile")
		}
		if e := validateGGUFVision(p.GGUF.Vision); e != nil {
			return e
		}
	}
	if _, e := loadLlamaRuntime(p.GGUF.Runtime); e != nil {
		return e
	}
	return nil
}
func buildGGUFPlan(m Model, h Host, o Options) (Plan, error) {
	p := Plan{Schema: 1, Backend: o.Backend, Model: m.Hub.ID, Revision: m.Hub.SHA, License: m.Hub.CardData.License, Name: o.Name, Port: o.Port, Context: o.Context, Task: "text-generation", Runtime: "llama.cpp", NativeLoader: "llama.cpp", RecipeID: "qwen35-gguf-text", RecipeStatus: "experimental", DisableThinking: true, TemplateCompletions: true, DType: "GGUF " + strings.TrimSuffix(strings.TrimPrefix(o.GGUFFile, "Qwen3.8-27B-"), ".gguf"), GGUF: m.GGUF, GPUMode: "native", GPUCount: len(h.GPUs), Listen: o.Listen, APIKeyFile: o.APIKeyFile, ModelCacheDir: h.ModelCacheDir}
	if o.Backend != "native" || o.GPUMode != "" && o.GPUMode != "auto" {
		return p, fmt.Errorf("GGUF is native-only; Docker GPU options are not supported")
	}
	if o.Research || o.TrustModelCode {
		return p, fmt.Errorf("GGUF does not execute repository Python; research/code flags do not apply to this recipe")
	}
	if o.Task != "auto" && o.Task != "" && o.Task != "text-generation" {
		return p, fmt.Errorf("GGUF produces text only; image understanding uses --vision, not image/video generation tasks")
	}
	if o.GGUFFile == "" || m.GGUF == nil {
		return p, fmt.Errorf("select --gguf-file %s; a bounded GGUF header probe is required", ggufFilename)
	}
	if o.Vision != (m.GGUF.Vision != nil) || (!o.Vision && o.MMProjFile != "") {
		return p, fmt.Errorf("vision requires explicit --vision --mmproj-file selection and reviewed projector header inspection")
	}
	if o.Vision && (o.MMProjFile == "" || o.MMProjFile != m.GGUF.Vision.File.Name) {
		return p, fmt.Errorf("select the reviewed --mmproj-file %s", ggufProjectorFile)
	}
	pin, e := ggufTarget(m.Hub.ID, m.Hub.SHA, o.GGUFFile)
	if e != nil {
		return p, e
	}
	if pin != m.GGUF.File {
		return p, fmt.Errorf("selected GGUF differs from inspected immutable artifact")
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
	p.Files = nil
	for _, pin := range ggufSelectedArtifacts(m.GGUF) {
		p.Files = append(p.Files, pin.Name)
	}
	p.WeightBytes, p.WeightResidentBytes = ggufBudgets(m.GGUF)
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
	if m.GGUF.Vision != nil {
		p.EstimatedVRAMMiB += 2048
		p.Warnings = append(p.Warnings, "Vision: one uploaded PNG/JPEG, 4 MiB/4,194,304-pixel input budget, at most 1024 image tokens, with upstream preprocessing/resizing; image tokens share the fixed context. The extra 2 GiB activation allowance is a heuristic, not measured peak memory.")
	}
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
	runtimeSpec, e := loadLlamaRuntime(p.GGUF.Runtime)
	if e != nil {
		return p, e
	}
	p.Warnings = append(p.Warnings, fmt.Sprintf("Runtime %s requires system GLIBC_%s / GLIBCXX_%s; checked before runtime downloads, with no host library replacement", runtimeSpec.ID, runtimeSpec.MinimumGLIBC, runtimeSpec.MinimumGLIBCXX))
	p.Warnings = append(p.Warnings, h.Warnings...)
	p.Warnings = append(p.Warnings, "Experimental GGUF: no independent live GPU/model inference validation for this artifact/mode; no implicit projector, MTP, calibration, quant conversion or CPU fallback", "llama.cpp b12110 CUDA12.8 is separately checksum-pinned; no Python/PyTorch/driver installation or replacement", "Full GPU layer placement, device CUDA0, one slot, fixed context and --fit off are explicit; fixed backend CPU metadata/tensors can remain on host", "Source file bytes, encoded tensor-byte budget and runtime/host memory are distinct; context/recurrent state/temporary allocation may still OOM")
	if e := validateGGUFPlan(p); e != nil {
		return p, e
	}
	return p, nil
}
