package modelship

import (
	"fmt"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

const (
	diffusionRepo     = "leejet/Qwen-Image-2.1-GGUF"
	diffusionRevision = "0bca76da6edd8a18211e9717454615c1bbaaf6ed"
	diffusionFreeMiB  = 40 << 10
)

var diffusionGPUUUID = regexp.MustCompile(`^GPU-[[:xdigit:]]{8}(-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12}$`)

type diffusionArtifact struct {
	Repo, Revision, RemotePath string
	ArtifactPin
}

func diffusionArtifacts() [3]diffusionArtifact {
	return [3]diffusionArtifact{
		{diffusionRepo, diffusionRevision, "qwen_image_2.1-Q4_K.gguf", ArtifactPin{Name: "qwen_image_2.1-Q4_K.gguf", Size: 4197494816, SHA256: "29f9c83c249ff0292fb2943fceddfa2319b446601866c82a4f8be062abea72c2"}},
		{"Qwen/Qwen3-VL-8B-Instruct-GGUF", "f982a07559d4a2f6c8744d840bf6fccab30eea96", "Qwen3VL-8B-Instruct-Q4_K_M.gguf", ArtifactPin{Name: "Qwen3VL-8B-Instruct-Q4_K_M.gguf", Size: 5027784800, SHA256: "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2"}},
		{"Comfy-Org/Qwen-Image-2.1", "aacf31a6d48e707f352ecba714cee0680da3479b", "vae/qwen_image_2.1_vae_bf16.safetensors", ArtifactPin{Name: "qwen_image_2.1_vae_bf16.safetensors", Size: 675509688, SHA256: "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"}},
	}
}

func diffusionWeights() ([]string, int64) {
	var names []string
	var bytes int64
	for _, pin := range diffusionArtifacts() {
		names = append(names, pin.Name)
		bytes += pin.Size
	}
	return names, bytes
}

func checkDiffusionGPU(g GPU) error {
	if !diffusionGPUUUID.MatchString(g.UUID) || !driverAtLeast(g.Driver, 570, 26) || !computeAtLeast(g.Compute, 8, 0) {
		return fmt.Errorf("GGUF diffusion requires a physical NVIDIA GPU, compute capability 8.0+, and driver 570.26+")
	}
	if g.TotalMiB < diffusionFreeMiB || g.FreeMiB < diffusionFreeMiB || g.FreeMiB > g.TotalMiB {
		return fmt.Errorf("GGUF diffusion requires 40 GiB of currently free VRAM on the selected GPU; --allow-risk cannot override this experimental admission gate")
	}
	return nil
}

// This profile is selected explicitly and built from an immutable, reviewed
// catalog. Do not infer diffusion support from a GGUF suffix, architecture, or
// metadata that happens to be returned for a different revision.
func buildDiffusionPlan(repo, revision string, h Host, o Options) (Plan, error) {
	if !strings.EqualFold(repo, diffusionRepo) || revision != diffusionRevision {
		return Plan{}, fmt.Errorf("GGUF diffusion requires %s with explicit --revision %s", diffusionRepo, diffusionRevision)
	}
	if o.Backend != "native" || (o.Task != "auto" && o.Task != diffusionTask && o.Task != "") ||
		(o.GPUMode != "auto" && o.GPUMode != "") || o.AllowRisk || o.Vision || o.MMProjFile != "" || o.GGUFFile != "" || o.TrustModelCode {
		return Plan{}, fmt.Errorf("first GGUF diffusion profile is native generation only; text/vision GGUF, edit, risk override and custom-code flags do not apply")
	}
	cache := h.ModelCacheDir
	if cache == "" {
		var err error
		cache, err = modelCachePath()
		if err != nil {
			return Plan{}, err
		}
	}
	names, bytes := diffusionWeights()
	p := Plan{Schema: 1, Backend: "native", Model: diffusionRepo, Revision: diffusionRevision, License: "qwen-research",
		RecipeID: diffusionProfileID, RecipeStatus: "experimental", DType: "Q4_K diffusion, Q4_K_M encoder, BF16 VAE", Runtime: diffusionRuntime,
		NativeLoader: diffusionRuntime, Task: diffusionTask, Research: o.Research, GPUMode: "native", GPUCount: len(h.GPUs),
		Name: o.Name, Port: o.Port, Listen: o.Listen, APIKeyFile: o.APIKeyFile, ModelCacheDir: cache, Files: names, WeightBytes: bytes,
		WeightResidentBytes: bytes, WeightMemoryBasis: "encoded artifact placement budget, not peak VRAM", Diffusion: diffusionProfileID,
		Warnings: []string{"Fixed 512×512, 40 Euler steps and guidance 6; generation only; no edit or LoRA.",
			"A dedicated idle GPU with 40 GiB free is required. This is a conservative admission threshold, not measured peak VRAM or a fit guarantee."}}
	if p.Listen == "0.0.0.0" {
		p.Warnings = append(p.Warnings, "Network listener enabled with bearer auth: raw HTTP is not encrypted; use trusted HTTPS ingress or an SSH tunnel.")
	}
	if !o.Experimental {
		p.Blockers = append(p.Blockers, "GGUF diffusion requires separate --experimental opt-in")
	}
	if !o.Research {
		p.Blockers = append(p.Blockers, "GGUF diffusion requires --research acknowledgement of the Qwen Research License")
	}
	if h.OS != "linux" || h.Arch != "amd64" {
		p.Blockers = append(p.Blockers, "GGUF diffusion requires Linux amd64 with a compatible glibc userspace")
	}
	found := false
	for _, gpu := range h.GPUs {
		if gpu.Index == o.GPU {
			p.GPU, found = gpu, true
			break
		}
	}
	if !found {
		return p, fmt.Errorf("selected GPU index is absent")
	}
	if err := checkDiffusionGPU(p.GPU); err != nil {
		p.Blockers = append(p.Blockers, err.Error())
	}
	need := (bytes + (1 << 20) - 1) >> 20
	p.WeightsFitTotal, p.WeightsFitFree = p.GPU.TotalMiB >= need, p.GPU.FreeMiB >= need
	if p.APIKeyFile == "" {
		p.Blockers = append(p.Blockers, "GGUF diffusion requires --api-key-file even on loopback")
	}
	if len(p.Name) > 63 || !containerName.MatchString(p.Name) || p.Port < 1024 || p.Port > 65535 || !filepath.IsAbs(cache) {
		return p, fmt.Errorf("invalid GGUF diffusion name, port or cache path")
	}
	return p, nil
}

func validateDiffusionPlan(p Plan) error {
	names, bytes := diffusionWeights()
	if p.Diffusion != diffusionProfileID || p.Runtime != diffusionRuntime || p.NativeLoader != diffusionRuntime ||
		p.Model != diffusionRepo || p.Revision != diffusionRevision || p.RecipeID != diffusionProfileID || p.RecipeStatus != "experimental" ||
		p.Backend != "native" || p.GPUMode != "native" || p.Task != diffusionTask || !p.Research || p.APIKeyFile == "" ||
		p.Vision != nil || p.TextFeatures != nil || p.GGUF != nil || p.Context != 0 || p.Python != "" || p.Pipeline != "" ||
		p.TrustModelCode || p.CodeRevision != "" || len(p.CodeHashes) != 0 || len(p.NativeRequirements) != 0 || len(p.Blockers) != 0 ||
		p.WeightBytes != bytes || p.WeightResidentBytes != bytes || !slices.Equal(p.Files, names) || !filepath.IsAbs(p.ModelCacheDir) ||
		len(p.Name) > 63 || !containerName.MatchString(p.Name) || p.Port < 1024 || p.Port > 65535 {
		return fmt.Errorf("invalid reviewed GGUF diffusion deployment")
	}
	return checkDiffusionGPU(p.GPU)
}
