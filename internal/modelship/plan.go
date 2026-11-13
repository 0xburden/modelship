package modelship

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const VLLMImage = "vllm/vllm-openai:v0.10.2"
const ImageRuntime = "modelship-diffusers:0.1.0"

type Options struct {
	Vision     bool
	MMProjFile string

	GGUFFile string

	Listen         string
	APIKeyFile     string
	Backend        string
	Python         string
	Task           string
	GPU            int
	GPUMode        string
	Context        int
	Port           int
	Name           string
	Revision       string
	AllowRisk      bool
	Experimental   bool
	Research       bool
	TrustModelCode bool
}
type Plan struct {
	Vision *NativeVisionProfile `json:"vision,omitempty"`
	GGUF   *GGUFSpec            `json:"gguf,omitempty"`

	Listen              string            `json:"listen,omitempty"`
	APIKeyFile          string            `json:"api_key_file,omitempty"`
	Schema              int               `json:"schema"`
	Backend             string            `json:"backend"`
	RecipeID            string            `json:"recipe_id,omitempty"`
	RecipeStatus        string            `json:"recipe_status,omitempty"`
	DType               string            `json:"dtype,omitempty"`
	NativeLoader        string            `json:"native_loader,omitempty"`
	MinPython           string            `json:"min_python,omitempty"`
	MinTorch            string            `json:"min_torch,omitempty"`
	ComponentDTypes     map[string]string `json:"component_dtypes,omitempty"`
	Research            bool              `json:"research,omitempty"`
	TrustModelCode      bool              `json:"trust_model_code,omitempty"`
	CodeRevision        string            `json:"code_revision,omitempty"`
	CodeHashes          map[string]string `json:"code_hashes,omitempty"`
	RequestedModel      string            `json:"requested_model,omitempty"`
	NativeRequirements  []string          `json:"native_requirements,omitempty"`
	TemplateCompletions bool              `json:"template_completions,omitempty"`
	DisableThinking     bool              `json:"disable_thinking,omitempty"`
	Python              string            `json:"python,omitempty"`
	Model               string            `json:"model"`
	Revision            string            `json:"revision"`
	License             any               `json:"license,omitempty"`
	Task                string            `json:"task"`
	Runtime             string            `json:"runtime"`
	Image               string            `json:"image"`
	Pipeline            string            `json:"pipeline,omitempty"`
	GPU                 GPU               `json:"gpu"`
	GPUMode             string            `json:"gpu_mode"`
	GPUCount            int               `json:"host_gpu_count"`
	Context             int               `json:"context,omitempty"`
	Port                int               `json:"port"`
	Name                string            `json:"name"`
	Files               []string          `json:"weight_files"`
	WeightBytes         int64             `json:"weight_download_bytes"`
	ModelCacheDir       string            `json:"model_cache_dir,omitempty"`
	WeightResidentBytes int64             `json:"weight_resident_bytes"`
	WeightMemoryBasis   string            `json:"weight_memory_basis"`
	WeightsFitTotal     bool              `json:"weights_fit_total"`
	WeightsFitFree      bool              `json:"weights_fit_free"`
	EstimatedVRAMMiB    int64             `json:"estimated_vram_mib"`
	GPUMemoryFraction   float64           `json:"gpu_memory_fraction,omitempty"`
	Warnings            []string          `json:"warnings"`
	Blockers            []string          `json:"blockers,omitempty"`
}

func BuildPlan(m Model, h Host, o Options) (Plan, error) {
	backend := o.Backend
	if backend == "" {
		backend = "docker"
	}
	if backend != "docker" && backend != "native" {
		return Plan{}, fmt.Errorf("--backend must be docker or native")
	}
	o.Backend = backend
	if err := accessOptions(&o); err != nil {
		return Plan{}, err
	}
	if ggufOnly(m) || o.GGUFFile != "" || m.GGUF != nil {
		return buildGGUFPlan(m, h, o)
	}
	if o.MMProjFile != "" {
		return Plan{}, fmt.Errorf("--mmproj-file applies only to GGUF; PyTorch vision uses its bundled image processor")
	}
	p := Plan{Backend: backend, Schema: 1, Model: m.Hub.ID, Revision: m.Hub.SHA, License: m.Hub.CardData.License, Port: o.Port, Name: o.Name, GPUMode: o.GPUMode, GPUCount: len(h.GPUs)}
	p.Listen, p.APIKeyFile = o.Listen, o.APIKeyFile
	if p.Listen == "0.0.0.0" {
		p.Warnings = append(p.Warnings, "Network listener enabled with bearer auth: use trusted HTTPS ingress; raw HTTP is not encrypted")
	}
	if p.GPUMode == "" {
		p.GPUMode = "auto"
	}
	if p.GPUMode != "auto" && p.GPUMode != "nvidia" && p.GPUMode != "cdi" {
		return p, fmt.Errorf("--gpu-mode must be auto, nvidia, or cdi")
	}
	if backend == "native" {
		if p.GPUMode != "auto" {
			return p, fmt.Errorf("--gpu-mode applies only to the Docker backend")
		}
		p.GPUMode = "native"
		p.Python = o.Python
		if p.Python == "" {
			p.Python = "python3"
		}
	}
	if p.GPUMode == "cdi" && len(h.GPUs) != 1 {
		return p, fmt.Errorf("CDI all-device mode is restricted to hosts with exactly one visible GPU; refusing to expose unselected GPUs")
	}
	recipe, known, err := findRecipe(m)
	if err != nil {
		return p, err
	}
	if !known {
		return p, fmt.Errorf("unsupported model architecture/pipeline %v / %q; use modelship inspect for metadata", m.Config.Architectures, m.Index.Class)
	}
	if err = validateRecipeShape(recipe, m); err != nil {
		return p, err
	}
	if recipe.Status == "planned" {
		return p, fmt.Errorf("recipe %s is recognized but not deployable yet: %s", recipe.ID, strings.Join(recipe.Blockers, "; "))
	}
	if !contains(recipe.Backends, backend) {
		return p, fmt.Errorf("recipe %s does not support backend %s", recipe.ID, backend)
	}
	p.RecipeID = recipe.ID
	p.RecipeStatus = recipe.Status
	p.DType = recipe.DType
	p.NativeLoader = recipe.NativeLoader
	p.MinPython = recipe.MinPython
	p.MinTorch = recipe.MinTorch
	if len(recipe.ComponentDTypes) > 0 {
		p.ComponentDTypes = make(map[string]string, len(recipe.ComponentDTypes))
		for component, dtype := range recipe.ComponentDTypes {
			p.ComponentDTypes[component] = dtype
		}
	}
	p.NativeRequirements = append([]string{}, recipe.NativeRequirements...)
	p.TemplateCompletions = recipe.TemplateCompletions
	p.DisableThinking = recipe.DisableThinking
	p.Warnings = append(p.Warnings, recipe.Notes...)
	if recipe.Status == "experimental" && !o.Experimental {
		p.Blockers = append(p.Blockers, "experimental recipe requires explicit --experimental opt-in")
	}
	applyRecipePolicy(&p, recipe, m, o)
	if backend == "native" {
		p.ModelCacheDir = h.ModelCacheDir
	}
	if o.Context < 128 || o.Context > 131072 {
		return p, fmt.Errorf("--context must be between 128 and 131072")
	}
	if o.Port < 1024 || o.Port > 65535 {
		return p, fmt.Errorf("--port must be between 1024 and 65535")
	}
	if len(o.Name) > 63 || !containerName.MatchString(o.Name) {
		return p, fmt.Errorf("invalid container name: use 1-63 letters, numbers, hyphens or underscores")
	}
	found := false
	for _, g := range h.GPUs {
		if g.Index == o.GPU {
			p.GPU = g
			found = true
			break
		}
	}
	if !found {
		return p, fmt.Errorf("GPU index %d not found", o.GPU)
	}
	if h.OS != "linux" || h.Arch != "amd64" {
		p.Blockers = append(p.Blockers, "deployment currently requires Linux amd64")
	}
	if backend == "docker" && !h.Docker {
		p.Blockers = append(p.Blockers, "Docker daemon is unavailable")
	}
	if backend == "docker" && h.DockerEndpoint != "" && !strings.HasPrefix(h.DockerEndpoint, "unix://") {
		p.Blockers = append(p.Blockers, "Docker must target a local Unix socket, not a remote server")
	}
	if backend == "docker" && !driverAtLeast(p.GPU.Driver, 570, 26) {
		p.Blockers = append(p.Blockers, "pinned CUDA 12.8 runtimes require NVIDIA driver >= 570.26 (conservative minimum)")
	}
	p.Warnings = append(p.Warnings, h.Warnings...)
	if backend == "native" {
		p.Warnings = append(p.Warnings, "Native backend reuses the selected Python installation's CUDA-enabled PyTorch through a system-site-packages venv; it never installs PyTorch. CUDA computation is checked before and after dependency installation.", "Native services default to 127.0.0.1; explicit network listeners require bearer auth. Workers run as your user without a sandbox. They survive SSH disconnection, but do not auto-restart after crashes or reboot.")
	}
	if p.GPUMode == "auto" {
		p.Warnings = append(p.Warnings, "GPU mode auto: try NVIDIA --gpus; fall back to CDI only for a recognized Thunder Compute rejection on a single-GPU host")
	}
	if p.GPUMode == "cdi" {
		p.Warnings = append(p.Warnings, "CDI mode uses --device nvidia.com/gpu=all and requires exactly one visible GPU", "Provider-managed Docker (including Thunder Compute) may not isolate container networking; localhost publishing alone may not protect the unauthenticated API. Verify network isolation before deployment.")
	}
	p.Warnings = append(p.Warnings,
		"Memory estimates are heuristics, not fit or throughput guarantees; downloads also require tokenizer/config files and runtime dependency disk space.",
		"Review the model license and accept any gated access terms before deployment.",
		"Only one GPU is used; multi-GPU sharding and automatic quantization are not implemented.")
	if recipe.MinCompute != "" {
		var major, minor int
		if _, err := fmt.Sscanf(recipe.MinCompute, "%d.%d", &major, &minor); err != nil || !computeAtLeast(p.GPU.Compute, major, minor) {
			p.Blockers = append(p.Blockers, "recipe requires compute capability >= "+recipe.MinCompute)
		}
	}
	var textKVBytes float64
	if recipe.Kind == "image" || recipe.Kind == "video" {
		minimum := recipe.MemoryFloorMiB
		p.Task = o.Task
		if p.Task == "auto" {
			p.Task = "text-to-image"
			if recipe.Kind == "video" {
				p.Task = "text-to-video"
			}
		}
		if recipe.Kind == "video" {
			if p.Task != "text-to-video" {
				return p, fmt.Errorf("this model requires --task text-to-video")
			}
		} else if p.Task != "text-to-image" && p.Task != "image-to-image" {
			return p, fmt.Errorf("this model requires --task text-to-image or image-to-image")
		}
		if !contains(recipe.Tasks, p.Task) {
			return p, fmt.Errorf("recipe %s does not support task %s", recipe.ID, p.Task)
		}
		p.Runtime = "diffusers"
		if backend == "docker" {
			p.Image = ImageRuntime
		}
		p.Pipeline = m.Index.Class
		if p.Pipeline == "FluxPipeline" {
			if !computeAtLeast(p.GPU.Compute, 8, 0) {
				p.Blockers = append(p.Blockers, "FLUX bfloat16 deployment requires compute capability >= 8.0")
			}
			p.Warnings = append(p.Warnings, "FLUX support is for standard FluxPipeline repositories, not FLUX.2, adapters, GGUF, or quantized variants; no CPU offload.")
		} else if !computeAtLeast(p.GPU.Compute, 7, 0) {
			p.Blockers = append(p.Blockers, "image runtime requires compute capability >= 7.0")
		}
		p.Files, p.WeightBytes = weightFiles(m.Hub.Siblings, true)
		// Files may be fp32 on disk while loading in half precision; retain a conservative estimate.
		p.EstimatedVRAMMiB = max64(minimum, ceilMiB(float64(p.WeightBytes)*0.65)+4096)
		if recipe.Kind == "video" {
			p.Warnings = append(p.Warnings, "Video budget assumes the service's bounded resolution/frame/step limits and one active job. Both experts stay resident; no offload or quantization.")
		} else {
			p.Warnings = append(p.Warnings, "Image memory estimate assumes one request at a time and <=1024x1024 images; larger shapes or long prompts can OOM.")
		}
	} else {
		if o.Task != "auto" && o.Task != "text-generation" {
			return p, fmt.Errorf("this is a Transformers model, not a supported Diffusers pipeline")
		}
		if !recipe.TextOnly && m.Hub.PipelineTag != "" && m.Hub.PipelineTag != "text-generation" {
			return p, fmt.Errorf("unsupported repository task %q", m.Hub.PipelineTag)
		}
		if q := strings.TrimSpace(string(m.Config.Quantization)); q != "" && q != "null" && q != "{}" {
			return p, fmt.Errorf("prequantized Transformers weights are not supported yet; use an unquantized safetensors repository")
		}
		if backend == "docker" && !computeAtLeast(p.GPU.Compute, 7, 5) {
			p.Blockers = append(p.Blockers, "text runtime requires compute capability >= 7.5")
		}
		p.Task = "text-generation"
		p.Runtime = "vllm"
		p.Image = VLLMImage
		if backend == "native" {
			p.Runtime = "transformers"
			p.Image = ""
			if !computeAtLeast(p.GPU.Compute, 7, 0) {
				p.Blockers = append(p.Blockers, "native float16 text runtime requires compute capability >= 7.0")
			}
		}
		p.Context = o.Context
		c := m.Config
		if c.TextConfig != nil {
			c = *c.TextConfig
		}
		// vLLM expresses its budget as a fraction of total VRAM. Respect memory
		// already used by other workloads rather than reserving 90% of total.
		p.GPUMemoryFraction = math.Floor(0.9*float64(p.GPU.FreeMiB)/float64(p.GPU.TotalMiB)*10000) / 10000
		if p.GPUMemoryFraction < 0.0001 {
			p.Blockers = append(p.Blockers, "no usable GPU memory budget")
		}
		if c.MaxPosition > 0 && o.Context > c.MaxPosition {
			return p, fmt.Errorf("--context %d exceeds model max_position_embeddings %d", o.Context, c.MaxPosition)
		}
		p.Files, p.WeightBytes = weightFiles(m.Hub.Siblings, false)
		weightMemory := float64(p.WeightBytes)
		if m.Hub.Safetensors.Total > 0 {
			weightMemory = float64(m.Hub.Safetensors.Total) * 2
		}
		headDim := c.HeadDim
		if headDim == 0 && c.Heads > 0 {
			headDim = c.Hidden / c.Heads
		}
		kvHeads := c.KVHeads
		if kvHeads == 0 {
			kvHeads = c.Heads
		}
		kv := float64(2) * float64(c.Layers) * float64(kvHeads) * float64(headDim) * float64(o.Context) * 2
		if kv <= 0 {
			kv = 2 * 1024 * 1024 * 1024
			p.Warnings = append(p.Warnings, "KV-cache dimensions missing; assuming 2 GiB for a single sequence")
		}
		textKVBytes = kv
		p.EstimatedVRAMMiB = ceilMiB(weightMemory*1.15+kv) + 2048
		p.Warnings = append(p.Warnings, "Text deployment uses "+p.DType+", eager mode and one concurrent sequence; optimized concurrency tuning is future work.")
		if backend == "native" {
			p.Warnings = append(p.Warnings, "Native Transformers API supports non-streaming completions/chat only; it is not a replacement for vLLM throughput or full OpenAI API compatibility.")
		}
	}
	if len(p.Files) == 0 || p.WeightBytes <= 0 {
		if !m.WeightMetadataChecked {
			return p, fmt.Errorf("no sized default safetensors weights found; single-file, variant-only and adapter repositories are not supported")
		}
	}
	var weightErr error
	p.WeightResidentBytes, p.WeightMemoryBasis, weightErr = residentWeightBytes(m, p.Files, p.WeightBytes, p.DType, p.ComponentDTypes)
	if weightErr != nil {
		p.Blockers = append(p.Blockers, "weights-alone VRAM preflight failed: "+weightErr.Error()+"; --allow-risk cannot override weight metadata safety")
	} else {
		// Compare integer bytes directly, without rounding a potentially huge
		// model through float64 or overflowing GPU MiB conversion.
		fits := func(mib int64) bool {
			return mib > 0 && (mib > math.MaxInt64/(1024*1024) || p.WeightResidentBytes <= mib*(1024*1024))
		}
		p.WeightsFitTotal = fits(p.GPU.TotalMiB)
		p.WeightsFitFree = fits(p.GPU.FreeMiB)
		if !p.WeightsFitTotal {
			p.Blockers = append(p.Blockers, fmt.Sprintf("weights alone require %d resident bytes, exceeding selected GPU total VRAM (%d MiB); --allow-risk cannot override", p.WeightResidentBytes, p.GPU.TotalMiB))
		}
		if !p.WeightsFitFree {
			p.Blockers = append(p.Blockers, fmt.Sprintf("weights alone require %d resident bytes, exceeding selected GPU currently free VRAM (%d MiB); --allow-risk cannot override", p.WeightResidentBytes, p.GPU.FreeMiB))
		}
		// The independent heuristic includes runtime overhead and remains the
		// only memory check --allow-risk may override.
		p.EstimatedVRAMMiB = max64(p.EstimatedVRAMMiB, ceilMiB(float64(p.WeightResidentBytes)*1.15+textKVBytes)+2048)
	}
	p.Warnings = append(p.Warnings, "Resident weight bytes are a requested-dtype lower bound: runtime buffers, modules retained in FP32, activations, caches and allocator overhead can require more. A fit flag is not an OOM guarantee.")
	if o.Vision {
		profile, e := nativeVisionFor(m)
		if e != nil {
			return p, e
		}
		p.Vision = profile
		if e = validateNativeVisionPlan(p); e != nil {
			return p, e
		}
		p.EstimatedVRAMMiB += profile.ActivationAllowanceMiB
		p.Warnings = append(p.Warnings, "Experimental PyTorch image understanding: one uploaded PNG/JPEG, 4 MiB / 4,194,304 pixels / 4096 per side. Gemma's reviewed PIL processor resizes to at most 280 image soft tokens; these share the fixed context. No URLs, audio/video/tools, quantization or CPU weight offload. The extra 2 GiB activation allowance is an unmeasured heuristic, not a peak-memory guarantee. Full checkpoint weights (including vision) are already counted; real image inference remains unvalidated.")
	}
	if backend == "native" && h.ModelCacheInspected {
		missing := p.WeightBytes - h.CachedWeightBytes
		if missing < 0 {
			missing = 0
		}
		budget := ceilMiB(float64(missing)*1.05) + 4096
		if h.ModelCacheFreeMiB < budget {
			p.Blockers = append(p.Blockers, fmt.Sprintf("native model-cache disk has %d MiB free; missing weights plus 5%%/4 GiB headroom need about %d MiB; choose a larger HF_HUB_CACHE filesystem; --allow-risk cannot override", h.ModelCacheFreeMiB, budget))
		}
	}
	diskBudget := ceilMiB(float64(p.WeightBytes)*1.2) + 20*1024
	if h.DockerDiskFreeMiB > 0 && h.DockerDiskFreeMiB < diskBudget {
		p.Warnings = append(p.Warnings, fmt.Sprintf("Docker disk has %d MiB free; budget roughly %d MiB for weights plus images/build cache (existing caches are not accounted for)", h.DockerDiskFreeMiB, diskBudget))
	}
	if h.RAMAvailableMiB > 0 && h.RAMAvailableMiB < ceilMiB(float64(p.WeightBytes))+4096 {
		p.Warnings = append(p.Warnings, "Available host RAM may be insufficient for loading; disk weight size plus 4 GiB is a rough budget, not a guaranteed requirement")
	}
	if p.EstimatedVRAMMiB > p.GPU.FreeMiB*90/100 {
		msg := fmt.Sprintf("estimated VRAM %d MiB exceeds 90%% of free GPU memory (%d MiB)", p.EstimatedVRAMMiB, p.GPU.FreeMiB)
		if o.AllowRisk {
			p.Warnings = append(p.Warnings, "OVERRIDDEN: "+msg)
		} else {
			p.Blockers = append(p.Blockers, msg+"; --allow-risk overrides only this memory heuristic")
		}
	}
	return p, nil
}
func weightFiles(files []ModelFile, image bool) ([]string, int64) {
	var names []string
	var total int64
	seen := map[string]bool{}
	for _, f := range files {
		if !strings.HasSuffix(f.Name, ".safetensors") || strings.Contains(f.Name, ".fp16.") || strings.Contains(f.Name, ".bf16.") {
			continue
		}
		parts := strings.Split(f.Name, "/")
		if !image && len(parts) != 1 {
			continue
		}
		if image && (len(parts) != 2 || !(parts[0] == "unet" || parts[0] == "transformer" || parts[0] == "transformer_2" || parts[0] == "vae" || strings.HasPrefix(parts[0], "text_encoder") || parts[0] == "safety_checker")) {
			continue
		}
		base := parts[len(parts)-1]
		if !(base == "model.safetensors" || base == "diffusion_pytorch_model.safetensors" || strings.HasPrefix(base, "model-") || strings.HasPrefix(base, "diffusion_pytorch_model-")) {
			continue
		}
		if f.Bytes() <= 0 || f.Bytes() > math.MaxInt64-total || seen[f.Name] {
			return nil, 0
		} // Never silently undercount an unknown shard.
		names = append(names, f.Name)
		seen[f.Name] = true
		total += f.Bytes()
	}
	sort.Strings(names)
	return names, total
}
func computeAtLeast(s string, major, minor int) bool { return driverAtLeast(s, major, minor) }
func ceilMiB(b float64) int64                        { return int64(math.Ceil(b / (1024 * 1024))) }
func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
