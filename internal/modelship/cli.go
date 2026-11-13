package modelship

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"
)

const help = `modelship — NVIDIA model deployment (Linux amd64)

Commands:
  launch                             Guided model setup (also the interactive default)
  catalog [--json]                    List implemented, experimental, and planned recipes
  inspect MODEL [--gguf-file FILE] [--json]              Inspect metadata without a GPU or model download
  doctor                             Inspect NVIDIA GPUs and Docker
  setup                              Print prerequisite installation guidance
  plan MODEL [options]               Resolve weights and print a deployment plan
  deploy MODEL [options]             Confirm, install runtime, and serve a model
  status|logs|stop|start --name NAME  Manage a deployment (add --backend native for native)
  uninstall --name NAME --yes        Remove deployment (keep model caches)
  smoke --name NAME                 Generate a short text completion
  auth-keygen [--file PATH]          Create a private API key (never printed)
  configure --backend native --name NAME --listen IP --api-key-file PATH
  version

MODEL is org/repo or https://huggingface.co/org/repo.

Plan/deploy options:
  --task auto|text-generation|text-to-image|image-to-image|text-to-video
  --revision main   --gpu 0   --context 4096   --port 8000
  --backend docker|native   --python python3 (Python-based native recipes; GGUF uses llama.cpp)
  --listen 127.0.0.1|0.0.0.0 --api-key-file PATH (native only; public binds require auth)
  --experimental     Explicitly opt in to an experimental implemented recipe
  --research         Acknowledge non-commercial research/evaluation use (Qwen Image)
  --trust-model-code Explicit approval of the pinned, hash-verified MOSS Python code
  --gguf-file FILE  Explicit GGUF variant; no automatic selection or conversion
  --weights-repo ID  Explicit official WAN Diffusers conversion; never inferred
  --name modelship  --allow-risk   --gpu-mode auto|nvidia|cdi (Docker)
  --json            Print machine-readable plan (plan only)
  --yes             Skip confirmation (deploy only)
  --wait 30m        Readiness timeout; 0 skips waiting (deploy only)

Use HF_TOKEN for gated/private models (separate from API auth). Native defaults to 127.0.0.1;
Docker publishes localhost ports subject to the provider's network isolation.
No driver installation, automatic repository-code execution, unsafe checkpoint
loading, arbitrary model conversion, or automatic public exposure. Safetensors
and selected GGUF headers check weight-placement budgets; activation/cache estimates are still heuristics.
`

func Run(args []string, version string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if len(args) == 0 {
		if terminalFile(os.Stdin) && terminalFile(os.Stdout) {
			return launchCommand(ctx, nil)
		}
		fmt.Print(help)
		return nil
	}
	if args[0] == "help" || args[0] == "--help" || args[0] == "-h" {
		fmt.Print(help)
		return nil
	}
	switch args[0] {
	case "launch":
		return launchCommand(ctx, args[1:])
	case "__native-run":
		if len(args) != 3 {
			return fmt.Errorf("internal native-run requires name and run ID")
		}
		return NativeExec(ctx, args[1], args[2])
	case "auth-keygen":
		return authKeygenCommand(args[1:])
	case "configure":
		return configureNativeCommand(ctx, args[1:])
	case "catalog":
		return catalogCommand(args[1:])
	case "inspect":
		return inspectCommand(ctx, args[1:])
	case "version":
		fmt.Println(version)
		return nil
	case "setup":
		fmt.Print(`Prerequisites (Ubuntu 22.04/24.04, NVIDIA GPU):
  1. Install a supported NVIDIA driver >=570.26; reboot if required.
     https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/
  2. Install Docker Engine and ensure this user can access its local daemon.
     https://docs.docker.com/engine/install/ubuntu/
  3. Standard NVIDIA Docker: install NVIDIA Container Toolkit, then configure Docker:
       sudo nvidia-ctk runtime configure --runtime=docker
       sudo systemctl restart docker
     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
     Thunder Compute: prefer --backend native with working host CUDA PyTorch.
     Its Docker/CDI path is experimental; do NOT install/reconfigure its runtime.
     Native mode does not need Docker or NVIDIA Container Toolkit.
  4. Run: modelship doctor

Docker access is effectively root access. Do not add untrusted users to its group.
modelship does not replace drivers or run remote installation scripts as root.
Allow disk headroom for container images plus weights; ensure adequate host RAM.
`)
		return nil
	case "doctor":
		c, cancel := context.WithTimeout(ctx, 20*time.Second)
		defer cancel()
		fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
		backend := fs.String("backend", "docker", "docker or native")
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		if fs.NArg() != 0 || (*backend != "docker" && *backend != "native") {
			return fmt.Errorf("doctor supports --backend docker|native")
		}
		h, err := DetectForBackend(c, *backend)
		if e := printJSON(h); e != nil {
			return e
		}
		return err
	case "plan", "deploy":
		return planCommand(ctx, args[0], args[1:])
	case "status", "logs", "stop", "start", "uninstall", "smoke":
		fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
		name := fs.String("name", "modelship", "deployment name")
		backend := fs.String("backend", "docker", "docker or native")
		yes := fs.Bool("yes", false, "confirm removal")
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		if fs.NArg() != 0 {
			return fmt.Errorf("unexpected positional arguments")
		}
		if args[0] == "uninstall" && !*yes {
			return fmt.Errorf("uninstall requires --yes; model caches are retained")
		}
		if *backend == "native" {
			if args[0] == "smoke" {
				return SmokeNative(ctx, *name)
			}
			return ManageNative(ctx, args[0], *name)
		}
		if *backend != "docker" {
			return fmt.Errorf("--backend must be docker or native")
		}
		if args[0] == "smoke" {
			return Smoke(ctx, *name)
		}
		return Manage(ctx, args[0], *name)
	default:
		return fmt.Errorf("unknown command %q; run modelship help", args[0])
	}
}
func planCommand(ctx context.Context, action string, args []string) error {
	fs := flag.NewFlagSet(action, flag.ContinueOnError)
	o := Options{}
	fs.StringVar(&o.Backend, "backend", "docker", "docker or native")
	fs.StringVar(&o.GGUFFile, "gguf-file", "", "explicit approved standalone GGUF filename (native experimental only)")
	fs.StringVar(&o.Listen, "listen", "", "native IPv4 listener (default 127.0.0.1); 0.0.0.0 requires --api-key-file")
	fs.StringVar(&o.APIKeyFile, "api-key-file", "", "private bearer key file; never a literal key value")
	fs.StringVar(&o.Python, "python", "python3", "native base Python with working CUDA PyTorch")
	fs.StringVar(&o.Task, "task", "auto", "model task")
	fs.StringVar(&o.Revision, "revision", "main", "Hugging Face revision")
	fs.IntVar(&o.GPU, "gpu", 0, "physical NVIDIA GPU index")
	fs.StringVar(&o.GPUMode, "gpu-mode", "auto", "GPU container access: auto, nvidia, or cdi (single GPU)")
	fs.IntVar(&o.Context, "context", 4096, "maximum text context length")
	fs.IntVar(&o.Port, "port", 8000, "localhost port")
	fs.StringVar(&o.Name, "name", "modelship", "container name")
	fs.BoolVar(&o.AllowRisk, "allow-risk", false, "override VRAM heuristic only")
	fs.BoolVar(&o.Experimental, "experimental", false, "opt in to experimental implemented recipes")
	fs.BoolVar(&o.Research, "research", false, "acknowledge non-commercial research/evaluation use under the model license")
	fs.BoolVar(&o.TrustModelCode, "trust-model-code", false, "approve reviewed, revision-pinned and SHA-256-verified MOSS repository Python")
	weightsRepo := fs.String("weights-repo", "", "explicit supported weights repository; --revision applies to this selected repository")
	asJSON := fs.Bool("json", false, "machine-readable plan")
	yes := fs.Bool("yes", false, "skip confirmation")
	wait := fs.Duration("wait", 30*time.Minute, "readiness timeout")
	// Support both 'plan repo --flags' and 'plan --flags repo'.
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		args = append(append([]string{}, args[1:]...), args[0])
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("%s requires one model URL or repo ID", action)
	}
	if *wait < 0 {
		return fmt.Errorf("--wait cannot be negative")
	}
	if action == "deploy" && *asJSON {
		return fmt.Errorf("--json is supported only with plan")
	}
	if o.GPUMode != "auto" && o.GPUMode != "nvidia" && o.GPUMode != "cdi" {
		return fmt.Errorf("--gpu-mode must be auto, nvidia, or cdi")
	}
	if o.Task != "auto" && o.Task != "text-generation" && o.Task != "text-to-image" && o.Task != "image-to-image" && o.Task != "text-to-video" {
		return fmt.Errorf("invalid --task")
	}
	if o.Backend != "docker" && o.Backend != "native" {
		return fmt.Errorf("--backend must be docker or native")
	}
	if err := accessOptions(&o); err != nil {
		return err
	}
	id, err := ParseModel(fs.Arg(0))
	if err != nil {
		return err
	}
	requestedID := id
	id, err = resolveWeightRepo(requestedID, *weightsRepo)
	if err != nil {
		return err
	}
	hctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	host, err := DetectForBackend(hctx, o.Backend)
	cancel()
	if err != nil {
		return err
	}
	model, err := inspectForPlan(ctx, id, o.Revision)
	if err != nil {
		return err
	}
	if o.GGUFFile != "" {
		hub := NewHub()
		hub.Token = os.Getenv("HF_TOKEN")
		if err = hub.inspectGGUF(ctx, &model, o.GGUFFile); err != nil {
			return err
		}
	}
	if o.Backend == "native" {
		InspectModelCache(&host, model)
	}
	p, err := BuildPlan(model, host, o)
	if err != nil {
		return err
	}
	if requestedID != id {
		p.RequestedModel = requestedID
	}
	if *asJSON {
		if err = printJSON(p); err != nil {
			return err
		}
	} else {
		printPlan(p)
	}
	if action == "plan" {
		if len(p.Blockers) > 0 {
			return fmt.Errorf("plan contains deployment blockers")
		}
		return nil
	}
	if len(p.Blockers) > 0 {
		return fmt.Errorf("resolve blockers before deploying")
	}
	if !*yes {
		fmt.Fprint(os.Stderr, "Download/build runtime and deploy this plan? [y/N] ")
		answer, err := bufio.NewReader(os.Stdin).ReadString('\n')
		if err != nil {
			return fmt.Errorf("confirmation unavailable; use --yes for noninteractive deployment")
		}
		if strings.ToLower(strings.TrimSpace(answer)) != "y" {
			return fmt.Errorf("deployment cancelled")
		}
	}
	return Deploy(ctx, p, *wait)
}
func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
func printPlan(p Plan) { writePlan(os.Stdout, p) }
func writePlan(w io.Writer, p Plan) {
	if p.RequestedModel != "" {
		fmt.Fprintf(w, "Requested:   %s\nExplicit weights repository follows:\n", p.RequestedModel)
	}
	fmt.Fprintf(w, "Model:       %s\nRevision:    %s\nTask:        %s\nRuntime:     %s (%s)\nGPU:         %d · %s · %d MiB free\nWeights:     %d files · %.2f GiB (plus supporting files)\nEst. VRAM:   %d MiB\nEndpoint:    http://127.0.0.1:%d\nDeployment:  %s\n",
		p.Model, p.Revision, p.Task, p.Runtime, p.Image, p.GPU.Index, p.GPU.Name, p.GPU.FreeMiB, len(p.Files), float64(p.WeightBytes)/(1024*1024*1024), p.EstimatedVRAMMiB, p.Port, p.Name)
	fmt.Fprintf(w, "Weight VRAM: %.2f GiB (%s); fits total=%t, fits free=%t\n", float64(p.WeightResidentBytes)/(1024*1024*1024), p.WeightMemoryBasis, p.WeightsFitTotal, p.WeightsFitFree)
	if p.ModelCacheDir != "" {
		fmt.Fprintf(w, "Model cache: %s\n", p.ModelCacheDir)
	}
	if p.GGUF != nil {
		fmt.Fprintf(w, "GGUF file:   %s (%d bytes); SHA-256 %s\nRuntime pin: %s; CUDA0/all GPU layers; auto-fit disabled; text-only\n", p.GGUF.File.Name, p.GGUF.File.Size, p.GGUF.File.SHA256, p.GGUF.Runtime)
	}
	if p.Context > 0 {
		fmt.Fprintf(w, "Context:     %d tokens\n", p.Context)
	}
	fmt.Fprintf(w, "Backend:     %s\n", p.Backend)
	if p.Backend == "native" {
		fmt.Fprintf(w, "Listen:      %s; bearer authentication=%t\n", nativeListen(p), p.APIKeyFile != "")
	}
	if p.RecipeID != "" {
		fmt.Fprintf(w, "Recipe:      %s (%s), %s\n", p.RecipeID, p.RecipeStatus, p.DType)
	}
	if p.Research {
		fmt.Fprintln(w, "Use:         non-commercial research/evaluation acknowledged; review exact model terms")
	}
	if p.CodeRevision != "" {
		fmt.Fprintf(w, "Code review: %s; execution approved=%t (runs as your user, no sandbox)\n", p.CodeRevision, p.TrustModelCode)
		names := make([]string, 0, len(p.CodeHashes))
		for name := range p.CodeHashes {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			fmt.Fprintf(w, "  SHA-256 %s  %s\n", p.CodeHashes[name], name)
		}
	}
	if p.Python != "" {
		fmt.Fprintf(w, "Base Python: %s\n", p.Python)
	}
	fmt.Fprintf(w, "GPU access:  %s\n", p.GPUMode)
	fmt.Fprintln(w, "Weight files:")
	for _, f := range p.Files {
		fmt.Fprintln(w, "  "+terminalText(f))
	}
	for _, warning := range p.Warnings {
		fmt.Fprintln(w, "Warning: "+terminalText(warning))
	}
	for _, b := range p.Blockers {
		fmt.Fprintln(w, "BLOCKED: "+terminalText(b))
	}
}

// Shared metadata/weight inspection for both guided and flag-based planning.
func inspectForPlan(ctx context.Context, id, revision string) (Model, error) {
	hub := NewHub()
	hub.Token = os.Getenv("HF_TOKEN")
	model, err := hub.Inspect(ctx, id, revision)
	if err != nil {
		return model, err
	}
	// Record errors as non-overridable blockers rather than silently guessing.
	if !ggufOnly(model) {
		_ = hub.InspectWeightMetadata(ctx, &model)
	}
	return model, nil
}
