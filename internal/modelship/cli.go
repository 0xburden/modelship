package modelship

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

const help = `modelship — NVIDIA model deployment (Linux amd64)

Commands:
  catalog [--json]                    List implemented, experimental, and planned recipes
  inspect MODEL [--json]              Inspect metadata without a GPU or model download
  doctor                             Inspect NVIDIA GPUs and Docker
  setup                              Print prerequisite installation guidance
  plan MODEL [options]               Resolve weights and print a deployment plan
  deploy MODEL [options]             Confirm, install runtime, and serve a model
  status|logs|stop|start --name NAME  Manage a deployment (add --backend native for native)
  uninstall --name NAME --yes        Remove deployment (keep model caches)
  smoke --name NAME                 Generate a short text completion
  version

MODEL is org/repo or https://huggingface.co/org/repo.

Plan/deploy options:
  --task auto|text-generation|text-to-image|image-to-image
  --revision main   --gpu 0   --context 4096   --port 8000
  --backend docker|native   --python python3 (native base with working PyTorch)
  --experimental     Explicitly opt in to an experimental implemented recipe
  --name modelship  --allow-risk   --gpu-mode auto|nvidia|cdi (Docker)
  --json            Print machine-readable plan (plan only)
  --yes             Skip confirmation (deploy only)
  --wait 30m        Readiness timeout; 0 skips waiting (deploy only)

Use HF_TOKEN for gated/private models. Native services bind directly to 127.0.0.1;
Docker publishes localhost ports subject to the provider's network isolation.
No driver installation, remote model code, arbitrary model conversion, or
automatic public exposure. Plans are estimates, not compatibility guarantees.
`

func Run(args []string, version string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if len(args) == 0 || args[0] == "help" || args[0] == "--help" || args[0] == "-h" {
		fmt.Print(help)
		return nil
	}
	switch args[0] {
	case "__native-run":
		if len(args) != 3 {
			return fmt.Errorf("internal native-run requires name and run ID")
		}
		return NativeExec(args[1], args[2])
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
	if o.Task != "auto" && o.Task != "text-generation" && o.Task != "text-to-image" && o.Task != "image-to-image" {
		return fmt.Errorf("invalid --task")
	}
	if o.Backend != "docker" && o.Backend != "native" {
		return fmt.Errorf("--backend must be docker or native")
	}
	id, err := ParseModel(fs.Arg(0))
	if err != nil {
		return err
	}
	hctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	host, err := DetectForBackend(hctx, o.Backend)
	cancel()
	if err != nil {
		return err
	}
	hub := NewHub()
	hub.Token = os.Getenv("HF_TOKEN")
	model, err := hub.Inspect(ctx, id, o.Revision)
	if err != nil {
		return err
	}
	p, err := BuildPlan(model, host, o)
	if err != nil {
		return err
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
func printPlan(p Plan) {
	fmt.Printf("Model:       %s\nRevision:    %s\nTask:        %s\nRuntime:     %s (%s)\nGPU:         %d · %s · %d MiB free\nWeights:     %d files · %.2f GiB (plus supporting files)\nEst. VRAM:   %d MiB\nEndpoint:    http://127.0.0.1:%d\nContainer:   %s\n",
		p.Model, p.Revision, p.Task, p.Runtime, p.Image, p.GPU.Index, p.GPU.Name, p.GPU.FreeMiB, len(p.Files), float64(p.WeightBytes)/(1024*1024*1024), p.EstimatedVRAMMiB, p.Port, p.Name)
	if p.Context > 0 {
		fmt.Printf("Context:     %d\n", p.Context)
	}
	fmt.Printf("Backend:     %s\n", p.Backend)
	if p.RecipeID != "" {
		fmt.Printf("Recipe:      %s (%s), %s\n", p.RecipeID, p.RecipeStatus, p.DType)
	}
	if p.Python != "" {
		fmt.Printf("Base Python: %s\n", p.Python)
	}
	fmt.Printf("GPU access:  %s\n", p.GPUMode)
	fmt.Println("Weight files:")
	for _, f := range p.Files {
		fmt.Println("  " + f)
	}
	for _, w := range p.Warnings {
		fmt.Println("Warning: " + w)
	}
	for _, b := range p.Blockers {
		fmt.Println("BLOCKED: " + b)
	}
}
