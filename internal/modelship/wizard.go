package modelship

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

var errWizardCanceled = errors.New("setup canceled; no deployment or API key was created")

// Only the UI is new: inspection, policy/fit planning and deployment stay shared
// with the existing automation commands. Tests replace side-effect boundaries.
type wizardDeps struct {
	gguf   func(context.Context, *Model, string) error
	vision func(context.Context, *Model, string) error

	inspect    func(context.Context, string, string) (Model, error)
	detect     func(context.Context, string) (Host, error)
	cache      func(*Host, Model)
	build      func(Model, Host, Options) (Plan, error)
	probe      func(context.Context, string, Plan) (TorchInfo, error)
	createKey  func(string) error
	deploy     func(context.Context, Plan, time.Duration) error
	executable string
}

func defaultWizardDeps() wizardDeps {
	exe, _ := os.Executable()
	return wizardDeps{vision: func(ctx context.Context, m *Model, file string) error {
		hub := NewHub()
		hub.Token = os.Getenv("HF_TOKEN")
		return hub.inspectGGUFProjector(ctx, m, file)
	}, gguf: func(ctx context.Context, m *Model, file string) error {
		hub := NewHub()
		hub.Token = os.Getenv("HF_TOKEN")
		return hub.inspectGGUF(ctx, m, file)
	}, inspect: inspectForPlan, detect: DetectForBackend, cache: InspectModelCache, build: BuildPlan, probe: probeTorch,
		createKey: func(path string) error { _, err := generateAPIKey(path); return err }, deploy: Deploy, executable: exe}
}
func launchCommand(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("launch", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "modelship launch — interactive model setup; use deploy MODEL --yes for automation")
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("launch accepts no arguments; use deploy MODEL [options] for automation")
	}
	if !terminalFile(os.Stdin) || !terminalFile(os.Stdout) {
		return fmt.Errorf("launch requires terminal input and output; download install.sh and run it from a terminal, or use deploy MODEL --yes")
	}
	return runWizard(ctx, os.Stdin, os.Stdout, defaultWizardDeps())
}

type wizardUI struct {
	labels  map[string]string
	session *wizardSession

	ctx context.Context
	in  *bufio.Reader
	out io.Writer
}
type wizardLine struct {
	text string
	err  error
}

// Explain a default's provenance without decorating the actual input value or
// adding another question. A bracketed value remains exactly what Enter accepts.
func (w *wizardUI) explain(source, detail string) {
	fmt.Fprintf(w.out, "  %s — %s\n", source, detail)
}

func (w *wizardUI) ask(label, fallback string, valid func(string) error) (string, error) {
	if w.session != nil {
		s := w.session
		if s.cursor < len(s.answers) {
			value := s.answers[s.cursor]
			if valid == nil || valid(value) == nil {
				s.cursor++
				return value, nil
			}
			s.answers = s.answers[:s.cursor]
		}
		if s.redoSet {
			fallback = s.redo
			s.redoSet = false
		}
	}
	for {
		if err := w.ctx.Err(); err != nil {
			return "", err
		}
		question := fmt.Sprintf("%s: ", label)
		if fallback != "" {
			question = fmt.Sprintf("%s [%s]: ", label, fallback)
		}
		var line wizardLine
		if w.session != nil && w.session.terminal != nil {
			line.text, line.err = readWizardTerminal(w.ctx, w.session.terminal, w.out, question)
		} else {
			fmt.Fprint(w.out, question)
			// A canceled signal must exit even while terminal input is waiting. The CLI
			// process owns stdin and exits on cancellation; no deployment runs in here.
			result := make(chan wizardLine, 1)
			go func() { data, err := w.in.ReadSlice('\n'); result <- wizardLine{string(data), err} }()
			select {
			case <-w.ctx.Done():
				return "", w.ctx.Err()
			case line = <-result:
			}
		}
		if errors.Is(line.err, errWizardBack) || errors.Is(line.err, errWizardCanceled) || errors.Is(line.err, context.Canceled) || errors.Is(line.err, context.DeadlineExceeded) {
			return "", line.err
		}
		if line.err != nil {
			return "", fmt.Errorf("terminal input ended or exceeded 8 KiB; setup canceled before deployment")
		}
		value := strings.TrimSpace(line.text)
		if value == "\x1b" || value == ":back" {
			return "", errWizardBack
		}
		if value == "" {
			value = fallback
		}
		if strings.IndexFunc(value, func(r rune) bool { return unicode.IsControl(r) || unicode.Is(unicode.Cf, r) }) >= 0 {
			fmt.Fprintln(w.out, "Control characters are not allowed.")
			continue
		}
		if valid != nil {
			if err := valid(value); err != nil {
				fmt.Fprintln(w.out, err)
				continue
			}
		}
		if w.session != nil {
			w.session.answers = append(w.session.answers, value)
			w.session.cursor++
		}
		return value, nil
	}
}
func choiceValue(value string, choices []string) (string, bool) {
	if n, e := strconv.Atoi(value); e == nil && n >= 1 && n <= len(choices) {
		return choices[n-1], true
	}
	if len(value) == 1 {
		c := strings.ToLower(value)[0]
		if c >= 'a' && int(c-'a') < len(choices) {
			return choices[int(c-'a')], true
		}
	}
	for _, choice := range choices {
		if strings.EqualFold(choice, value) {
			return choice, true
		}
	}
	return "", false
}
func (w *wizardUI) choose(label string, choices []string, fallback string) (string, error) {
	return w.selectChoice(label, choices, fallback, true)
}
func (w *wizardUI) selectChoice(label string, choices []string, fallback string, auto bool) (string, error) {
	if len(choices) == 0 {
		return "", fmt.Errorf("no supported choices for %s", label)
	}
	if len(choices) == 1 && auto {
		fmt.Fprintf(w.out, "%s: %s (only supported choice; selected automatically)\n", label, choices[0])
		return choices[0], nil
	}
	if w.session != nil && w.session.redoSet && w.session.cursor == len(w.session.answers) {
		if v, ok := choiceValue(w.session.redo, choices); ok {
			fallback = v
		}
	}
	for i, choice := range choices {
		letter := ""
		if i < 26 {
			letter = fmt.Sprintf(" / %c", 'a'+i)
		}
		mark := ""
		if choice == fallback {
			mark = " (default)"
		}
		display := choice
		if w.labels[choice] != "" {
			display = w.labels[choice]
		}
		fmt.Fprintf(w.out, "  %d%s) %s%s\n", i+1, letter, display, mark)
	}
	value, e := w.ask(label+" (number, letter, or full value)", fallback, func(value string) error {
		if _, ok := choiceValue(value, choices); !ok {
			return fmt.Errorf("Choose 1–%d, a letter, or a listed value", len(choices))
		}
		return nil
	})
	if e != nil {
		return "", e
	}
	v, _ := choiceValue(value, choices)
	if w.session != nil && w.session.cursor > 0 {
		w.session.answers[w.session.cursor-1] = v
	}
	return v, nil
}
func (w *wizardUI) confirm(message string) (bool, error) {
	value, err := w.ask(message+" (y/N)", "n", func(value string) error {
		if !contains([]string{"y", "yes", "n", "no"}, strings.ToLower(value)) {
			return fmt.Errorf("Enter y or n")
		}
		return nil
	})
	return strings.EqualFold(value, "y") || strings.EqualFold(value, "yes"), err
}
func (w *wizardUI) consent(message string) error {
	ok, err := w.confirm(message)
	if err != nil {
		return err
	}
	if !ok {
		return errWizardCanceled
	}
	return nil
}
func (w *wizardUI) integer(label string, fallback, min, max int) (int, error) {
	value, err := w.ask(label, strconv.Itoa(fallback), func(value string) error {
		v, e := strconv.Atoi(value)
		if e != nil || v < min || v > max {
			return fmt.Errorf("Use an integer from %d to %d", min, max)
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	v, _ := strconv.Atoi(value)
	return v, nil
}
func wizardPath(value string) (string, error) {
	if value == "~" || strings.HasPrefix(value, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		value = filepath.Join(home, strings.TrimPrefix(value, "~/"))
		if strings.HasSuffix(value, "/~") {
			value = home
		}
	}
	return filepath.Abs(value)
}
func newKeyPath(path string) error {
	if _, err := os.Lstat(path); !os.IsNotExist(err) {
		return fmt.Errorf("New key path must not exist; choose another path or reuse an existing private key")
	}
	parent, err := os.Lstat(filepath.Dir(path))
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("Cannot inspect key parent directory")
	}
	if err == nil && (!parent.IsDir() || parent.Mode().Perm()&0077 != 0) {
		return fmt.Errorf("Key parent directory must be real and private (0700)")
	}
	return nil
}
func defaultWizardKey(name string) (string, error) {
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		base = filepath.Join(home, ".config")
	}
	if !filepath.IsAbs(base) {
		return "", fmt.Errorf("XDG_CONFIG_HOME must be absolute")
	}
	return filepath.Join(base, "modelship", name+".key"), nil
}
func (w *wizardUI) access(o *Options) (bool, error) {
	fmt.Fprintln(w.out, "local = GPU-server loopback or SSH tunnel, NOT your laptop/browser localhost. https-ingress = network listener behind a provider HTTPS URL, such as Thunder forwarding.")
	w.explain("Safety default", "local keeps the worker private (127.0.0.1). https-ingress requires bearer auth; it does not configure provider forwarding or install TLS.")
	choice, err := w.choose("API access", []string{"local", "https-ingress"}, "local")
	if err != nil {
		return false, err
	}
	o.Listen = "127.0.0.1"
	choices, defaultKey := []string{"none", "new", "existing"}, "none"
	if choice == "https-ingress" {
		o.Listen = "0.0.0.0"
		choices = []string{"new", "existing"}
		defaultKey = "new"
		fmt.Fprintln(w.out, "Network access requires bearer auth and trusted HTTPS ingress. Raw HTTP is not encrypted; no provider forwarding is changed automatically.")
	}
	if choice == "local" {
		w.explain("Safety default", "none means no bearer key on this private loopback listener; choose new or existing if you also want local API authentication.")
	} else {
		w.explain("Required authentication", "new creates a random private key only after approval; existing reuses a private key FILE. Neither option prints its value.")
	}
	keyChoice, err := w.choose("API key", choices, defaultKey)
	if err != nil || keyChoice == "none" {
		return false, err
	}
	fallback, err := defaultWizardKey(o.Name)
	if err != nil {
		return false, err
	}
	w.explain("Computed path", "XDG_CONFIG_HOME (or ~/.config) + modelship/deployment-name.key; this is not a detected existing key. New files are created only after approval and are never overwritten; existing must already be a private 0600 file.")
	path, err := w.ask("Private key FILE path on this server (never paste a key value)", fallback, func(value string) error {
		if apiKeyPattern.MatchString(value) {
			return fmt.Errorf("That looks like a key value; enter the private FILE path instead")
		}
		path, e := wizardPath(value)
		if e != nil {
			return fmt.Errorf("Invalid key file path")
		}
		if keyChoice == "new" {
			return newKeyPath(path)
		}
		_, e = readAPIKey(path)
		return e
	})
	if err != nil {
		return false, err
	}
	o.APIKeyFile, err = wizardPath(path)
	return keyChoice == "new", err
}

func runWizardAttempt(ctx context.Context, input io.Reader, output io.Writer, d wizardDeps, session *wizardSession) error {
	w := wizardUI{session: session, ctx: ctx, in: session.reader, out: output}
	fmt.Fprintln(output, "Modelship — guided model setup\nInspect → choose → review → deploy. No model tensor payloads, installs or keys before final confirmation.\nUse HF_TOKEN in your environment for gated/private repositories; never paste credentials here.")
	fmt.Fprintln(output, "Press Enter to accept the value in brackets. Menus accept numbers, letters or full values; Esc goes back (or type :back). Ctrl-C cancels. Type a value to change it; y/N questions default to No. A single supported choice is selected automatically.\nDefaults below are labeled as detected, recipe-controlled, recommended presets, or safety choices—not promises of hardware capacity.")
	requested, err := w.ask("Hugging Face model page or org/repo", "", func(value string) error {
		if strings.HasPrefix(value, "hf_") && len(value) > 25 {
			return fmt.Errorf("Enter a model page, not a credential; use HF_TOKEN in your environment")
		}
		_, e := ParseModel(value)
		return e
	})
	if err != nil {
		return err
	}
	requested, _ = ParseModel(requested)
	selected, weightsRepo := requested, ""
	if requested == "Wan-AI/Wan2.2-T2V-A14B" {
		fmt.Fprintln(output, "The original WAN PTH layout is unsupported. The explicitly supported alternative is Wan-AI/Wan2.2-T2V-A14B-Diffusers.")
		if err = w.consent("Select that official Diffusers weights repository instead"); err != nil {
			return err
		}
		weightsRepo = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
		selected, err = resolveWeightRepo(requested, weightsRepo)
		if err != nil {
			return err
		}
	}
	defaultRevision := "main"
	if selected == ggufRepo {
		defaultRevision = ggufRevision
	} else if selected == gemmaVisionRepo {
		defaultRevision = gemmaVisionRevision
	}
	catalog, err := LoadRecipes()
	if err != nil {
		return err
	}
	for _, r := range catalog.Recipes {
		if contains(r.Repositories, selected) && r.CodeRevision != "" {
			defaultRevision = r.CodeRevision
		}
	}
	if defaultRevision == "main" {
		w.explain("Upstream branch default", "main means the repository's current upstream branch, not a fixed or locally tested version. It is resolved once now; review/deployment use the exact pinned commit SHA.")
	} else if selected == gemmaVisionRepo {
		w.explain("Reviewed image-profile pin", "This optional Gemma image adapter was reviewed at this exact commit. Text-only is still the default; other revisions/variants are not substituted or assumed vision-compatible.")
	} else if selected == ggufRepo {
		w.explain("GGUF artifact pin", "This first GGUF adapter uses a reviewed immutable artifact manifest; another revision or file is not silently substituted.")
	} else {
		w.explain("Reviewed recipe pin", "This custom-code recipe defaults to its reviewed immutable commit, not the latest upstream branch. Source-code approval is still separate.")
	}
	revision, err := w.ask("Model revision (branch, tag or commit)", defaultRevision, func(v string) error {
		if v == "" || len(v) > 200 {
			return fmt.Errorf("Enter a revision name or commit")
		}
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Fprintln(output, "Inspecting repository/configuration and bounded tensor headers; no model tensor payloads are downloaded...")
	model, err := d.inspect(ctx, selected, revision)
	if err != nil {
		return err
	}
	recipe, known, err := findRecipe(model)
	if err != nil {
		return err
	}
	if !known {
		return fmt.Errorf("no compatible reviewed runtime adapter; use modelship inspect for metadata")
	}
	if err = validateRecipeShape(recipe, model); err != nil {
		return err
	}
	if recipe.Status == "planned" {
		return fmt.Errorf("recipe %s is recognized but not deployable: %s", recipe.ID, strings.Join(recipe.Blockers, "; "))
	}
	ggufFile, mmprojFile := "", ""
	vision := false
	if recipe.Format == "gguf" {
		options, e := ggufChoices(model)
		if e != nil {
			return e
		}
		choices := []string{}
		w.labels = map[string]string{}
		for _, pin := range options {
			choices = append(choices, pin.Name)
			w.labels[pin.Name] = fmt.Sprintf("%s — %.2f GiB download", strings.TrimSuffix(strings.TrimPrefix(pin.Name, "Qwen3.8-27B-"), ".gguf"), float64(pin.Size)/(1<<30))
		}
		fallback := ggufFilename
		if !contains(choices, fallback) && len(choices) > 0 {
			fallback = ""
		}
		w.explain("Recommended quantization preset", "UD-Q4_K_M is a balanced, user-reported working preset—not automatic hardware fitting. These are reviewed immutable files; alternatives are experimental, not independently GPU-tested. Sizes are download bytes, not peak VRAM. Split BF16, MTP, imatrix and unreviewed files are excluded.")
		ggufFile, err = w.selectChoice("GGUF quantization", choices, fallback, false)
		w.labels = nil
		if err != nil {
			return err
		}
		if d.gguf == nil {
			return fmt.Errorf("GGUF header inspector unavailable")
		}
		if err = d.gguf(ctx, &model, ggufFile); err != nil {
			return err
		}
		hasProjector := false
		for _, f := range model.Hub.Siblings {
			if f.Name == ggufProjectorFile {
				hasProjector = true
			}
		}
		model.GGUF.Vision = nil
		if hasProjector {
			w.explain("Input capability", "Text-only is the safety default. Vision loads the separately pinned F16 projector, adds memory/disk budgets and permits one uploaded image per conversation. It is image understanding, not image generation. Real-GPU image understanding is user-confirmed; coverage of all quants/hardware and peak memory is not established.")
			mode, e := w.choose("Input mode", []string{"text-only", "vision-images"}, "text-only")
			if e != nil {
				return e
			}
			vision = mode == "vision-images"
			if vision {
				mmprojFile = ggufProjectorFile
				if d.vision == nil {
					return fmt.Errorf("vision projector inspector unavailable")
				}
				if err = d.vision(ctx, &model, mmprojFile); err != nil {
					return err
				}
				fmt.Fprintf(output, "Paired projector: %s @ %s; %.2f GiB additional encoded tensors\n", mmprojFile, model.Hub.SHA, float64(model.GGUF.Vision.Metadata.TensorBytes)/(1<<30))
			}
		}
	}
	fmt.Fprintf(output, "Resolved: %s @ %s\nRecipe: %s (%s), %s\n", model.Hub.ID, model.Hub.SHA, recipe.ID, recipe.Status, recipe.DType)
	// Remote card text is untrusted terminal data; quote rather than emit escapes.
	license, _ := json.Marshal(model.Hub.CardData.License)
	if name, ok := model.Hub.CardData.License.(string); ok {
		license = []byte(name)
	}
	if len(license) > 512 {
		license = append(license[:512], []byte("...")...)
	}
	fmt.Fprintf(output, "Declared license: %q — review repository terms and access grants.\n", string(license))
	backendDefault := recipe.Backends[0]
	if contains(recipe.Backends, "native") {
		backendDefault = "native"
	}
	if recipe.Format == "gguf" {
		w.explain("Pinned format", "The selected GGUF contains mixed encoded tensor types; this is not a global compute dtype or automatic quantization choice. Full GPU layer placement is explicit; CPU offload is not a fallback.")
		w.explain("Standalone native backend", "This adapter uses pinned llama.cpp CUDA12.8, not Python/PyTorch. Runtime checksum/device verification happens after approval and before model download; no host packages or drivers are changed.")
	} else {
		w.explain("Recipe-controlled setting", "Default compute dtype is "+recipe.DType+" for this adapter (component overrides may apply); this is not an auto-fit selection. No automatic quantization or CPU offload is enabled.")
		fmt.Fprintln(output, "Native uses existing working CUDA PyTorch without replacing it. Docker requires your provider's supported local GPU runtime.")
		if backendDefault == "native" {
			w.explain("Recommended backend", "native is the preset because this recipe supports it—not because Python/PyTorch has already been verified. The CUDA preflight checks that later.")
		} else {
			w.explain("Recipe default", "Backend choices come from this adapter's supported paths, not automatic proof that the host runtime is configured.")
		}
	}
	backend, err := w.choose("Backend", recipe.Backends, backendDefault)
	if err != nil {
		return err
	}
	if recipe.Format != "gguf" && backend == "native" {
		if _, e := nativeVisionFor(model); e == nil {
			w.explain("Input capability", "Text-only stays the default. The reviewed Gemma image mode uses the same full BF16 checkpoint, a pinned PIL image processor and an extra activation heuristic. One uploaded PNG/JPEG + text; not image generation/audio/video. CPU tests are not actual GPU image-inference validation.")
			mode, e := w.choose("Input mode", []string{"text-only", "vision-images"}, "text-only")
			if e != nil {
				return e
			}
			vision = mode == "vision-images"
		}
	}
	fmt.Fprintln(output, "Inspecting local GPUs and available memory...")
	hctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	host, err := d.detect(hctx, backend)
	cancel()
	if err != nil {
		return err
	}
	if len(host.GPUs) == 0 {
		return fmt.Errorf("no NVIDIA GPU available")
	}
	gpuChoices := make([]string, 0, len(host.GPUs))
	for _, g := range host.GPUs {
		fmt.Fprintf(output, "GPU %d: %s — %d/%d MiB free\n", g.Index, strconv.QuoteToASCII(g.Name), g.FreeMiB, g.TotalMiB)
		gpuChoices = append(gpuChoices, "GPU "+strconv.Itoa(g.Index))
	}
	w.explain("Detected default", "The first GPU returned by discovery, not the largest or best-fit GPU. MiB are binary memory units; free VRAM is a snapshot and will be rechecked before launch.")
	gpu, err := w.choose("GPU index", gpuChoices, gpuChoices[0])
	if err != nil {
		return err
	}
	index, _ := strconv.Atoi(strings.TrimPrefix(gpu, "GPU "))
	o := Options{Vision: vision, MMProjFile: mmprojFile, GGUFFile: ggufFile, Backend: backend, Revision: model.Hub.SHA, GPU: index, GPUMode: "auto", Python: "python3", Context: 4096}
	taskDefault := recipe.Tasks[0]
	if contains(recipe.Tasks, model.Hub.PipelineTag) {
		taskDefault = model.Hub.PipelineTag
		w.explain("Detected task suggestion", "Hugging Face metadata suggests this task, and the adapter permits it; only supported choices are shown.")
	} else {
		w.explain("Recipe task default", "The adapter's first supported task, not a guessed capability. A deployment serves the selected task only.")
	}
	o.Task, err = w.choose("Task", recipe.Tasks, taskDefault)
	if err != nil {
		return err
	}
	w.explain("Naming preset", "modelship is a generic local label for status/logs/start/stop, not the HF repository name. Choose an unused label; an existing deployment is not replaced.")
	o.Name, err = w.ask("Deployment name", "modelship", func(value string) error {
		if len(value) > 63 || !containerName.MatchString(value) {
			return fmt.Errorf("Use 1–63 letters/numbers with hyphens or underscores")
		}
		return nil
	})
	if err != nil {
		return err
	}
	defaultPort := 8000
	if recipe.Kind == "video" {
		defaultPort = 18032
	} else if recipe.Kind == "image" {
		defaultPort = 8001
	}
	w.explain("Recommended port preset", "Text uses 8000, images 8001, video 18032. This is the inference HTTP port on the GPU server—not SSH or a Thunder SSH port. Range: 1024–65535; forward this same port if using HTTPS ingress.")
	o.Port, err = w.integer("Inference API port (server-side)", defaultPort, 1024, 65535)
	if err != nil {
		return err
	}
	if o.Task == "text-generation" {
		w.explain("Conservative preset", wizardContextHint(model))
		contextCap := 131072
		if recipe.Format == "gguf" {
			contextCap = 32768
		}
		o.Context, err = w.integer("Context window (tokens)", 4096, 128, contextCap)
		if err != nil {
			return err
		}
	}
	createKey := false
	if backend == "native" {
		if recipe.Format != "gguf" {
			w.explain("PATH preset", "python3 is a command name, not a detected CUDA environment. Choose the interpreter that already has working CUDA PyTorch; Modelship checks it and never installs/replaces host PyTorch or drivers.")
			o.Python, err = w.ask("Existing CUDA Python executable", "python3", func(v string) error {
				if v == "" {
					return fmt.Errorf("Enter a Python executable or path")
				}
				return nil
			})
			if err != nil {
				return err
			}
		} else {
			o.Python = ""
			w.explain("Standalone runtime", "This GGUF adapter uses a separately verified llama.cpp CUDA build. Python/PyTorch are not required, installed or changed; CUDA runtime verification happens after approval, before model download.")
		}
		createKey, err = w.access(&o)
		if err != nil {
			return err
		}
		d.cache(&host, model)
	} else {
		w.explain("Compatibility preset", "auto tries NVIDIA GPU access; CDI fallback is limited to a recognized Thunder rejection on a single-GPU host. This does not install/configure a Docker GPU runtime.")
		o.GPUMode, err = w.choose("Docker GPU access mode", []string{"auto", "nvidia", "cdi"}, "auto")
		if err != nil {
			return err
		}
	}
	if recipe.Status == "experimental" {
		if err = w.consent("This recipe is experimental; explicitly opt in"); err != nil {
			return err
		}
		o.Experimental = true
	}
	if recipe.RequiresResearch {
		if err = w.consent("I acknowledge non-commercial research/evaluation-only use under this model's license"); err != nil {
			return err
		}
		o.Research = true
	}
	if recipe.RequiresCustomCode {
		fmt.Fprintf(output, "Repository Python will run as your user, NOT in a sandbox. Reviewed commit: %s\n", recipe.CodeRevision)
		names := make([]string, 0, len(recipe.CodeHashes))
		for name := range recipe.CodeHashes {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			fmt.Fprintf(output, "Review https://huggingface.co/%s/blob/%s/%s\nSHA-256 %s\n", selected, recipe.CodeRevision, name, recipe.CodeHashes[name])
		}
		if err = w.consent("I have reviewed these pinned source files and explicitly authorize their execution"); err != nil {
			return err
		}
		o.TrustModelCode = true
	}
	waitDefault := "30m"
	if recipe.Kind == "video" {
		waitDefault = "90m"
	}
	w.explain("Startup wait preset", "30m normally, 90m for video: how long to wait for the started worker to download/load and become ready—not a generation timeout, speed estimate, or pip/image-build deadline. 0 returns without waiting; timeout/interruption can leave the worker running.")
	waitText, err := w.ask("Worker readiness wait (duration; 0 skips waiting)", waitDefault, func(v string) error {
		t, e := time.ParseDuration(v)
		if e != nil || t < 0 {
			return fmt.Errorf("Use a nonnegative duration, e.g. 30m or 90m")
		}
		return nil
	})
	if err != nil {
		return err
	}
	wait, _ := time.ParseDuration(waitText)
	build := func(options Options) (Plan, error) {
		// A NEW key is a reviewed future action, not an already existing secret. Keep
		// all shared model/VRAM/disk checks, then attach and structurally validate the
		// planned access boundary. Deploy/start still require the real private file.
		planned := options
		if createKey {
			planned.Listen = "127.0.0.1"
			planned.APIKeyFile = ""
		}
		p, e := d.build(model, host, planned)
		if e != nil {
			return p, e
		}
		if createKey {
			p.Listen = options.Listen
			p.APIKeyFile = options.APIKeyFile
			if e = validateNativeAccess(p, false); e != nil {
				return p, e
			}
			p.Warnings = append(p.Warnings, "A private API key will be generated only after final approval; use HTTPS ingress for a network listener")
		}
		if requested != selected {
			p.RequestedModel = requested
		}
		return p, nil
	}
	plan, err := build(o)
	if err != nil {
		return err
	}
	if len(plan.Blockers) > 0 {
		riskOptions := o
		riskOptions.AllowRisk = true
		riskPlan, e := build(riskOptions)
		if e != nil {
			return e
		}
		if len(riskPlan.Blockers) == 0 {
			writePlan(output, plan)
			if err = w.consent("Override ONLY the conservative runtime-memory heuristic (not physical weights, disk or policy)"); err != nil {
				return err
			}
			o = riskOptions
			plan = riskPlan
		}
	}
	if len(plan.Blockers) > 0 {
		writePlan(output, plan)
		return fmt.Errorf("deployment blocked; no key/install/deployment was created — resolve the hard blockers first")
	}
	if backend == "native" && recipe.Format != "gguf" {
		fmt.Fprintln(output, "Checking existing Python/PyTorch with a small CUDA preflight (no packages installed)...")
		info, e := d.probe(ctx, o.Python, plan)
		if e != nil {
			return e
		}
		if e = validateNativeTorch(plan, info); e != nil {
			return e
		}
		fmt.Fprintf(output, "Existing PyTorch: %s · CUDA %s · Python %s\n", strconv.QuoteToASCII(info.Version), strconv.QuoteToASCII(info.CUDA), strconv.QuoteToASCII(info.PythonVersion))
	}
	fmt.Fprintln(output, "\nREVIEW — exact pinned deployment plan")
	writePlan(output, plan)
	fmt.Fprintf(output, "Revision selection: %s -> pinned %s (no floating branch at deployment).\n", strconv.QuoteToASCII(revision), plan.Revision)
	if wait == 0 {
		fmt.Fprintln(output, "Readiness wait: skipped (0); the worker may still be downloading/loading after the CLI returns.")
	} else {
		fmt.Fprintf(output, "Readiness wait: %s for worker startup only; this does not limit individual generations.\n", waitText)
	}
	if backend == "native" {
		if nativeListen(plan) == "127.0.0.1" {
			fmt.Fprintln(output, "Access: GPU-server loopback/SSH only. The displayed localhost endpoint is not your laptop/browser.")
		} else {
			fmt.Fprintln(output, "Access: authenticated network listener; use trusted HTTPS ingress. No provider URL/forwarding is created automatically.")
		}
	}
	if createKey {
		fmt.Fprintf(output, "Planned action: create private key file %s (key value never printed).\n", strconv.QuoteToASCII(o.APIKeyFile))
	}
	fmt.Fprintln(output, "No other deployments will be stopped or removed. Host drivers/PyTorch are not replaced.")
	fmt.Fprintln(output, "Equivalent command (key file must exist; no secret values):")
	fmt.Fprintln(output, wizardCommand(d.executable, requested, weightsRepo, o, plan, wait))
	if err = w.consent("Create the planned key if needed, install dependencies/download missing weights, and deploy this exact plan"); err != nil {
		return err
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if session != nil {
		session.committed = true
	}
	if createKey {
		if err = d.createKey(o.APIKeyFile); err != nil {
			return err
		}
		fmt.Fprintln(output, "Private API key created; its value is not displayed.")
	}
	if err = d.deploy(ctx, plan, wait); err != nil {
		return err
	}
	fmt.Fprintln(output, "Deployment submitted. Leaving this session does not cancel a retained worker.")
	fmt.Fprintf(output, "Status: %s status --backend %s --name %s\n", shellWord(d.executable), shellWord(o.Backend), shellWord(plan.Name))
	if plan.APIKeyFile != "" {
		fmt.Fprintf(output, "Console: use the upstream bearer-key value from private file %s; keep it secret.\n", strconv.QuoteToASCII(plan.APIKeyFile))
	}
	if nativeListen(plan) == "0.0.0.0" {
		fmt.Fprintln(output, "Configure trusted HTTPS/provider forwarding for the selected port; no forwarding was changed automatically.")
	}
	return nil
}

// Match the planner's top-level/nested context source. Advertised model context
// is metadata, not a measured VRAM-fit or quality guarantee; defaults stay put.
func wizardContextHint(model Model) string {
	if model.GGUF != nil {
		return fmt.Sprintf("4096-token preset, not the model maximum. This first GGUF adapter accepts 128–32768; header advertises %d. Quantized weights do not eliminate KV/recurrent-state or activation memory. No automatic context shrinking or CPU fallback.", model.GGUF.Metadata.Context)
	}
	config := model.Config
	if config.TextConfig != nil {
		config = *config.TextConfig
	}
	hint := "4096 tokens is a conservative preset, not necessarily the model maximum. Prompt + generated tokens share this window; it is not the maximum output-token setting. CLI range: 128–131072; memory-fit checks still apply."
	if config.MaxPosition > 0 {
		hint += fmt.Sprintf(" Model config advertises %d tokens (not a validated capacity).", config.MaxPosition)
	} else {
		hint += " A context maximum was not available in the inspected text configuration."
	}
	return hint
}

var shellSafe = regexp.MustCompile(`^[A-Za-z0-9_./:@=+,-]+$`)

func shellWord(value string) string {
	if shellSafe.MatchString(value) {
		return value
	}
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}
func wizardCommand(exe, requested, weights string, o Options, p Plan, wait time.Duration) string {
	if exe == "" {
		exe = "modelship"
	}
	args := []string{exe, "deploy", requested, "--backend", o.Backend, "--task", p.Task, "--revision", p.Revision, "--name", p.Name, "--gpu", strconv.Itoa(p.GPU.Index), "--port", strconv.Itoa(p.Port), "--wait", wait.String()}
	if weights != "" {
		args = append(args, "--weights-repo", weights)
	}
	if o.GGUFFile != "" {
		args = append(args, "--gguf-file", o.GGUFFile)
	}
	if o.Vision {
		args = append(args, "--vision")
		if o.MMProjFile != "" {
			args = append(args, "--mmproj-file", o.MMProjFile)
		}
	}
	if o.Backend == "native" {
		if p.GGUF == nil {
			args = append(args, "--python", o.Python)
		}
		args = append(args, "--listen", nativeListen(p))
		if p.APIKeyFile != "" {
			args = append(args, "--api-key-file", p.APIKeyFile)
		}
	} else {
		args = append(args, "--gpu-mode", o.GPUMode)
	}
	if p.Task == "text-generation" {
		args = append(args, "--context", strconv.Itoa(p.Context))
	}
	for _, option := range []struct {
		enabled bool
		flag    string
	}{{o.Experimental, "--experimental"}, {o.Research, "--research"}, {o.TrustModelCode, "--trust-model-code"}, {o.AllowRisk, "--allow-risk"}} {
		if option.enabled {
			args = append(args, option.flag)
		}
	}
	for i, value := range args {
		args[i] = shellWord(value)
	}
	return strings.Join(args, " ")
}
