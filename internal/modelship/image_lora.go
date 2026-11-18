package modelship

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

const loraSchemaVersion = 11
const loraAllowanceMiB = 1024 // Unmeasured temporary workspace, separate from resident weights.
const loraBundle = "qwen-image-lora-v1"

var loraAssets = []string{"api_auth.py", "native_entry.py", "server.py", "dual_image_profiles.json", "qwen_image_decode.py", "qwen_lora.py", "qwen_lora_profile.json"}

type imageLoRAProfile struct {
	Repo         string `json:"repo"`
	Revision     string `json:"revision"`
	File         string `json:"file"`
	Size         int64  `json:"size"`
	SHA256       string `json:"sha256"`
	BaseModel    string `json:"base_model"`
	BaseRevision string `json:"base_revision"`
	Trigger      string `json:"trigger"`
}

type NativeLoRA struct {
	Revision     string `json:"revision"`
	BundleSHA256 string `json:"bundle_sha256"`
	BaseSchema   int    `json:"base_schema"`
}

func loadLoRAProfile() imageLoRAProfile {
	var p imageLoRAProfile
	data, err := assets.ReadFile("assets/qwen_lora_profile.json")
	if err != nil || json.Unmarshal(data, &p) != nil {
		panic("invalid embedded LoRA profile")
	}
	return p
}

func validateLoRABase(p Plan) error {
	a := loadLoRAProfile()
	// Reuse the identical pinned recipe contract for single-task deployments.
	dual := p
	dual.Task = dualImageTask
	if (p.Task != dualImageTask && p.Task != "text-to-image" && p.Task != "image-to-image") || p.Model != a.BaseModel || p.Revision != a.BaseRevision || p.RecipeID != "qwen-image21" || validateDualImageExecution(dual) != nil || p.WeightResidentBytes > math.MaxInt64-a.Size || p.EstimatedVRAMMiB <= 0 || p.EstimatedVRAMMiB > math.MaxInt64/2 {
		return fmt.Errorf("LoRA requires the exact reviewed native Qwen Image 2.1 BF16 research deployment with saved GPU and weight budgets")
	}
	return nil
}

func validateLoRAConfig(cfg NativeConfig) error {
	if cfg.LoRA == nil {
		if cfg.Schema == loraSchemaVersion {
			return fmt.Errorf("native schema 11 requires reviewed LoRA metadata")
		}
		return nil
	}
	a := cfg.LoRA
	digest, err := hex.DecodeString(a.BundleSHA256)
	if cfg.Schema != loraSchemaVersion || a.BaseSchema < nativeSchemaVersion || a.BaseSchema > dualImageSchemaVersion || cfg.Plan.Task == dualImageTask && a.BaseSchema != dualImageSchemaVersion || a.Revision != loadLoRAProfile().Revision || err != nil || len(digest) != 32 || hex.EncodeToString(digest) != a.BundleSHA256 {
		return fmt.Errorf("invalid reviewed native LoRA metadata")
	}
	return validateLoRABase(cfg.Plan)
}

func loraWeightPlan(cfg NativeConfig) Plan {
	p := cfg.Plan
	if cfg.LoRA != nil {
		p.WeightResidentBytes += loadLoRAProfile().Size
	}
	return p
}

func loraRuntime(dir string, a *NativeLoRA) string {
	return filepath.Join(dir, loraBundle, a.BundleSHA256)
}

func loraBundleDigest(dir string) (string, error) {
	h := sha256.New()
	for _, name := range loraAssets {
		var data []byte
		var err error
		if dir == "" {
			data, err = assets.ReadFile("assets/" + name)
		} else {
			path := filepath.Join(dir, name)
			if err = privateRegular(path); err != nil {
				return "", err
			}
			f, e := os.Open(path)
			if e != nil {
				return "", e
			}
			data, err = io.ReadAll(io.LimitReader(f, 256*1024+1))
			f.Close()
		}
		if err != nil || len(data) > 256*1024 {
			return "", fmt.Errorf("invalid LoRA runtime bundle asset")
		}
		fmt.Fprintf(h, "%s:%x\n", name, sha256.Sum256(data))
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func verifyLoRARuntime(dir string, a *NativeLoRA) error {
	return verifyLoRABundle(loraRuntime(dir, a), a.BundleSHA256, loadLoRAProfile())
}

func verifyLoRABundle(root, expectedDigest string, profile imageLoRAProfile) error {
	for _, path := range []string{filepath.Dir(root), root} {
		st, err := os.Lstat(path)
		if err != nil || !st.IsDir() || st.Mode().Perm()&0077 != 0 {
			return fmt.Errorf("LoRA runtime bundle must be a private directory")
		}
	}
	digest, err := loraBundleDigest(root)
	if err != nil || digest != expectedDigest {
		return fmt.Errorf("LoRA runtime bundle digest mismatch; remove adapter to restore the base runtime")
	}
	file := filepath.Join(root, profile.File)
	if err := privateRegular(file); err != nil {
		return err
	}
	st, err := os.Stat(file)
	if err != nil || st.Size() != profile.Size {
		return fmt.Errorf("LoRA artifact size differs from the reviewed pin")
	}
	digest, err = digestFile(file)
	if err != nil || digest != profile.SHA256 {
		return fmt.Errorf("LoRA artifact digest differs from the reviewed pin")
	}
	return nil
}

func stageLoRA(ctx context.Context, dir string, a *NativeLoRA) error {
	profile := loadLoRAProfile()
	client := artifactHTTPClient()
	defer client.CloseIdleConnections()
	return stageLoRABundle(ctx, loraRuntime(dir, a), a.BundleSHA256, profile, func(ctx context.Context, path string) error {
		url := "https://huggingface.co/" + profile.Repo + "/resolve/" + profile.Revision + "/" + profile.File
		// This reviewed file is public: never forward HF_TOKEN or any worker key.
		return downloadVerified(ctx, client, url, "", path, ArtifactPin{Name: profile.File, Size: profile.Size, SHA256: profile.SHA256})
	})
}

// The native lifecycle lock serializes staging. A private sibling keeps the
// verified download/resumable partial across failures; native start only sees
// the final directory, published atomically after all bytes are verified.
func stageLoRABundle(ctx context.Context, root, digest string, profile imageLoRAProfile, download func(context.Context, string) error) error {
	if err := privateDir(filepath.Dir(root)); err != nil {
		return err
	}
	if _, err := os.Lstat(root); !os.IsNotExist(err) {
		if err != nil {
			return err
		}
		return verifyLoRABundle(root, digest, profile)
	}
	staging := root + ".partial"
	if err := privateDir(staging); err != nil {
		return err
	}
	path := filepath.Join(staging, profile.File)
	if _, err := os.Lstat(path); os.IsNotExist(err) {
		var stat syscall.Statfs_t
		// Reserve full file size even when a resumable partial already exists.
		if syscall.Statfs(staging, &stat) != nil || stat.Bsize <= 0 || stat.Bavail < uint64((profile.Size+(64<<20)+int64(stat.Bsize)-1)/int64(stat.Bsize)) {
			return fmt.Errorf("insufficient or unknown disk space for the LoRA artifact plus 64 MiB headroom")
		}
	}
	if err := download(ctx, path); err != nil {
		return err
	}
	for _, name := range loraAssets {
		data, err := assets.ReadFile("assets/" + name)
		if err != nil {
			return err
		}
		dest := filepath.Join(staging, name)
		// An interrupted write may be truncated. Unlink before recreating to
		// avoid following a link or altering another inode through a hardlink.
		if err := os.Remove(dest); err != nil && !os.IsNotExist(err) {
			return err
		}
		f, err := os.OpenFile(dest, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			return err
		}
		_, writeErr := f.Write(data)
		closeErr := f.Close()
		if writeErr != nil || closeErr != nil {
			return fmt.Errorf("cannot stage reviewed LoRA runtime assets")
		}
	}
	if err := verifyLoRABundle(staging, digest, profile); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return os.Rename(staging, root)
}

func loraCommand(ctx context.Context, args []string) error {
	if len(args) == 0 || (args[0] != "add" && args[0] != "remove") {
		return fmt.Errorf("use lora add [--name NAME] [--research --yes] HF_URL or lora remove [--name NAME] --yes")
	}
	action := args[0]
	fs := flag.NewFlagSet("lora "+action, flag.ContinueOnError)
	name := fs.String("name", "modelship", "existing native deployment")
	yes := fs.Bool("yes", false, "confirm modifying the stopped deployment")
	research := false
	allowRisk := false
	if action == "add" {
		fs.BoolVar(&research, "research", false, "acknowledge adapter's non-commercial research license")
		fs.BoolVar(&allowRisk, "allow-risk", false, "acknowledge the unmeasured runtime-memory heuristic only")
	}
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	profile := loadLoRAProfile()
	if action == "add" {
		if fs.NArg() != 1 {
			return fmt.Errorf("lora add requires one reviewed Hugging Face repository URL or ID after flags")
		}
		repo, err := ParseModel(fs.Arg(0))
		if err != nil || !strings.EqualFold(repo, profile.Repo) {
			return fmt.Errorf("only the reviewed SeedraAI Qwen Image 2.1 Polaroid adapter is supported")
		}
	} else if fs.NArg() != 0 || !*yes {
		return fmt.Errorf("lora remove requires --yes and no model argument")
	}
	dir, err := nativeDir(*name)
	if err != nil {
		return err
	}
	if _, _, err = nativeConfig(*name); err != nil {
		return err
	}
	unlock, err := nativeLock(dir)
	if err != nil {
		return err
	}
	defer unlock()
	cfg, _, err := nativeConfig(*name)
	if err != nil {
		return err
	}
	if action == "add" {
		if err = validateLoRABase(cfg.Plan); err != nil {
			return err
		}
		if cfg.LoRA != nil || cfg.Schema < nativeSchemaVersion {
			return fmt.Errorf("remove the existing adapter first, or use a compatible native schema 9/10 deployment")
		}
		fmt.Printf("Adapter: %s@%s\nArtifact: %s (%d bytes), SHA256 %s\nBase: %s@%s (%s)\nLicense: Qwen non-commercial research; trigger: %s\nDefault strength: 0 (base only); range: 0–1\nAdded resident budget: %d bytes; temporary allowance: %d MiB (unmeasured)\n", profile.Repo, profile.Revision, profile.File, profile.Size, profile.SHA256, profile.BaseModel, profile.BaseRevision, cfg.Plan.Task, profile.Trigger, profile.Size, loraAllowanceMiB)
		if !*yes {
			fmt.Println("Preview only. Stop the deployment explicitly, rerun with --research --yes, then start explicitly. No artifact downloaded.")
			return nil
		}
		if !research {
			return fmt.Errorf("lora add requires independent --research acknowledgment")
		}
	} else if cfg.LoRA == nil {
		return fmt.Errorf("no adapter is configured")
	}
	run, e := loadNativeRun(dir)
	if e == nil && nativeAlive(dir, run) {
		return fmt.Errorf("stop the deployment explicitly before changing its LoRA; no worker was interrupted")
	}
	if e != nil && !os.IsNotExist(e) {
		return fmt.Errorf("cannot verify stopped worker identity; configuration unchanged")
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if action == "remove" {
		cfg.Schema, cfg.LoRA = cfg.LoRA.BaseSchema, nil
	} else {
		bundle, err := loraBundleDigest("")
		if err != nil {
			return err
		}
		cfg.LoRA = &NativeLoRA{Revision: profile.Revision, BundleSHA256: bundle, BaseSchema: cfg.Schema}
		cfg.Schema = loraSchemaVersion
		if err = verifyLoRAAdmission(ctx, cfg, allowRisk); err != nil {
			return err
		}
		if err = stageLoRA(ctx, dir, cfg.LoRA); err != nil {
			return err
		}
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if err = writeJSONFile(filepath.Join(dir, "config.json"), cfg); err != nil {
		return err
	}
	fmt.Printf("Updated LoRA configuration for %s. Worker remains stopped; start explicitly. Original runtime and caches retained.\n", *name)
	return nil
}

func verifyLoRAAdmission(ctx context.Context, cfg NativeConfig, allowRisk bool) error {
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	h, err := DetectForBackend(ctx, "native")
	if err != nil {
		return fmt.Errorf("cannot refresh GPU budget before configuring LoRA")
	}
	if err = verifyWeightFit(loraWeightPlan(cfg), h.GPUs); err != nil {
		return err
	}
	estimate := cfg.Plan.EstimatedVRAMMiB + loraAllowanceMiB + (loadLoRAProfile().Size+(1<<20)-1)/(1<<20)
	for _, g := range h.GPUs {
		if g.UUID != cfg.Plan.GPU.UUID {
			continue
		}
		fmt.Printf("Estimated runtime budget: %d MiB; GPU: %d MiB free / %d MiB total\n", estimate, g.FreeMiB, g.TotalMiB)
		if estimate > g.FreeMiB*90/100 && !allowRisk {
			return fmt.Errorf("runtime estimate exceeds 90%% of current free VRAM; --allow-risk acknowledges only this heuristic")
		}
	}
	return nil
}

func loraHealthMatches(a *NativeLoRA, revision string) bool {
	if a == nil {
		return revision == ""
	}
	return revision == a.Revision
}
