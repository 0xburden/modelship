package modelship

import (
	"context"
	"errors"

	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoRAConfigAndStoppedRollback(t *testing.T) {
	profile := loadLoRAProfile()
	m := imageFixture()
	m.Hub.ID, m.Hub.SHA, m.Index.Class = profile.BaseModel, profile.BaseRevision, "QwenImage21Pipeline"
	h := hostFixture()
	h.GPUs[0].FreeMiB, h.GPUs[0].TotalMiB = 81920, 81920
	o := optsFixture()
	o.Backend, o.Experimental, o.Research, o.Task = "native", true, true, dualImageTask
	p, err := BuildPlan(m, h, o)
	if err != nil || len(p.Blockers) != 0 {
		t.Fatalf("fixture: %+v %v", p, err)
	}
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir, err := nativeDir(p.Name)
	if err != nil {
		t.Fatal(err)
	}
	if err = privateDir(dir); err != nil {
		t.Fatal(err)
	}
	config := filepath.Join(dir, "config.json")
	cfg := NativeConfig{Schema: 10, Plan: p}
	if err = writeJSONFile(config, cfg); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(config)
	entry := filepath.Join(dir, "native_entry.py")
	if err = os.WriteFile(entry, []byte("original"), 0600); err != nil {
		t.Fatal(err)
	}
	args := []string{"add", "--name", p.Name, "https://huggingface.co/" + profile.Repo}
	if err = loraCommand(context.Background(), args); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(config); string(got) != string(before) {
		t.Fatal("preview changed state")
	}
	if _, err = os.Stat(filepath.Join(dir, loraBundle)); !os.IsNotExist(err) {
		t.Fatal("preview staged files")
	}
	if err = loraCommand(context.Background(), []string{"add", "--name", p.Name, "--yes", profile.Repo}); err == nil || !strings.Contains(err.Error(), "--research") {
		t.Fatal(err)
	}
	unlock, err := nativeLock(dir)
	if err != nil {
		t.Fatal(err)
	}
	err = loraCommand(context.Background(), args)
	unlock()
	if err == nil {
		t.Fatal("lifecycle lock ignored")
	}
	add := &NativeLoRA{Revision: profile.Revision, BundleSHA256: strings.Repeat("c", 64), BaseSchema: 10}
	cfg.LoRA = add
	if validateLoRAConfig(cfg) == nil {
		t.Fatal("accepted LoRA in schema 10")
	}
	cfg.Schema = 11
	if err = validateLoRAConfig(cfg); err != nil {
		t.Fatal(err)
	}
	for _, task := range []string{"text-to-image", "image-to-image", "text-generation"} {
		invalid := cfg
		invalid.Plan.Task = task
		if (validateLoRAConfig(invalid) != nil) != (task == "text-generation") {
			t.Fatal(task)
		}
	}
	if loraWeightPlan(cfg).WeightResidentBytes != p.WeightResidentBytes+profile.Size || !loraHealthMatches(add, profile.Revision) || loraHealthMatches(add, "") {
		t.Fatal("invalid weight/health accounting")
	}
	if err = writeJSONFile(config, cfg); err != nil {
		t.Fatal(err)
	}
	if err = loraCommand(context.Background(), []string{"remove", "--name", p.Name, "--yes"}); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(config); string(got) != string(before) {
		t.Fatal("rollback did not restore original config")
	}
	if got, _ := os.ReadFile(entry); string(got) != "original" {
		t.Fatal("rollback touched original runtime")
	}
}

func TestLoRABundleRejectsModifiedOrMissingFiles(t *testing.T) {
	d := t.TempDir()
	expected, err := loraBundleDigest("")
	if err != nil {
		t.Fatal(err)
	}
	a := &NativeLoRA{BundleSHA256: expected}
	root := loraRuntime(d, a)
	if err = privateDir(root); err != nil {
		t.Fatal(err)
	}
	for _, n := range loraAssets {
		data, e := assets.ReadFile("assets/" + n)
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(filepath.Join(root, n), data, 0600); e != nil {
			t.Fatal(e)
		}
	}
	if got, e := loraBundleDigest(root); e != nil || got != expected {
		t.Fatalf("digest %s: %v", got, e)
	}
	if verifyLoRARuntime(d, a) == nil {
		t.Fatal("started without adapter file")
	}
	if err = os.WriteFile(filepath.Join(root, "server.py"), []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = verifyLoRARuntime(d, a); err == nil || !strings.Contains(err.Error(), "bundle digest") {
		t.Fatal(err)
	}
}

func TestLoRAStagingRecoversInterruptedBundle(t *testing.T) {
	digest, err := loraBundleDigest("")
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(t.TempDir(), loraBundle, digest)
	partial := root + ".partial"
	if err := privateDir(filepath.Dir(root)); err != nil {
		t.Fatal(err)
	}
	if err := privateDir(partial); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(partial, "server.py"), []byte("truncated write"), 0600); err != nil {
		t.Fatal(err)
	}
	data := []byte("synthetic pinned fixture")
	artifact := filepath.Join(partial, "fixture.safetensors")
	if err := os.WriteFile(artifact, data, 0600); err != nil {
		t.Fatal(err)
	}
	hash, err := digestFile(artifact)
	if err != nil {
		t.Fatal(err)
	}
	profile := imageLoRAProfile{File: filepath.Base(artifact), Size: int64(len(data)), SHA256: hash}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	download := func(_ context.Context, path string) error {
		if path != artifact {
			t.Fatalf("wrong staged artifact path: %s", path)
		}
		if _, err := os.Stat(root); !os.IsNotExist(err) {
			t.Fatal("published before verification")
		}
		return nil
	}
	if err := stageLoRABundle(ctx, root, digest, profile, download); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := os.Stat(root); !os.IsNotExist(err) {
		t.Fatal("published canceled bundle")
	}
	if err := stageLoRABundle(context.Background(), root, digest, profile, download); err != nil {
		t.Fatal(err)
	}
	if err := verifyLoRABundle(root, digest, profile); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(partial); !os.IsNotExist(err) {
		t.Fatal("partial directory still present after completion")
	}
	if err := stageLoRABundle(context.Background(), root, digest, profile, func(context.Context, string) error { t.Fatal("downloaded already verified bundle"); return nil }); err != nil {
		t.Fatal(err)
	}
}
