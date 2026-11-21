package modelship

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func diffusionFixturePlan(t *testing.T) Plan {
	t.Helper()
	g := GPU{Index: 0, UUID: "GPU-00000000-0000-0000-0000-000000000001", Compute: "8.6", Driver: "610.43.02", TotalMiB: 49140, FreeMiB: 49140}
	var m Model
	m.Hub.ID, m.Hub.SHA = strings.ToLower(diffusionRepo), diffusionRevision
	keyDir := t.TempDir()
	if err := os.Chmod(keyDir, 0700); err != nil {
		t.Fatal(err)
	}
	key := filepath.Join(keyDir, "key")
	if err := os.WriteFile(key, []byte(diffusionFixtureKey+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := BuildPlan(m, Host{OS: "linux", Arch: "amd64", GPUs: []GPU{g}, ModelCacheDir: t.TempDir()}, Options{
		Diffusion: true, Research: true, Experimental: true, Backend: "native", APIKeyFile: key, Name: "fixture", Port: 18032,
	})
	if err != nil || validateDiffusionPlan(p) != nil || p.WeightBytes != 9900789304 || !p.WeightsFitFree || !p.WeightsFitTotal {
		t.Fatalf("reviewed plan rejected: %v; %+v", err, p)
	}
	return p
}

func TestDiffusionPlanningIsolation(t *testing.T) {
	p := diffusionFixturePlan(t)
	h := Host{OS: "linux", Arch: "amd64", GPUs: []GPU{p.GPU}, ModelCacheDir: t.TempDir()}
	o := Options{Backend: "native", Research: true, Experimental: true, APIKeyFile: p.APIKeyFile, Name: p.Name, Port: p.Port}
	for _, change := range []func(*Options){
		func(o *Options) { o.AllowRisk = true }, func(o *Options) { o.Backend = "docker" },
		func(o *Options) { o.Task = dualImageTask }, func(o *Options) { o.Vision = true },
		func(o *Options) { o.GGUFFile = "different.gguf" }, func(o *Options) { o.TrustModelCode = true },
	} {
		bad := o
		change(&bad)
		if _, err := buildDiffusionPlan(diffusionRepo, diffusionRevision, h, bad); err == nil {
			t.Fatal("mixed runtime policy accepted")
		}
	}
	if _, err := buildDiffusionPlan(diffusionRepo, "main", h, o); err == nil {
		t.Fatal("mutable revision accepted")
	}
	h.GPUs[0].FreeMiB = diffusionFreeMiB - 1
	o.Research, o.Experimental, o.APIKeyFile = false, false, ""
	blocked, err := buildDiffusionPlan(diffusionRepo, diffusionRevision, h, o)
	if err != nil || len(blocked.Blockers) != 4 || !blocked.WeightsFitFree || validateDiffusionPlan(blocked) == nil {
		t.Fatalf("consents, auth and experimental admission are not separate: %+v %v", blocked.Blockers, err)
	}
	p.Research = false
	if err := DeployNative(context.Background(), p, 0); err == nil || !strings.Contains(err.Error(), "invalid reviewed GGUF diffusion deployment") {
		t.Fatal("automatic runtime source must not bypass native plan validation")
	}
}

func TestDiffusionEditPlanningIsolation(t *testing.T) {
	gen := diffusionFixturePlan(t)
	h := Host{OS: "linux", Arch: "amd64", GPUs: []GPU{gen.GPU}, ModelCacheDir: gen.ModelCacheDir}
	o := Options{Diffusion: true, Task: diffusionEditTask, Backend: "native", Research: true, Experimental: true, APIKeyFile: gen.APIKeyFile, Name: gen.Name, Port: gen.Port}
	var m Model
	m.Hub.ID, m.Hub.SHA = diffusionRepo, diffusionRevision
	edit, err := BuildPlan(m, h, o)
	if err != nil || edit.Diffusion != diffusionEditProfile || edit.RecipeID != diffusionEditProfile || edit.Task != diffusionEditTask ||
		edit.WeightBytes != 11059819128 || edit.WeightResidentBytes != edit.WeightBytes || len(edit.Files) != 4 ||
		edit.Files[3] != "mmproj-Qwen3VL-8B-Instruct-F16.gguf" || diffusionCache(edit) == diffusionCache(gen) {
		t.Fatalf("edit profile selection or pinned bundle: %+v, %v", edit, err)
	}
	pins, err := diffusionRuntimePins(diffusionEditTask)
	if err != nil || validateDiffusionPlan(edit) != nil || pins[0].Name != "modelship-sd-edit-worker" || pins[0] == diffusionWorkerPin ||
		pins[1].Name != "modelship-sd-probe" || pins[1] == diffusionProbePin {
		t.Fatal("edit profile did not select the reviewed resident runtime")
	}
	for _, change := range []func(*Plan){
		func(p *Plan) { p.Task = diffusionTask }, func(p *Plan) { p.RecipeID = diffusionProfileID },
		func(p *Plan) { p.Diffusion = diffusionProfileID }, func(p *Plan) { p.Files = gen.Files },
		func(p *Plan) { p.WeightResidentBytes = gen.WeightResidentBytes },
	} {
		bad := edit
		bad.Blockers = nil
		change(&bad)
		if validateDiffusionPlan(bad) == nil {
			t.Fatal("saved edit plan accepted a different task, profile, or bundle")
		}
	}
}

func TestDiffusionSavedSchemaAndConfigure(t *testing.T) {
	p := diffusionFixturePlan(t)
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir, err := nativeDir(p.Name)
	if err != nil || privateDir(dir) != nil {
		t.Fatal("fixture directory")
	}
	cfg := NativeConfig{Schema: diffusionSchemaVersion, Plan: p}
	persist := func(cfg NativeConfig) {
		t.Helper()
		if err := writeJSONFile(filepath.Join(dir, "config.json"), cfg); err != nil {
			t.Fatal(err)
		}
	}
	for _, change := range []func(*NativeConfig){
		func(c *NativeConfig) { c.Schema-- }, func(c *NativeConfig) { c.Schema++ },
		func(c *NativeConfig) { c.Plan.Diffusion = "" }, func(c *NativeConfig) { c.Plan.WeightResidentBytes-- },
		func(c *NativeConfig) { c.Plan.GGUF = &GGUFSpec{} }, func(c *NativeConfig) { c.RuntimeBinary = "bin/alternate" },
	} {
		bad := cfg
		change(&bad)
		persist(bad)
		if _, _, err := nativeConfig(p.Name); err == nil {
			t.Fatal("mismatched saved execution accepted")
		}
	}
	persist(cfg)
	if _, _, err := nativeConfig(p.Name); err != nil {
		t.Fatal(err)
	}
	empty, loop := "", "127.0.0.1"
	if err := ConfigureNative(context.Background(), p.Name, nil, &empty); err == nil {
		t.Fatal("authentication removal accepted")
	}
	if err := ConfigureNative(context.Background(), p.Name, &loop, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(filepath.Join(dir, "native_entry.py")); !os.IsNotExist(err) {
		t.Fatal("access-only config unexpectedly wrote a Python entry")
	}
}

func TestDiffusionVerifiedWheelExtraction(t *testing.T) {
	ctx := context.Background()
	inputs, libs := t.TempDir(), t.TempDir()
	var b bytes.Buffer
	z := zip.NewWriter(&b)
	const want = "nvidia/cuda_runtime/lib/libcudart.so.12"
	for name, contents := range map[string]string{want: "fixture SONAME payload", "../../not-extracted": "no"} {
		f, err := z.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.WriteString(f, contents); err != nil {
			t.Fatal(err)
		}
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	pin := diffusionWheel{ArtifactPin{Name: "fixture.whl", Size: int64(b.Len()), SHA256: fmt.Sprintf("%x", sha256.Sum256(b.Bytes()))}, []string{want}}
	if err := os.WriteFile(filepath.Join(inputs, pin.Name), b.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	if err := diffusionWheelLibraries(ctx, inputs, libs, pin, true); err != nil {
		t.Fatal(err)
	}
	if entries, err := os.ReadDir(libs); err != nil || len(entries) != 1 || entries[0].Name() != filepath.Base(want) {
		t.Fatal("extracted unreviewed files")
	}
	if err := diffusionWheelLibraries(ctx, inputs, libs, pin, false); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(libs, filepath.Base(want))
	if err := os.Chmod(path, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("fixture TAMPER payload"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, 0400); err != nil {
		t.Fatal(err)
	}
	if err := diffusionWheelLibraries(ctx, inputs, libs, pin, false); err == nil {
		t.Fatal("tampered installed CUDA library accepted")
	}
}

func TestDiffusionLockAndEnv(t *testing.T) {
	p := diffusionFixturePlan(t)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("HF_TOKEN", "synthetic-fixture-not-inherited")
	lock, err := lockDiffusionGPU(p.GPU.UUID)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	if f, err := lockDiffusionGPU(p.GPU.UUID); err == nil {
		f.Close()
		t.Fatal("two unrelated lock descriptors admitted")
	}
	lock.Close()
	lock2, err := lockDiffusionGPU(p.GPU.UUID)
	if err != nil {
		t.Fatal(err)
	}
	lock2.Close()
	env := strings.Join(diffusionEnv("/private/fixture", p.GPU.UUID, diffusionFD(4)), "\n")
	if strings.Contains(env, "synthetic-fixture") || !strings.Contains(env, "CUDA_VISIBLE_DEVICES="+p.GPU.UUID) || !strings.Contains(env, "LD_LIBRARY_PATH=/proc/self/fd/4") {
		t.Fatal("unreviewed or missing child environment")
	}
	if diffusionProbePassed("invalid MODELSHIP_SD_CUDA0_ALLOCATED") || !diffusionProbePassed("CUDA diagnostic\nMODELSHIP_SD_CUDA0_ALLOCATED\n") {
		t.Fatal("invalid CUDA probe marker rule")
	}
}

func TestDiffusionIdleInventoryUnknownAndActive(t *testing.T) {
	uuid := diffusionFixturePlan(t).GPU.UUID
	gpu := func(processes string) string { return "<gpu><uuid>" + uuid + "</uuid>" + processes + "</gpu>" }
	for _, tc := range []struct {
		body string
		want bool
	}{
		{gpu("<processes>\n  </processes>"), true},
		{gpu("<processes/>"), true},
		{gpu("<processes>N/A</processes>"), false},
		{gpu("<processes><process_info><pid>23</pid><type>G</type></process_info></processes>"), false},
		{gpu("<processes><process_info><pid>24</pid><type>C</type></process_info></processes>"), false},
		{gpu(""), false},
		{gpu("<processes/><processes/>"), false},
		{gpu("<processes/>") + gpu("<processes/>"), false},
		{"<gpu><uuid>other</uuid><processes/></gpu>", false},
	} {
		body := []byte("<?xml version=\"1.0\"?><!DOCTYPE nvidia_smi_log SYSTEM \"nvsmi_device_v12.dtd\"><nvidia_smi_log>" + tc.body + "</nvidia_smi_log>")
		if got := diffusionIdleInventory(body, uuid); got != tc.want {
			t.Fatalf("unexpected idle=%v for %s", got, tc.body)
		}
	}
}

func TestDiffusionRemainingAndAncestors(t *testing.T) {
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, "protected")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := diffusionDir(dir); err != nil {
		t.Fatal(err)
	}
	pin := ArtifactPin{Name: "fixture.gguf", Size: 10}
	if err := os.WriteFile(filepath.Join(dir, pin.Name+".partial"), []byte("123"), 0600); err != nil {
		t.Fatal(err)
	}
	if got := diffusionCacheRemaining(dir, pin); got != 7 {
		t.Fatalf("remaining=%d", got)
	}
	if err := os.Chmod(root, 0777); err != nil {
		t.Fatal(err)
	}
	if err := diffusionDir(dir); err == nil {
		t.Fatal("writable ancestor accepted")
	}
}

func TestDiffusionRetainsStateWhilePrivateChildLockHeld(t *testing.T) {
	p := diffusionFixturePlan(t)
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir, err := nativeDir(p.Name)
	if err != nil || privateDir(dir) != nil {
		t.Fatal("fixture directory")
	}
	if err := writeJSONFile(filepath.Join(dir, "config.json"), NativeConfig{Schema: diffusionSchemaVersion, Plan: p}); err != nil {
		t.Fatal(err)
	}
	lock, err := lockDiffusionFile(dir, ".diffusion-child.lock")
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	for _, op := range []string{"stop", "uninstall"} {
		if err := ManageNative(context.Background(), op, p.Name); err == nil || !strings.Contains(err.Error(), "state retained") {
			t.Fatalf("%s unexpectedly accepted: %v", op, err)
		}
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	if err := ManageNative(context.Background(), "uninstall", p.Name); err != nil {
		t.Fatal(err)
	}
}
