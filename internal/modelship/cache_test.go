package modelship

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestModelCachePrecedenceAndTilde(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	for _, key := range []string{"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"} {
		t.Setenv(key, "")
	}
	check := func(want string) {
		t.Helper()
		got, err := modelCachePath()
		if err != nil || got != want {
			t.Fatalf("%s != %s: %v", got, want, err)
		}
	}
	check(filepath.Join(home, ".cache", "huggingface", "hub"))
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, "xdg"))
	check(filepath.Join(home, "xdg", "huggingface", "hub"))
	t.Setenv("HF_HOME", "~/research")
	check(filepath.Join(home, "research", "hub"))
	t.Setenv("HUGGINGFACE_HUB_CACHE", filepath.Join(home, "old-cache"))
	check(filepath.Join(home, "old-cache"))
	t.Setenv("HF_HUB_CACHE", "~/weights")
	check(filepath.Join(home, "weights"))
}

func TestCacheInspectionCountsOnlyExactPinnedSizedWeights(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HF_HUB_CACHE", root)
	m := textFixture()
	m.Hub.Siblings = []ModelFile{{Name: "model.safetensors", Size: 32}, {Name: "model-00002-of-00002.safetensors", Size: 40}, {Name: "config.json", Size: 10}}
	snapshot := filepath.Join(root, "models--org--model", "snapshots", m.Hub.SHA)
	if err := os.MkdirAll(snapshot, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(snapshot, "model.safetensors"), make([]byte, 32), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(snapshot, "model-00002-of-00002.safetensors"), make([]byte, 39), 0600); err != nil {
		t.Fatal(err)
	}
	var h Host
	InspectModelCache(&h, m)
	if !h.ModelCacheInspected || h.CachedWeightBytes != 32 || h.ModelCacheDir != root || h.ModelCacheFreeMiB <= 0 {
		t.Fatalf("%+v", h)
	}
	m.Hub.SHA = strings.Repeat("c", 40)
	InspectModelCache(&h, m)
	if h.CachedWeightBytes != 0 {
		t.Fatal("counted a different revision's cached weights")
	}
}

func TestMissingCacheInspectionDoesNotCreateDirectories(t *testing.T) {
	root := filepath.Join(t.TempDir(), "not-created", "hub")
	t.Setenv("HF_HUB_CACHE", root)
	var h Host
	InspectModelCache(&h, textFixture())
	if !h.ModelCacheInspected || h.ModelCacheFreeMiB <= 0 {
		t.Fatalf("%+v", h)
	}
	if _, err := os.Stat(root); !os.IsNotExist(err) {
		t.Fatal("inspection created model cache")
	}
}

func TestCacheFileAncestorIsNotSilentlyAccepted(t *testing.T) {
	root := t.TempDir()
	file := filepath.Join(root, "file")
	if err := os.WriteFile(file, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HF_HUB_CACHE", file)
	var h Host
	InspectModelCache(&h, textFixture())
	if h.ModelCacheInspected || len(h.Warnings) == 0 {
		t.Fatalf("%+v", h)
	}
}

func TestNativeDiskGuardAccountsForPinnedCacheAndCannotBeOverridden(t *testing.T) {
	o := optsFixture()
	o.Backend = "native"
	o.AllowRisk = true
	h := hostFixture()
	h.ModelCacheInspected = true
	h.ModelCacheDir = "/cache"
	h.ModelCacheFreeMiB = 5000
	p, err := BuildPlan(textFixture(), h, o)
	if err != nil || !strings.Contains(strings.Join(p.Blockers, ";"), "native model-cache disk") {
		t.Fatalf("%+v %v", p, err)
	}
	h.CachedWeightBytes = textFixture().Hub.Siblings[0].Size
	p, err = BuildPlan(textFixture(), h, o)
	if err != nil || len(p.Blockers) > 0 || p.ModelCacheDir != "/cache" {
		t.Fatalf("cached restart incorrectly budgeted full download: %+v %v", p, err)
	}
}
