package modelship

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Exercise the same descriptor-numbered exec/load boundary with a real ELF
// and fixture data. No CUDA or runtime downloads are needed.
func TestDiffusionExecFromVerifiedDescriptors(t *testing.T) {
	root := t.TempDir()
	bundle := filepath.Join(root, "bundle:with;delimiters")
	if err := os.Mkdir(bundle, 0700); err != nil {
		t.Fatal(err)
	}
	marker := []byte("verified fixture payload")
	if err := os.WriteFile(filepath.Join(bundle, "fixture"), marker, 0600); err != nil {
		t.Fatal(err)
	}
	dir, err := diffusionDirectoryFD(bundle)
	if err != nil {
		t.Fatal(err)
	}
	defer dir.Close()
	bin, err := os.ReadFile("/bin/cat")
	if err != nil {
		t.Fatal(err)
	}
	pin := ArtifactPin{Name: "cat", Size: int64(len(bin)), SHA256: fmt.Sprintf("%x", sha256.Sum256(bin))}
	if err := os.WriteFile(filepath.Join(root, pin.Name), bin, 0500); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	verified, err := diffusionVerifiedFile(ctx, root, pin, 0500, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	defer verified.Close()
	if err := os.Remove(filepath.Join(root, pin.Name)); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(bundle, filepath.Join(root, "renamed")); err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(ctx, diffusionFD(3), diffusionFD(4)+"/fixture")
	cmd.Env = diffusionEnv(root, "fixture", diffusionFD(4))
	cmd.ExtraFiles = []*os.File{verified, dir}
	release := guardDiffusionChild(cmd)
	defer release()
	got, err := cmd.CombinedOutput()
	if err != nil || string(got) != string(marker) {
		t.Fatalf("descriptor launch failed: combined stdout/stderr=%q err=%v", got, err)
	}
}

func TestDiffusionChildDeathFixture(t *testing.T) {
	if os.Getenv("MODELSHIP_TEST_DIFFUSION_PARENT_EXIT") != "1" {
		return
	}
	lock, err := lockDiffusionFile(os.Getenv("MODELSHIP_TEST_DIFFUSION_ROOT"), "worker.lock")
	if err != nil {
		os.Exit(2)
	}
	cmd := exec.Command("/bin/sleep", "30")
	cmd.ExtraFiles = []*os.File{lock}
	_ = guardDiffusionChild(cmd) // Keep the creating thread through os.Exit.
	if cmd.Start() != nil {
		os.Exit(3)
	}
	os.Exit(0)
}

func TestDiffusionChildReleasesInheritedLockAfterParentExit(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestDiffusionChildDeathFixture$")
	cmd.Env = append(os.Environ(), "MODELSHIP_TEST_DIFFUSION_PARENT_EXIT=1", "MODELSHIP_TEST_DIFFUSION_ROOT="+root)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fixture supervisor failed: %v %s", err, strings.TrimSpace(string(out)))
	}
	for {
		if lock, err := lockDiffusionFile(root, "worker.lock"); err == nil {
			lock.Close()
			return
		}
		select {
		case <-ctx.Done():
			t.Fatal("child kept the inherited lock after supervisor death")
		case <-time.After(20 * time.Millisecond):
		}
	}
}
