package modelship

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const diffusionRuntimeRelease = "https://github.com/0xburden/modelship/releases/download/runtime-sd-532d2c1-u2204-r1/"
const diffusionRuntimeNotice = "THIRD_PARTY_NOTICES.txt"

func diffusionArchivePin(task string) ArtifactPin {
	var pin ArtifactPin
	switch task {
	case diffusionTask:
		pin = ArtifactPin{Name: "sd-532d2c1-u2204-cuda12.8-generation-r1.tar.gz", Size: 112140725, SHA256: "ef44df98f335c4b88356ab2bf91ab9d794297d97f46bac262cb8e1d19ac9bdf6"}
	case diffusionEditTask:
		pin = ArtifactPin{Name: "sd-532d2c1-u2204-cuda12.8-edit-r1.tar.gz", Size: 112133206, SHA256: "cd73f1e269c6f79dc3729b3b6755b81236d539d50f8bab410ad2af795629baa5"}
	default:
		return ArtifactPin{}
	}
	pin.URL = diffusionRuntimeRelease + pin.Name
	return pin
}

// Called only after deploy confirmation. Use separate private staging so the
// downloader does not reacquire the deployment lock; restart never calls this.
func fetchDiffusionRuntime(ctx context.Context, client *http.Client, dir string, pin ArtifactPin, binaries [2]ArtifactPin) error {
	if pin.Size <= 0 || pin.Size > 512<<20 {
		return fmt.Errorf("invalid pinned diffusion runtime archive size")
	}
	stage, err := os.MkdirTemp(dir, ".runtime-stage-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(stage)
	file := filepath.Join(stage, pin.Name)
	if err := downloadVerified(ctx, client, pin.URL, "", file, pin); err != nil {
		return fmt.Errorf("GGUF diffusion runtime: %w", err)
	}
	if err := extractDiffusionRuntime(ctx, file, stage, binaries); err != nil {
		return err
	}
	for _, name := range []string{diffusionRuntimeNotice, binaries[0].Name, binaries[1].Name} {
		if err := os.Link(filepath.Join(stage, name), filepath.Join(dir, name)); err != nil {
			return fmt.Errorf("could not install verified diffusion runtime")
		}
	}
	return nil
}

// Only the three reviewed regular files may occur, in the packaged order.
// Ignore archive mode/ownership; verify each ELF against the existing GPU-tested pin.
func extractDiffusionRuntime(ctx context.Context, file, dir string, binaries [2]ArtifactPin) error {
	f, err := os.Open(file)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("invalid diffusion runtime gzip")
	}
	defer gz.Close()
	r := tar.NewReader(diffusionContextReader{ctx, gz})
	for i, name := range []string{diffusionRuntimeNotice, binaries[0].Name, binaries[1].Name} {
		h, err := r.Next()
		if err != nil || h.Format != tar.FormatUSTAR || h.Typeflag != tar.TypeReg || h.Name != name || h.Size <= 0 ||
			(i == 0 && h.Size > 256<<10) || (i > 0 && h.Size != binaries[i-1].Size) {
			return fmt.Errorf("unexpected diffusion runtime archive member")
		}
		mode := os.FileMode(0400)
		if i > 0 {
			mode = 0500
		}
		out, err := os.OpenFile(filepath.Join(dir, name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, mode)
		if err != nil {
			return err
		}
		digest, copyErr := diffusionDigest(ctx, r, out, h.Size)
		closeErr := out.Close()
		if copyErr != nil || closeErr != nil || i > 0 && digest != binaries[i-1].SHA256 {
			return fmt.Errorf("pinned diffusion runtime member verification failed")
		}
	}
	if _, err := r.Next(); err != io.EOF {
		return fmt.Errorf("unexpected trailing diffusion runtime member")
	}
	// Python's USTAR writer pads its last 10 KiB record with zeros.
	padding, err := io.ReadAll(io.LimitReader(diffusionContextReader{ctx, gz}, 10240))
	if err != nil || len(padding) == 10240 {
		return fmt.Errorf("invalid diffusion runtime archive trailer")
	}
	for _, b := range padding {
		if b != 0 {
			return fmt.Errorf("invalid diffusion runtime archive padding")
		}
	}
	return nil
}
