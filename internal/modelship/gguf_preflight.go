package modelship

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Consume the entire pipe but retain bounded diagnostics, even for a noisy
// executable. The child has a curated environment without HF/API credentials.
type boundedDiagnostic struct {
	mu        sync.Mutex
	data      bytes.Buffer
	truncated bool
}

func (b *boundedDiagnostic) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := len(p)
	keep := min(n, (64<<10)-b.data.Len())
	if keep > 0 {
		_, _ = b.data.Write(p[:keep])
	}
	b.truncated = b.truncated || keep < n
	return n, nil
}
func diagnosticText(raw string) string {
	lines := strings.Split(raw, "\n")
	for i := range lines {
		lines[i] = terminalText(lines[i])
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}
func probeLlamaCUDA(ctx context.Context, cfg NativeConfig, dir string) error {
	return probeLlamaCUDAWithin(ctx, cfg, dir, 60*time.Second)
}
func probeLlamaCUDAWithin(ctx context.Context, cfg NativeConfig, dir string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	path, e := fileWithin(dir, cfg.RuntimeBinary)
	if e != nil {
		return e
	}
	cmd := exec.CommandContext(ctx, path, "--list-devices")
	cmd.Env = llamaEnv(cfg, dir)
	cmd.Dir = dir
	cmd.WaitDelay = time.Second
	var output boundedDiagnostic
	cmd.Stdout = &output
	cmd.Stderr = &output
	e = cmd.Run()
	message := diagnosticText(output.data.String())
	if output.truncated {
		message += "\n[diagnostics truncated at 64 KiB]"
	}
	status := "success"
	if e != nil {
		status = terminalText(e.Error())
	}
	report := "Command: llama-server --list-devices\nResult: " + status + "\n" + message + "\n"
	log := filepath.Join(dir, "runtime-preflight.log")
	f, logErr := os.CreateTemp(dir, ".runtime-preflight-")
	if logErr == nil {
		tmp := f.Name()
		defer os.Remove(tmp)
		_, logErr = f.WriteString(report)
		closeErr := f.Close()
		if logErr == nil {
			logErr = closeErr
		}
		if logErr == nil {
			logErr = os.Rename(tmp, log)
		}
	}
	if e == nil && strings.Contains(message, "CUDA0") {
		if logErr != nil {
			return fmt.Errorf("CUDA enumeration passed but private preflight diagnostics could not be saved")
		}
		return nil
	}
	reason := "llama.cpp CUDA device enumeration failed"
	switch {
	case ctx.Err() != nil:
		reason = "llama.cpp device enumeration timed out or was canceled"
	case strings.Contains(message, "not found") && (strings.Contains(message, "GLIBC_") || strings.Contains(message, "GLIBCXX_") || strings.Contains(message, "CXXABI_")):
		reason = "llama.cpp host ABI mismatch (glibc/libstdc++); do not replace host libraries"
	case strings.Contains(message, "error while loading shared libraries") || strings.Contains(message, "cannot open shared object file"):
		reason = "llama.cpp shared-library dependency could not be loaded"
	case e == nil:
		reason = "llama.cpp did not report CUDA0; no CPU fallback is allowed"
	}
	detail := message
	if len(detail) > 2048 {
		detail = detail[:2048] + "\n[see private report for remaining output]"
	}
	if detail == "" && e != nil {
		detail = terminalText(e.Error())
	}
	location := "Private diagnostics: " + log
	if logErr != nil {
		location = "Private diagnostics could not be saved"
	}
	return fmt.Errorf("%s. %s\n%s\nNo model weights downloaded by this preflight; no CPU fallback or host library changes attempted", reason, location, detail)
}
