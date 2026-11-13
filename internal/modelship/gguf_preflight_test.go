package modelship

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestGGUFHostABIReportsActualVersionGap(t *testing.T) {
	for _, tc := range []struct {
		have, need runtimeABI
		ok         bool
	}{
		{runtimeABI{"2.35", "3.4.30"}, runtimeABI{"2.38", "3.4.32"}, false},
		{runtimeABI{"2.35", "3.4.29"}, runtimeABI{"2.35", "3.4.29"}, true},
		{runtimeABI{"2.39", "3.4.33"}, runtimeABI{"2.35", "3.4.29"}, true},
		{runtimeABI{"2.35", "3.4.28"}, runtimeABI{"2.35", "3.4.29"}, false},
		{runtimeABI{"", ""}, runtimeABI{"2.35", "3.4.29"}, false},
	} {
		e := validateRuntimeABI(tc.have, tc.need)
		if (e == nil) != tc.ok {
			t.Fatal(tc, e)
		}
	}
	if got := maxABISymbol([]string{"GLIBC_2.9", "GLIBC_2.35", "GLIBC_PRIVATE", "GLIBC_ABI_DT_RELR", "GLIBC_2.4", "GLIBC_2.x"}, "GLIBC"); got != "2.35" {
		t.Fatal(got)
	}
	if got := maxABISymbol([]string{"GLIBCXX_3.4.9", "GLIBCXX_3.4.32", "GLIBCXX_DEBUG_MESSAGE_LENGTH"}, "GLIBCXX"); got != "3.4.32" {
		t.Fatal(got)
	}
}
func probeFixture(t *testing.T, body string) (NativeConfig, string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "runtime", "llama-server")
	if e := os.Mkdir(filepath.Dir(path), 0700); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(path, []byte("#!/bin/sh\n"+body+"\n"), 0700); e != nil {
		t.Fatal(e)
	}
	return NativeConfig{Plan: Plan{GPU: GPU{UUID: "GPU-fixture"}}, RuntimeBinary: "runtime/llama-server"}, dir
}
func TestGGUFPreflightPreservesUsefulLoaderDiagnosticsPrivately(t *testing.T) {
	cfg, dir := probeFixture(t, `printf '%s\n' 'libllama.so: version GLIBC_2.38 not found' 'libstdc++.so.6: version GLIBCXX_3.4.32 not found' >&2; exit 1`)
	e := probeLlamaCUDA(context.Background(), cfg, dir)
	if e == nil || !strings.Contains(e.Error(), "host ABI mismatch") || !strings.Contains(e.Error(), "GLIBCXX_3.4.32") {
		t.Fatal(e)
	}
	path := filepath.Join(dir, "runtime-preflight.log")
	st, e := os.Stat(path)
	if e != nil || st.Mode().Perm() != 0600 {
		t.Fatal("diagnostic report not private", e)
	}
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "GLIBC_2.38") || !strings.Contains(string(data), "exit status 1") {
		t.Fatal("real loader error discarded")
	}
}
func TestGGUFPreflightDistinguishesDependencyDeviceAndTimeout(t *testing.T) {
	for _, tc := range []struct {
		name, script, want string
		timeout            time.Duration
	}{
		{"library", `echo 'error while loading shared libraries: libcublas.so.12: cannot open shared object file' >&2; exit 127`, "shared-library dependency", time.Second},
		{"cpu-only", `echo 'Available devices: CPU'`, "did not report CUDA0", time.Second},
		{"timeout", `exec sleep 10`, "timed out", 20 * time.Millisecond},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, dir := probeFixture(t, tc.script)
			e := probeLlamaCUDAWithin(context.Background(), cfg, dir, tc.timeout)
			if e == nil || !strings.Contains(e.Error(), tc.want) {
				t.Fatal(e)
			}
		})
	}
}
func TestGGUFPreflightScrubsEnvironmentAndTerminalControls(t *testing.T) {
	t.Setenv("HF_TOKEN", "sensitive-hf-value")
	t.Setenv("MODEL_API_KEY", "sensitive-api-value")
	cfg, dir := probeFixture(t, `printf '%s\n' "${HF_TOKEN-unset}" "${MODEL_API_KEY-unset}"; printf '\033[31mCUDA unavailable\n'; exit 1`)
	e := probeLlamaCUDA(context.Background(), cfg, dir)
	if e == nil || strings.ContainsAny(e.Error(), "\x1b\x00") || strings.Contains(e.Error(), "sensitive-") {
		t.Fatal("unsafe diagnostics", e)
	}
	data, _ := os.ReadFile(filepath.Join(dir, "runtime-preflight.log"))
	if strings.Contains(string(data), "sensitive-") || strings.ContainsRune(string(data), '\x1b') {
		t.Fatal("unsafe diagnostic report")
	}
}
func TestGGUFPreflightBoundsOutputAndRetainsSuccessReport(t *testing.T) {
	cfg, dir := probeFixture(t, `echo 'Available devices: CUDA0: fixture GPU'`)
	if e := probeLlamaCUDA(context.Background(), cfg, dir); e != nil {
		t.Fatal(e)
	}
	var buffer boundedDiagnostic
	data := []byte(strings.Repeat("x", 128<<10))
	n, e := buffer.Write(data)
	if e != nil || n != len(data) || buffer.data.Len() != 64<<10 || !buffer.truncated {
		t.Fatal("output bound failed")
	}
	if e := os.WriteFile(filepath.Join(dir, "untouched"), []byte("preserved"), 0600); e != nil {
		t.Fatal(e)
	}
	_ = os.Remove(filepath.Join(dir, "runtime-preflight.log"))
	if e := os.Symlink("untouched", filepath.Join(dir, "runtime-preflight.log")); e != nil {
		t.Fatal(e)
	}
	if e := probeLlamaCUDA(context.Background(), cfg, dir); e != nil {
		t.Fatal(e)
	}
	got, _ := os.ReadFile(filepath.Join(dir, "untouched"))
	if string(got) != "preserved" {
		t.Fatal("diagnostic write followed a symlink")
	}
}
