package modelship

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixturePin(body []byte) ArtifactPin {
	sum := sha256.Sum256(body)
	return ArtifactPin{Name: "fixture", Size: int64(len(body)), SHA256: hex.EncodeToString(sum[:])}
}
func TestGGUFArtifactDownloadPinnedResumableAndPrivate(t *testing.T) {
	body := []byte("synthetic runtime or model artifact bytes, NOT actual weights")
	for _, resume := range []bool{false, true} {
		t.Run(fmt.Sprint(resume), func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if r.Header.Get("Authorization") != "Bearer fixture-hf-token" {
					t.Error("missing private download credential")
				}
				start := 0
				if resume {
					start = 8
					if r.Header.Get("Range") != "bytes=8-" {
						t.Error("partial cache not resumed")
					}
					w.Header().Set("Content-Range", fmt.Sprintf("bytes 8-%d/%d", len(body)-1, len(body)))
					w.WriteHeader(206)
				}
				_, _ = w.Write(body[start:])
			}))
			defer server.Close()
			path := filepath.Join(t.TempDir(), "cache", "artifact")
			if resume {
				_ = os.MkdirAll(filepath.Dir(path), 0700)
				_ = os.WriteFile(path+".partial", body[:8], 0600)
			}
			if e := downloadVerified(context.Background(), server.Client(), server.URL, "fixture-hf-token", path, fixturePin(body)); e != nil {
				t.Fatal(e)
			}
			got, e := os.ReadFile(path)
			if e != nil || !bytes.Equal(got, body) {
				t.Fatal("corrupt artifact accepted")
			}
			st, _ := os.Stat(path)
			if st.Mode().Perm() != 0600 {
				t.Fatal("artifact is not private")
			}
			if e = downloadVerified(context.Background(), server.Client(), server.URL, "", path, fixturePin(body)); e != nil || calls != 1 {
				t.Fatal("verified cache downloaded again", e, calls)
			}
		})
	}
}
func TestGGUFArtifactRejectsCorruptionSymlinksAndUnsafeResume(t *testing.T) {
	body := []byte("synthetic safe bytes")
	for _, mode := range []string{"digest", "size", "range", "symlink", "bad-cache"} {
		t.Run(mode, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if mode == "range" {
					w.Header().Set("Content-Range", "bytes 0-5/6")
					w.WriteHeader(206)
				}
				_, _ = w.Write(body)
			}))
			defer server.Close()
			path := filepath.Join(t.TempDir(), "artifact")
			pin := fixturePin(body)
			switch mode {
			case "digest":
				pin.SHA256 = strings.Repeat("f", 64)
			case "size":
				pin.Size--
			case "range":
				_ = os.WriteFile(path+".partial", body[:2], 0600)
			case "symlink":
				outside := filepath.Join(t.TempDir(), "untouched")
				_ = os.WriteFile(outside, []byte("preserved"), 0600)
				_ = os.Symlink(outside, path+".partial")
			case "bad-cache":
				_ = os.WriteFile(path, body[:2], 0600)
			}
			if e := downloadVerified(context.Background(), server.Client(), server.URL, "", path, pin); e == nil {
				t.Fatal("unsafe artifact accepted")
			}
			if mode != "bad-cache" {
				if _, e := os.Lstat(path); !os.IsNotExist(e) {
					t.Fatal("unverified artifact promoted")
				}
			}
		})
	}
}
func TestGGUFDownloadRedirectPolicyNeverLeaksToken(t *testing.T) {
	client := artifactHTTPClient()
	initial, _ := http.NewRequest("GET", "https://huggingface.co/source", nil)
	request, _ := http.NewRequest("GET", "https://cdn.example/file", nil)
	request.Header.Set("Authorization", "Bearer must-not-leak")
	if e := client.CheckRedirect(request, []*http.Request{initial}); e != nil || request.Header.Get("Authorization") != "" {
		t.Fatal("credential forwarded to external asset host")
	}
	request.URL = &url.URL{Scheme: "http", Host: "cdn.example"}
	if e := client.CheckRedirect(request, []*http.Request{initial}); e == nil {
		t.Fatal("HTTPS downgrade accepted")
	}
}
func archiveFixture(t *testing.T, entries []*tar.Header, contents []string) string {
	t.Helper()
	var b bytes.Buffer
	gz := gzip.NewWriter(&b)
	tw := tar.NewWriter(gz)
	for i, entry := range entries {
		if e := tw.WriteHeader(entry); e != nil {
			t.Fatal(e)
		}
		if entry.Typeflag == tar.TypeReg {
			if _, e := tw.Write([]byte(contents[i])); e != nil {
				t.Fatal(e)
			}
		}
	}
	if e := tw.Close(); e != nil {
		t.Fatal(e)
	}
	if e := gz.Close(); e != nil {
		t.Fatal(e)
	}
	path := filepath.Join(t.TempDir(), "runtime.tgz")
	if e := os.WriteFile(path, b.Bytes(), 0600); e != nil {
		t.Fatal(e)
	}
	return path
}
func TestGGUFRuntimeExtractsNoHostPathsAndRecordsHashes(t *testing.T) {
	archive := archiveFixture(t, []*tar.Header{{Name: "build/bin/llama-server", Typeflag: tar.TypeReg, Size: 4, Mode: 0755}, {Name: "build/bin/libggml.so.1", Typeflag: tar.TypeReg, Size: 3, Mode: 0644}, {Name: "build/bin/libggml.so", Typeflag: tar.TypeSymlink, Linkname: "libggml.so.1"}}, []string{"fake", "lib", ""})
	dir := t.TempDir()
	if e := extractRuntime(archive, filepath.Join(dir, "runtime", "core")); e != nil {
		t.Fatal(e)
	}
	binary, libs, files, e := installedRuntime(dir)
	if e != nil || filepath.Base(binary) != "llama-server" || len(libs) != 1 || len(files) != 3 {
		t.Fatal("runtime manifest invalid", binary, libs, files, e)
	}
	for _, file := range files {
		if file.Link != "" {
			continue
		}
		actual, e := digestFile(filepath.Join(dir, file.Path))
		if e != nil || actual != file.SHA256 {
			t.Fatal("runtime hash mismatch")
		}
	}
}
func TestGGUFRuntimeArchiveRejectsPathAndLinkEscapes(t *testing.T) {
	for _, header := range []*tar.Header{
		{Name: "../escaped", Typeflag: tar.TypeReg, Size: 1, Mode: 0644},
		{Name: "/absolute", Typeflag: tar.TypeReg, Size: 1, Mode: 0644},
		{Name: "alias", Typeflag: tar.TypeSymlink, Linkname: "../outside"},
		{Name: "alias", Typeflag: tar.TypeSymlink, Linkname: "/etc/passwd"},
		{Name: "device", Typeflag: tar.TypeChar},
		{Name: "hardlink", Typeflag: tar.TypeLink, Linkname: "file"},
	} {
		t.Run(header.Name, func(t *testing.T) {
			archive := archiveFixture(t, []*tar.Header{header}, []string{"x"})
			if e := extractRuntime(archive, filepath.Join(t.TempDir(), "root")); e == nil {
				t.Fatal("unsafe runtime archive accepted")
			}
		})
	}
}
func TestGGUFNativeStateBoundaryAndImmutableArtifacts(t *testing.T) {
	m, h, o := ggufPlanFixture(t)
	p, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir, e := nativeDir(p.Name)
	if e != nil {
		t.Fatal(e)
	}
	if e = privateDir(dir); e != nil {
		t.Fatal(e)
	}
	for _, schema := range []int{1, 2, 3, 4, nativeSchemaVersion + 1} {
		cfg := NativeConfig{Schema: schema, Plan: p}
		if e = writeJSONFile(filepath.Join(dir, "config.json"), cfg); e != nil {
			t.Fatal(e)
		}
		if _, _, e = nativeConfig(p.Name); e == nil {
			t.Fatal("legacy/future CLI schema accepted GGUF state", schema)
		}
	}
	legacy := NativeConfig{Schema: 5, Plan: p}
	if e = writeJSONFile(filepath.Join(dir, "config.json"), legacy); e != nil {
		t.Fatal(e)
	}
	if _, _, e = nativeConfig(p.Name); e != nil {
		t.Fatal("healthy schema-5 GGUF compatibility lost", e)
	}
	cfg := NativeConfig{Schema: nativeSchemaVersion, Plan: p}
	if e = writeJSONFile(filepath.Join(dir, "config.json"), cfg); e != nil {
		t.Fatal(e)
	}
	if _, _, e = nativeConfig(p.Name); e != nil {
		t.Fatal(e)
	}
	if e = verifyGGUFInstall(cfg, dir); e == nil {
		t.Fatal("uninstalled runtime accepted")
	}
	runtimeSpec, e := loadLlamaRuntime(llamaLegacyRuntime)
	if e != nil {
		t.Fatal(e)
	}
	for _, pin := range runtimeSpec.Artifacts {
		if len(pin.SHA256) != 64 || pin.Size < 1 || !strings.Contains(pin.URL, "/b12110/") {
			t.Fatal("unverified/moving runtime pin")
		}
	}
}
func TestGGUFChildFlagsAndEnvironmentCannotSilentlyOffload(t *testing.T) {
	m, h, o := ggufPlanFixture(t)
	p, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	cfg := NativeConfig{Plan: p, RuntimeLibraries: []string{"runtime/core/build/bin", "runtime/cuda/lib"}}
	t.Setenv("HF_TOKEN", "never-inherit-token")
	t.Setenv("LLAMA_ARG_FIT", "on")
	t.Setenv("LD_PRELOAD", "never-inherit-preload")
	t.Setenv("HTTPS_PROXY", "never-inherit-proxy")
	env := strings.Join(llamaEnv(cfg, "/private/deployment"), "\n")
	for _, bad := range []string{"HF_TOKEN", "LLAMA_ARG_", "LD_PRELOAD", "HTTPS_PROXY", "never-inherit"} {
		if strings.Contains(env, bad) {
			t.Fatal("unsafe inherited environment", bad)
		}
	}
	args := strings.Join(llamaArgs(p, 12345, "/private/key"), " ")
	for _, required := range []string{"--device CUDA0", "--gpu-layers all", "--fit off", "--no-context-shift", "--parallel 1", "--api-key-file /private/key", "--no-mmproj", "--spec-type none", "--no-agent", "--no-webui"} {
		if !strings.Contains(args, required) {
			t.Fatal("missing explicit runtime policy", required)
		}
	}
	if strings.Contains(args, "Bearer") || strings.Contains(args, "--hf-repo") {
		t.Fatal("unsafe native engine arguments")
	}
}

func TestGGUFConfigurePreservesRuntimeWithoutWritingPythonAssets(t *testing.T) {
	m, h, o := ggufPlanFixture(t)
	p, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	dir, e := nativeDir(p.Name)
	if e != nil {
		t.Fatal(e)
	}
	if e = privateDir(dir); e != nil {
		t.Fatal(e)
	}
	cfg := NativeConfig{Schema: nativeSchemaVersion, Plan: p, RuntimeBinary: "runtime/llama-server", RuntimeLibraries: []string{"runtime/core"}, RuntimeFiles: []RuntimeFile{{Path: "runtime/llama-server", SHA256: strings.Repeat("a", 64)}}}
	if e = writeJSONFile(filepath.Join(dir, "config.json"), cfg); e != nil {
		t.Fatal(e)
	}
	listen, key := "0.0.0.0", fixtureAPIKey(t)
	if e = ConfigureNative(context.Background(), p.Name, &listen, &key); e != nil {
		t.Fatal(e)
	}
	got, _, e := nativeConfig(p.Name)
	if e != nil || got.RuntimeBinary != cfg.RuntimeBinary || len(got.RuntimeFiles) != 1 || got.Plan.GGUF.File.SHA256 != p.GGUF.File.SHA256 || got.Plan.Listen != listen {
		t.Fatal("access reconfiguration changed runtime identity", e)
	}
	for _, name := range []string{"venv", "native_entry.py", "api_auth.py"} {
		if _, e = os.Lstat(filepath.Join(dir, name)); !os.IsNotExist(e) {
			t.Fatal("GGUF configure installed Python/changed inference assets")
		}
	}
	gpu := p.GPU
	gpu.FreeMiB = 1
	if e = verifyWeightFit(got.Plan, []GPU{gpu}); e == nil {
		t.Fatal("cached restart ignored fresh free VRAM")
	}
}
