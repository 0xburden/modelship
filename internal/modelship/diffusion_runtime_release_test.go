package modelship

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestFetchDiffusionRuntime(t *testing.T) {
	worker, probe := []byte("tested worker"), []byte("tested probe")
	pin := func(name string, b []byte) ArtifactPin {
		return ArtifactPin{Name: name, Size: int64(len(b)), SHA256: fmt.Sprintf("%x", sha256.Sum256(b))}
	}
	pins := [2]ArtifactPin{pin("worker", worker), pin("probe", probe)}
	for _, tc := range []struct {
		name, entry               string
		kind                      byte
		extra, corrupt, wrongHash bool
	}{
		{name: "verified"},
		{name: "archive digest", wrongHash: true},
		{name: "member digest", corrupt: true},
		{name: "extra member", extra: true},
		{name: "path escape", entry: "../worker"},
		{name: "symbolic link", kind: tar.TypeSymlink},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var b bytes.Buffer
			gz := gzip.NewWriter(&b)
			tw := tar.NewWriter(gz)
			add := func(name string, kind byte, data []byte) {
				h := tar.Header{Name: name, Typeflag: kind, Mode: 0777, Size: int64(len(data)), Format: tar.FormatUSTAR}
				if kind == tar.TypeSymlink {
					h.Linkname = "/bin/sh"
					h.Size = 0
					data = nil
				}
				if err := tw.WriteHeader(&h); err != nil {
					t.Fatal(err)
				}
				if _, err := tw.Write(data); err != nil {
					t.Fatal(err)
				}
			}
			entry, kind := tc.entry, tc.kind
			if entry == "" {
				entry = pins[0].Name
			}
			if kind == 0 {
				kind = tar.TypeReg
			}
			payload := worker
			if tc.corrupt {
				payload = bytes.Repeat([]byte{'?'}, len(worker))
			}
			add(diffusionRuntimeNotice, tar.TypeReg, []byte("fixture notice"))
			add(entry, kind, payload)
			add(pins[1].Name, tar.TypeReg, probe)
			if tc.extra {
				add("other", tar.TypeReg, []byte("unreviewed"))
			}
			if err := tw.Close(); err != nil {
				t.Fatal(err)
			}
			// As with Python's USTAR writer, permit only bounded zero padding.
			if _, err := gz.Write(make([]byte, 1024)); err != nil {
				t.Fatal(err)
			}
			if err := gz.Close(); err != nil {
				t.Fatal(err)
			}
			a := pin("runtime.tar.gz", b.Bytes())
			if tc.wrongHash {
				a.SHA256 = pins[0].SHA256
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != "GET" || r.URL.Path != "/"+a.Name || r.Header.Get("Authorization") != "" {
					http.Error(w, "invalid request", 400)
					return
				}
				_, _ = w.Write(b.Bytes())
			}))
			defer server.Close()
			a.URL = server.URL + "/" + a.Name
			dir := t.TempDir()
			err := fetchDiffusionRuntime(context.Background(), server.Client(), dir, a, pins)
			entries, readErr := os.ReadDir(dir)
			if readErr != nil {
				t.Fatal(readErr)
			}
			if tc.name != "verified" {
				if err == nil || len(entries) != 0 {
					t.Fatalf("unverified bytes escaped staging: %v %v", err, entries)
				}
				return
			}
			if err != nil || len(entries) != 3 {
				t.Fatalf("install failed: %v %v", err, entries)
			}
			for _, p := range pins {
				if err := diffusionCheckFile(context.Background(), dir, p, 0500, io.Discard); err != nil {
					t.Fatal(err)
				}
			}
			st, err := os.Lstat(filepath.Join(dir, diffusionRuntimeNotice))
			if err != nil || st.Mode() != 0400 {
				t.Fatal("notice should be private and non-executable")
			}
		})
	}
}

// Optional release QA sends the actual staged bundles over HTTP through the
// production downloader and extractor, without a CUDA device or model cache.
func TestDiffusionReleaseArchives(t *testing.T) {
	for _, task := range []string{diffusionTask, diffusionEditTask} {
		asset := diffusionArchivePin(task)
		if asset.Size <= 0 || len(asset.SHA256) != 64 || asset.URL != diffusionRuntimeRelease+asset.Name {
			t.Fatalf("invalid release pin for %s", task)
		}
	}
	root := os.Getenv("MODELSHIP_DIFFUSION_RELEASE_TEST_DIR")
	if root == "" {
		t.Skip("set MODELSHIP_DIFFUSION_RELEASE_TEST_DIR for release archive QA")
	}
	for _, task := range []string{diffusionTask, diffusionEditTask} {
		t.Run(task, func(t *testing.T) {
			asset := diffusionArchivePin(task)
			pins, err := diffusionRuntimePins(task)
			if err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/"+asset.Name || r.Header.Get("Authorization") != "" {
					http.NotFound(w, r)
					return
				}
				http.ServeFile(w, r, filepath.Join(root, asset.Name))
			}))
			defer server.Close()
			download := asset
			download.URL = server.URL + "/" + asset.Name
			dir := t.TempDir()
			if err := fetchDiffusionRuntime(context.Background(), server.Client(), dir, download, pins); err != nil {
				t.Fatal(err)
			}
			for _, p := range pins {
				if err := diffusionCheckFile(context.Background(), dir, p, 0500, io.Discard); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
