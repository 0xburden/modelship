package modelship

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLaunchWeightCheckUsesCurrentMemoryAndSameGPUIdentity(t *testing.T) {
	p := Plan{WeightResidentBytes: 8*1024*1024 + 1, GPU: GPU{UUID: "GPU-selected"}}
	for _, test := range []struct {
		name      string
		gpus      []GPU
		errorText string
	}{
		{"fits", []GPU{{UUID: "GPU-selected", TotalMiB: 16, FreeMiB: 9}}, ""},
		{"free-changed", []GPU{{UUID: "GPU-selected", TotalMiB: 16, FreeMiB: 8}}, "currently free"},
		{"total-changed", []GPU{{UUID: "GPU-selected", TotalMiB: 8, FreeMiB: 8}}, "total VRAM"},
		{"different-device", []GPU{{UUID: "GPU-other", TotalMiB: 160, FreeMiB: 160}}, "identity"},
		{"missing-device", nil, "identity"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := verifyWeightFit(p, test.gpus)
			if test.errorText == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.errorText) {
				t.Fatal(err)
			}
		})
	}
	if err := verifyWeightFit(Plan{}, nil); err != nil {
		t.Fatal("legacy state no longer readable", err)
	}
}

func fakeNvidiaSMI(t *testing.T, row string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "nvidia-smi"), []byte("#!/bin/sh\nprintf '%s\\n' '"+row+"'\n"), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
}
