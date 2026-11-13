package modelship

import (
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

// modelCachePath follows huggingface_hub's cache precedence. Persisting this
// absolute path in native state prevents a restart from silently switching caches.
func modelCachePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	path := os.Getenv("HF_HUB_CACHE")
	if path == "" {
		path = os.Getenv("HUGGINGFACE_HUB_CACHE")
	}
	if path == "" {
		path = os.Getenv("HF_HOME")
		if path == "" {
			path = os.Getenv("XDG_CACHE_HOME")
			if path == "" {
				path = filepath.Join(home, ".cache")
			}
			path = filepath.Join(path, "huggingface")
		}
		if path == "~" {
			path = home
		} else if strings.HasPrefix(path, "~/") {
			path = filepath.Join(home, path[2:])
		}
		path = filepath.Join(path, "hub")
	}
	if path == "~" {
		path = home
	} else if strings.HasPrefix(path, "~/") {
		path = filepath.Join(home, path[2:])
	}
	return filepath.Abs(path)
}

// InspectModelCache does not create, delete, or change any cache file. It counts
// only correctly sized files at the exact pinned snapshot, not other revisions.
func InspectModelCache(h *Host, m Model) {
	warn := func(err error) {
		h.Warnings = append(h.Warnings, "Cannot inspect native model-cache disk: "+err.Error()+"; verify capacity manually")
	}
	path, err := modelCachePath()
	if err != nil {
		warn(err)
		return
	}
	h.ModelCacheDir = path
	ancestor := path
	for {
		st, e := os.Stat(ancestor)
		if e == nil {
			if !st.IsDir() {
				warn(fmt.Errorf("cache path ancestor is not a directory"))
				return
			}
			break
		}
		if !os.IsNotExist(e) || filepath.Dir(ancestor) == ancestor {
			warn(fmt.Errorf("cache filesystem is inaccessible"))
			return
		}
		ancestor = filepath.Dir(ancestor)
	}
	var stat syscall.Statfs_t
	if err = syscall.Statfs(ancestor, &stat); err != nil || stat.Bsize <= 0 || stat.Bavail > uint64(math.MaxInt64)/uint64(stat.Bsize) {
		warn(fmt.Errorf("available cache filesystem space could not be determined"))
		return
	}
	h.ModelCacheFreeMiB = int64(stat.Bavail) * int64(stat.Bsize) / (1024 * 1024)
	recipe, known, err := findRecipe(m)
	if err != nil || !known {
		return
	}
	files, _ := weightFiles(m.Hub.Siblings, recipe.Kind != "text")
	if m.GGUF != nil {
		files = nil
		for _, pin := range ggufSelectedArtifacts(m.GGUF) {
			files = append(files, pin.Name)
		}
	}
	selected := map[string]bool{}
	for _, name := range files {
		selected[name] = true
	}
	snapshot := filepath.Join(path, "models--"+strings.ReplaceAll(m.Hub.ID, "/", "--"), "snapshots", m.Hub.SHA)
	h.CachedWeightBytes = 0
	for _, file := range m.Hub.Siblings {
		if !selected[file.Name] || !fs.ValidPath(file.Name) || file.Bytes() <= 0 {
			continue
		}
		path := filepath.Join(snapshot, filepath.FromSlash(file.Name))
		if m.GGUF != nil {
			path = ggufArtifactPath(Plan{Model: m.Hub.ID, Revision: m.Hub.SHA, ModelCacheDir: h.ModelCacheDir, GGUF: m.GGUF}, file.Name)
		}
		st, err := os.Stat(path)
		if err == nil && st.Mode().IsRegular() && st.Size() == file.Bytes() {
			if h.CachedWeightBytes > math.MaxInt64-file.Bytes() {
				warn(fmt.Errorf("cached file size overflow"))
				return
			}
			h.CachedWeightBytes += file.Bytes()
		}
	}
	h.ModelCacheInspected = true
}
