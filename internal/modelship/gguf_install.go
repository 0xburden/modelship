package modelship

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"syscall"
	"time"
)

type RuntimeFile struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256,omitempty"`
	Link   string `json:"link,omitempty"`
}

var llamaArtifacts = []ArtifactPin{
	{Name: "core.tar.gz", Size: 187314621, SHA256: "4af1e1da23661db4a5b84b29aae16ebc053df689c5b8b1427c4670f2c6406d8c", URL: "https://github.com/ggml-org/llama.cpp/releases/download/b12110/llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz"},
	{Name: "cuda.tar.gz", Size: 594371477, SHA256: "6f1d562086d84ae85df880d3711079ec225602b51d530608e5001600625067dd", URL: "https://github.com/ggml-org/llama.cpp/releases/download/b12110/cudart-llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz"},
}

func digestFile(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	hash := sha256.New()
	if _, e = io.Copy(hash, f); e != nil {
		return "", e
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}
func privateRegular(path string) error {
	st, e := os.Lstat(path)
	if e != nil || !st.Mode().IsRegular() || st.Mode().Perm() != 0600 {
		return fmt.Errorf("cached artifact must be a real private regular file")
	}
	return nil
}
func downloadVerified(ctx context.Context, client *http.Client, url, token, path string, pin ArtifactPin) error {
	if e := privateDir(filepath.Dir(path)); e != nil {
		return e
	}
	unlock, e := nativeLock(filepath.Dir(path))
	if e != nil {
		return fmt.Errorf("artifact cache is in use by another installation")
	}
	defer unlock()
	if _, e := os.Lstat(path); e == nil {
		if e = privateRegular(path); e != nil {
			return e
		}
		st, _ := os.Stat(path)
		hash, e := digestFile(path)
		if e != nil || st.Size() != pin.Size || hash != pin.SHA256 {
			return fmt.Errorf("cached artifact differs from its pinned digest; remove the corrupt cache file explicitly")
		}
		return nil
	} else if !os.IsNotExist(e) {
		return fmt.Errorf("cannot inspect artifact cache")
	}
	part := path + ".partial"
	offset := int64(0)
	if st, e := os.Lstat(part); e == nil {
		if !st.Mode().IsRegular() || st.Mode().Perm() != 0600 || st.Size() > pin.Size {
			return fmt.Errorf("invalid partial artifact cache")
		}
		offset = st.Size()
	} else if !os.IsNotExist(e) {
		return fmt.Errorf("cannot inspect partial artifact")
	}
	if offset < pin.Size {
		req, e := http.NewRequestWithContext(ctx, "GET", url, nil)
		if e != nil {
			return fmt.Errorf("invalid artifact URL")
		}
		req.Header.Set("Accept-Encoding", "identity")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		if offset > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		}
		resp, e := client.Do(req)
		if e != nil {
			return fmt.Errorf("artifact download transport failed (partial cache retained)")
		}
		defer resp.Body.Close()
		if resp.Header.Get("Content-Encoding") != "" {
			return fmt.Errorf("encoded artifact response refused")
		}
		switch resp.StatusCode {
		case 200:
			offset = 0
		case 206:
			if offset == 0 || resp.Header.Get("Content-Range") != fmt.Sprintf("bytes %d-%d/%d", offset, pin.Size-1, pin.Size) {
				return fmt.Errorf("invalid artifact resume response")
			}
		default:
			return fmt.Errorf("artifact download HTTP %d", resp.StatusCode)
		}
		if resp.ContentLength >= 0 && resp.ContentLength != pin.Size-offset {
			return fmt.Errorf("artifact response length mismatch")
		}
		fd, e := syscall.Open(part, syscall.O_WRONLY|syscall.O_CREAT|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0600)
		if e != nil {
			return fmt.Errorf("cannot open private partial cache")
		}
		f := os.NewFile(uintptr(fd), part)
		if offset == 0 {
			e = f.Truncate(0)
		} else {
			_, e = f.Seek(offset, io.SeekStart)
		}
		var n int64
		if e == nil {
			n, e = io.Copy(f, io.LimitReader(resp.Body, pin.Size-offset+1))
		}
		if e == nil {
			e = f.Sync()
		}
		closeError := f.Close()
		if e == nil {
			e = closeError
		}
		if e != nil || n != pin.Size-offset {
			return fmt.Errorf("artifact download incomplete/oversized; partial cache retained")
		}
	}
	hash, e := digestFile(part)
	if e != nil || hash != pin.SHA256 {
		return fmt.Errorf("artifact SHA-256 mismatch; refusing to install/load")
	}
	return os.Rename(part, path)
}
func artifactHTTPClient() *http.Client {
	return &http.Client{Timeout: 2 * time.Hour, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) > 10 || req.URL.Scheme != "https" {
			return fmt.Errorf("artifact redirect refused")
		}
		if len(via) > 0 && req.URL.Host != via[0].URL.Host {
			req.Header.Del("Authorization")
		}
		return nil
	}}
}
func safeRelative(path string) bool {
	return path != "" && !filepath.IsAbs(path) && filepath.Clean(path) == path && path != ".." && !strings.HasPrefix(path, ".."+string(os.PathSeparator)) && !strings.ContainsAny(path, "\\\x00")
}
func realParent(root, path string) error {
	rel, e := filepath.Rel(root, filepath.Dir(path))
	if e != nil || !safeRelative(rel) && rel != "." {
		return fmt.Errorf("archive parent escape")
	}
	cur := root
	if rel == "." {
		return nil
	}
	for _, part := range strings.Split(rel, string(os.PathSeparator)) {
		cur = filepath.Join(cur, part)
		if e = os.Mkdir(cur, 0700); e != nil && !os.IsExist(e) {
			return e
		}
		st, e := os.Lstat(cur)
		if e != nil || !st.IsDir() {
			return fmt.Errorf("archive parent is not a real directory")
		}
	}
	return nil
}
func extractRuntime(archive, root string) error {
	if e := privateDir(root); e != nil {
		return e
	}
	root, e := filepath.EvalSymlinks(root)
	if e != nil {
		return e
	}
	f, e := os.Open(archive)
	if e != nil {
		return e
	}
	defer f.Close()
	gz, e := gzip.NewReader(f)
	if e != nil {
		return fmt.Errorf("invalid runtime gzip")
	}
	defer gz.Close()
	reader := tar.NewReader(gz)
	total := int64(0)
	links := map[string]string{}
	for count := 0; ; count++ {
		header, e := reader.Next()
		if e == io.EOF {
			break
		}
		if e != nil || count >= 4096 {
			return fmt.Errorf("invalid/oversized runtime archive")
		}
		name := strings.TrimPrefix(header.Name, "./")
		name = strings.TrimSuffix(name, "/")
		if name == "." || name == "" {
			continue
		}
		if !safeRelative(name) {
			return fmt.Errorf("runtime archive path escape")
		}
		path := filepath.Join(root, name)
		if e = realParent(root, path); e != nil {
			return e
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if e = privateDir(path); e != nil {
				return e
			}
		case tar.TypeReg, tar.TypeRegA:
			if header.Size < 0 || header.Size > 2<<30 || total > 4<<30-header.Size {
				return fmt.Errorf("runtime expansion budget exceeded")
			}
			total += header.Size
			out, e := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
			if e != nil {
				return fmt.Errorf("duplicate/unsafe runtime archive entry")
			}
			_, e = io.CopyN(out, reader, header.Size)
			ce := out.Close()
			if e == nil {
				e = ce
			}
			if e != nil {
				return e
			}
			if header.Mode&0111 != 0 {
				if e = os.Chmod(path, 0700); e != nil {
					return e
				}
			}
		case tar.TypeSymlink:
			if filepath.IsAbs(header.Linkname) || strings.ContainsAny(header.Linkname, "\\\x00") {
				return fmt.Errorf("runtime symlink escape")
			}
			target := filepath.Clean(filepath.Join(filepath.Dir(name), header.Linkname))
			if !safeRelative(target) {
				return fmt.Errorf("runtime symlink escape")
			}
			if _, exists := links[name]; exists {
				return fmt.Errorf("duplicate runtime symlink")
			}
			links[name] = header.Linkname
		default:
			return fmt.Errorf("runtime special/hardlink entry refused")
		}
	}
	// Create links only after regular files. Each target must resolve inside the
	// extracted private tree; regular-file writes can never traverse an archive link.
	for tries := 0; len(links) > 0 && tries < 16; tries++ {
		for name, target := range links {
			path := filepath.Join(root, name)
			resolved, e := filepath.EvalSymlinks(filepath.Join(filepath.Dir(path), target))
			if e != nil {
				continue
			}
			rel, e := filepath.Rel(root, resolved)
			if e != nil || !safeRelative(rel) {
				return fmt.Errorf("runtime symlink target escape")
			}
			if e = os.Symlink(target, path); e != nil {
				return fmt.Errorf("unsafe/duplicate runtime link")
			}
			delete(links, name)
		}
	}
	if len(links) > 0 {
		return fmt.Errorf("unresolved runtime links")
	}
	return nil
}
func installedRuntime(dir string) (string, []string, []RuntimeFile, error) {
	binary := ""
	libraries := map[string]bool{}
	files := []RuntimeFile{}
	err := filepath.WalkDir(filepath.Join(dir, "runtime"), func(path string, entry os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if entry.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(dir, path)
		if entry.Type()&os.ModeSymlink != 0 {
			target, e := os.Readlink(path)
			if e != nil {
				return e
			}
			if _, e = fileWithin(filepath.Join(dir, "runtime"), strings.TrimPrefix(rel, "runtime/")); e != nil {
				return e
			}
			files = append(files, RuntimeFile{Path: rel, Link: target})
			return nil
		}
		if !entry.Type().IsRegular() {
			return fmt.Errorf("invalid runtime file type")
		}
		hash, e := digestFile(path)
		if e != nil {
			return e
		}
		files = append(files, RuntimeFile{Path: rel, SHA256: hash})
		if entry.Name() == "llama-server" {
			if binary != "" {
				return fmt.Errorf("ambiguous llama-server")
			}
			binary = rel
		}
		if strings.Contains(entry.Name(), ".so") {
			parent, _ := filepath.Rel(dir, filepath.Dir(path))
			libraries[parent] = true
		}
		return nil
	})
	if err != nil || binary == "" {
		return "", nil, nil, fmt.Errorf("runtime bundle has no unique server")
	}
	dirs := []string{}
	for path := range libraries {
		dirs = append(dirs, path)
	}
	sort.Strings(dirs)
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return binary, dirs, files, nil
}
func fileWithin(root, path string) (string, error) {
	if !safeRelative(path) {
		return "", fmt.Errorf("invalid saved runtime path")
	}
	root, e := filepath.EvalSymlinks(root)
	if e != nil {
		return "", fmt.Errorf("runtime root missing")
	}
	resolved, e := filepath.EvalSymlinks(filepath.Join(root, path))
	if e != nil {
		return "", fmt.Errorf("runtime file missing")
	}
	rel, e := filepath.Rel(root, resolved)
	if e != nil || !safeRelative(rel) {
		return "", fmt.Errorf("runtime path escapes deployment")
	}
	return resolved, nil
}
func verifyGGUFInstall(cfg NativeConfig, dir string) error {
	if e := validateGGUFPlan(cfg.Plan); e != nil {
		return e
	}
	if len(cfg.RuntimeFiles) == 0 || len(cfg.RuntimeFiles) > 4096 {
		return fmt.Errorf("runtime file manifest missing")
	}
	binary, libs, files, e := installedRuntime(dir)
	if e != nil || binary != cfg.RuntimeBinary || !reflect.DeepEqual(libs, cfg.RuntimeLibraries) || !reflect.DeepEqual(files, cfg.RuntimeFiles) {
		return fmt.Errorf("installed runtime files, links or digest manifest differ from saved installation")
	}
	for _, pin := range cfg.RuntimeFiles {
		if !strings.HasPrefix(pin.Path, "runtime/") {
			return fmt.Errorf("invalid runtime manifest path")
		}
	}
	path := ggufModelPath(cfg.Plan)
	if e := privateRegular(path); e != nil {
		return e
	}
	st, e := os.Stat(path)
	if e != nil || st.Size() != cfg.Plan.GGUF.File.Size {
		return fmt.Errorf("cached GGUF size mismatch")
	}
	hash, e := digestFile(path)
	if e != nil || hash != cfg.Plan.GGUF.File.SHA256 {
		return fmt.Errorf("cached GGUF digest mismatch; refusing startup")
	}
	f, e := os.Open(path)
	if e != nil {
		return fmt.Errorf("cannot verify GGUF header")
	}
	defer f.Close()
	meta, e := parseGGUF(st.Size(), func(offset, n int64) ([]byte, error) { b := make([]byte, n); _, e := f.ReadAt(b, offset); return b, e })
	if e != nil || !reflect.DeepEqual(meta, cfg.Plan.GGUF.Metadata) {
		return fmt.Errorf("cached GGUF header differs from saved planning metadata")
	}
	return nil
}
func deployGGUF(ctx context.Context, p Plan, wait time.Duration) error {
	if e := validateGGUFPlan(p); e != nil {
		return e
	}
	if e := validateNativeAccess(p, true); e != nil {
		return e
	}
	if len(p.Blockers) > 0 {
		return fmt.Errorf("GGUF plan is blocked")
	}
	if e := verifyWeightFitNow(ctx, p); e != nil {
		return e
	}
	if e := checkLlamaRuntimeABI(); e != nil {
		return e
	}
	dir, e := nativeDir(p.Name)
	if e != nil {
		return e
	}
	if e = privateDir(filepath.Dir(dir)); e != nil {
		return e
	}
	if e = os.Mkdir(dir, 0700); e != nil {
		return fmt.Errorf("deployment already exists or cannot be created; no existing deployment changed")
	}
	unlock, e := nativeLock(dir)
	if e != nil {
		return e
	}
	defer unlock()
	cfg := NativeConfig{Schema: nativeSchemaVersion, Plan: p}
	if e = writeJSONFile(filepath.Join(dir, "config.json"), cfg); e != nil {
		return e
	}
	// Runtime/temporary extraction lives in deployment state, which may be on a
	// different filesystem from the model cache. Check both before downloading.
	var fs syscall.Statfs_t
	if e = syscall.Statfs(dir, &fs); e != nil || fs.Bsize <= 0 || fs.Bavail < uint64((12<<30)/int64(fs.Bsize)) {
		return fmt.Errorf("GGUF runtime state filesystem needs at least 12 GiB free")
	}
	stateFS := fs
	cacheDir := filepath.Dir(ggufModelPath(p))
	for _, path := range []string{filepath.Join(p.ModelCacheDir, "modelship-gguf"), filepath.Dir(cacheDir), cacheDir} {
		if e = privateDir(path); e != nil {
			return e
		}
	}
	missing := p.GGUF.File.Size
	if st, e := os.Stat(ggufModelPath(p)); e == nil && st.Size() == p.GGUF.File.Size {
		missing = 0
	}
	if e = syscall.Statfs(cacheDir, &fs); e != nil || fs.Bsize <= 0 {
		return fmt.Errorf("cannot inspect GGUF cache filesystem")
	}
	headroom := int64(4 << 30)
	if fs.Fsid == stateFS.Fsid {
		headroom = 12 << 30
	}
	if fs.Bavail < uint64((missing+missing/20+headroom+int64(fs.Bsize)-1)/int64(fs.Bsize)) {
		return fmt.Errorf("GGUF cache disk no longer fits the selected missing artifact plus headroom")
	}
	client := artifactHTTPClient()
	defer client.CloseIdleConnections()
	for i, pin := range llamaArtifacts {
		archive := filepath.Join(dir, "downloads", pin.Name)
		fmt.Printf("Downloading verified llama.cpp runtime component %d/%d (no host package changes).\n", i+1, len(llamaArtifacts))
		if e = downloadVerified(ctx, client, pin.URL, "", archive, pin); e != nil {
			return e
		}
		if e = extractRuntime(archive, filepath.Join(dir, "runtime", strings.TrimSuffix(pin.Name, ".tar.gz"))); e != nil {
			return e
		}
	}
	cfg.RuntimeBinary, cfg.RuntimeLibraries, cfg.RuntimeFiles, e = installedRuntime(dir)
	if e != nil {
		return e
	}
	if e = writeJSONFile(filepath.Join(dir, "config.json"), cfg); e != nil {
		return e
	}
	if e = probeLlamaCUDA(ctx, cfg, dir); e != nil {
		return e
	}
	if e = verifyWeightFitNow(ctx, p); e != nil {
		return e
	}
	fmt.Printf("Downloading only selected pinned GGUF (%d bytes); auxiliary/projector files are excluded.\n", p.GGUF.File.Size)
	url := "https://huggingface.co/" + p.Model + "/resolve/" + p.Revision + "/" + escapeWeightPath(p.GGUF.File.Name)
	if e = downloadVerified(ctx, client, url, os.Getenv("HF_TOKEN"), ggufModelPath(p), p.GGUF.File); e != nil {
		return e
	}
	if e = writeJSONFile(filepath.Join(dir, "config.json"), cfg); e != nil {
		return e
	}
	return startNative(ctx, cfg, dir, wait)
}
