package modelship

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/xml"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

const diffusionSchemaVersion = 12
const diffusionStartupTimeout = 20 * time.Minute
const diffusionRequestTimeout = 15 * time.Minute
const diffusionStateHeadroom = int64(4 << 30)

// Exact ELF pair from the build used for the integrated generation GPU test.
var diffusionWorkerPin = ArtifactPin{Name: "modelship-sd-worker", Size: 115618872, SHA256: "5e1e684468576b74040a7c201eff289452511136f34477c9f9fbdcfafdbba1e1"}
var diffusionProbePin = ArtifactPin{Name: "modelship-sd-probe", Size: 72735736, SHA256: "92ff72e75d53e502c0e5dffd1cf618201c28eb34a229b44775134d87b385c134"}

// Protocol-3 pair from CUDA build 44470677936 at 44b403789a9185d5d8bd61bac101574828db862d.
// This exact pair passed the integrated edit GPU test.
var diffusionEditWorkerPin = ArtifactPin{Name: "modelship-sd-edit-worker", Size: 115614808, SHA256: "519b0f987681f72da95d75df2d40e801cb23490b5a840d417510399fedabdaa2"}
var diffusionEditProbePin = ArtifactPin{Name: "modelship-sd-probe", Size: 72731640, SHA256: "e5f7357f4e920d336d04215c71cd605e37459de62f23ae5735a5724701352c3e"}

func diffusionRuntimePins(task string) ([2]ArtifactPin, error) {
	switch task {
	case diffusionTask:
		return [2]ArtifactPin{diffusionWorkerPin, diffusionProbePin}, nil
	case diffusionEditTask:
		for _, pin := range [2]ArtifactPin{diffusionEditWorkerPin, diffusionEditProbePin} {
			digest, err := hex.DecodeString(pin.SHA256)
			if pin.Size <= 0 || err != nil || len(digest) != sha256.Size || hex.EncodeToString(digest) != pin.SHA256 {
				return [2]ArtifactPin{}, fmt.Errorf("GGUF edit runtime has not been verified and pinned")
			}
		}
		return [2]ArtifactPin{diffusionEditWorkerPin, diffusionEditProbePin}, nil
	default:
		return [2]ArtifactPin{}, fmt.Errorf("invalid reviewed GGUF diffusion task")
	}
}

type diffusionWheel struct {
	ArtifactPin
	members []string
}

func diffusionWheels() [2]diffusionWheel {
	return [2]diffusionWheel{
		{ArtifactPin{Name: "cuda-runtime.whl", URL: "https://files.pythonhosted.org/packages/0d/9b/a997b638fcd068ad6e4d53b8551a7d30fe8b404d6f1804abf1df69838932/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", Size: 954765, SHA256: "adade8dcbd0edf427b7204d480d6066d33902cab2a4707dcfc48a2d0fd44ab90"}, []string{"nvidia/cuda_runtime/lib/libcudart.so.12"}},
		{ArtifactPin{Name: "cublas.whl", URL: "https://files.pythonhosted.org/packages/dc/61/e24b560ab2e2eaeb3c839129175fb330dfcfc29e5203196e5541a4c44682/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl", Size: 594346921, SHA256: "8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142"}, []string{"nvidia/cublas/lib/libcublas.so.12", "nvidia/cublas/lib/libcublasLt.so.12"}},
	}
}

func diffusionDir(path string) error {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil || !filepath.IsAbs(path) || resolved != path {
		return fmt.Errorf("diffusion requires canonical private directories without symlinks")
	}
	st, err := os.Lstat(path)
	if err != nil || !st.IsDir() || st.Mode().Perm() != 0700 || int(st.Sys().(*syscall.Stat_t).Uid) != os.Geteuid() {
		return fmt.Errorf("diffusion requires an owned 0700 directory")
	}
	return diffusionProtectedAncestors(filepath.Dir(path))
}

// Permit root's sticky /tmp only when each descendant is itself owned by
// root or this uid. Untrusted accounts must not be able to move a verified
// directory between verification and directory-descriptor acquisition.
func diffusionProtectedAncestors(path string) error {
	for {
		st, err := os.Lstat(path)
		if err != nil || !st.IsDir() {
			return fmt.Errorf("diffusion requires real, protected directory ancestors")
		}
		owner := int(st.Sys().(*syscall.Stat_t).Uid)
		if (owner != 0 && owner != os.Geteuid()) || (st.Mode().Perm()&0022 != 0 && (owner != 0 || st.Mode()&os.ModeSticky == 0)) {
			return fmt.Errorf("diffusion directory ancestor is writable by other users")
		}
		parent := filepath.Dir(path)
		if parent == path {
			return nil
		}
		path = parent
	}
}

func diffusionOpen(path string, perm os.FileMode, size int64) (*os.File, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("cannot open pinned diffusion artifact")
	}
	f := os.NewFile(uintptr(fd), path)
	st, err := f.Stat()
	if err != nil || st.Mode() != perm || st.Size() != size || int(st.Sys().(*syscall.Stat_t).Uid) != os.Geteuid() || st.Sys().(*syscall.Stat_t).Nlink != 1 {
		f.Close()
		return nil, fmt.Errorf("diffusion artifact has wrong type, owner, permissions, size or link count")
	}
	return f, nil
}

type diffusionContextReader struct {
	context.Context
	io.Reader
}

func (r diffusionContextReader) Read(b []byte) (int, error) {
	if err := r.Context.Err(); err != nil {
		return 0, err
	}
	return r.Reader.Read(b)
}

func diffusionDigest(ctx context.Context, src io.Reader, dst io.Writer, size int64) (string, error) {
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(dst, h), io.LimitReader(diffusionContextReader{ctx, src}, size+1))
	if err != nil || n != size {
		return "", fmt.Errorf("could not read or copy the exact diffusion artifact bytes")
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func diffusionVerifiedFile(ctx context.Context, dir string, pin ArtifactPin, perm os.FileMode, dst io.Writer) (*os.File, error) {
	f, err := diffusionOpen(filepath.Join(dir, pin.Name), perm, pin.Size)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", pin.Name, err)
	}
	got, err := diffusionDigest(ctx, f, dst, pin.Size)
	if err != nil || pin.SHA256 != got {
		f.Close()
		return nil, fmt.Errorf("%s: pinned diffusion artifact verification failed", pin.Name)
	}
	return f, nil
}

func diffusionCheckFile(ctx context.Context, dir string, pin ArtifactPin, perm os.FileMode, dst io.Writer) error {
	f, err := diffusionVerifiedFile(ctx, dir, pin, perm, dst)
	if err != nil {
		return err
	}
	return f.Close()
}

func diffusionCopyBinary(ctx context.Context, src, dst string, pin ArtifactPin) error {
	f, err := os.OpenFile(filepath.Join(dst, pin.Name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0500)
	if err != nil {
		return fmt.Errorf("could not stage pinned diffusion binary")
	}
	verify := diffusionCheckFile(ctx, src, pin, 0500, f)
	finish := f.Close()
	if verify != nil {
		return verify
	}
	return finish
}

// Operate only on the explicitly reviewed SONAME members of the verified
// upstream wheels. Keep the wheels (including their license files) intact in
// private state. No pip, arbitrary extraction, or repackaging of NVIDIA libs.
func diffusionWheelLibraries(ctx context.Context, inputs, libs string, pin diffusionWheel, install bool) error {
	f, err := diffusionOpen(filepath.Join(inputs, pin.Name), 0600, pin.Size)
	if err != nil {
		return err
	}
	defer f.Close()
	if hash, err := diffusionDigest(ctx, f, io.Discard, pin.Size); err != nil || hash != pin.SHA256 {
		return fmt.Errorf("diffusion CUDA dependency digest mismatch")
	}
	z, err := zip.NewReader(f, pin.Size) // ReadAt on the same verified descriptor.
	if err != nil {
		return fmt.Errorf("invalid pinned CUDA dependency archive")
	}
	for _, name := range pin.members {
		var found *zip.File
		for _, entry := range z.File {
			if entry.Name == name {
				if found != nil || !entry.Mode().IsRegular() || entry.UncompressedSize64 == 0 || entry.UncompressedSize64 > 1<<30 {
					return fmt.Errorf("invalid pinned CUDA library archive member")
				}
				found = entry
			}
		}
		if found == nil {
			return fmt.Errorf("missing pinned CUDA library archive member")
		}
		if err := diffusionWheelLibrary(ctx, libs, found, install); err != nil {
			return err
		}
	}
	return nil
}

func diffusionWheelLibrary(ctx context.Context, dir string, entry *zip.File, install bool) error {
	r, err := entry.Open()
	if err != nil {
		return err
	}
	defer r.Close()
	size := int64(entry.UncompressedSize64)
	name := filepath.Base(entry.Name)
	if install {
		f, err := os.OpenFile(filepath.Join(dir, name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0400)
		if err != nil {
			return err
		}
		_, copyErr := diffusionDigest(ctx, r, f, size)
		closeErr := f.Close()
		if copyErr != nil {
			return copyErr
		}
		return closeErr
	}
	want, err := diffusionDigest(ctx, r, io.Discard, size)
	if err != nil {
		return err
	}
	return diffusionCheckFile(ctx, dir, ArtifactPin{Name: name, Size: size, SHA256: want}, 0400, io.Discard)
}

func diffusionDisk(dir string, need int64) (syscall.Statfs_t, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil || st.Bsize <= 0 || st.Bavail < uint64((need+int64(st.Bsize)-1)/int64(st.Bsize)) {
		return st, fmt.Errorf("diffusion artifact filesystem requires at least %d MiB free for verified files and headroom", (need+(1<<20)-1)>>20)
	}
	return st, nil
}

func diffusionCache(p Plan) string {
	return filepath.Join(p.ModelCacheDir, "modelship-diffusion", p.Diffusion)
}

// Scope the same-account flock to the same location as the one-shot harness.
// Close (not LOCK_UN) after the child is reaped; inherited descriptors retain
// ownership if the supervisor dies. This does not reserve the GPU system-wide.
func lockDiffusionGPU(uuid string) (*os.File, error) {
	if !diffusionGPUUUID.MatchString(uuid) {
		return nil, fmt.Errorf("invalid physical GPU UUID")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	home, err = filepath.EvalSymlinks(home)
	if err != nil {
		return nil, err
	}
	root := filepath.Join(home, ".modelship-diffusion-smoke")
	if err := privateDir(root); err != nil {
		return nil, err
	}
	if err := diffusionDir(root); err != nil {
		return nil, err
	}
	return lockDiffusionFile(root, uuid+".lock")
}

func lockDiffusionFile(root, name string) (*os.File, error) {
	fd, err := syscall.Open(filepath.Join(root, name), syscall.O_RDWR|syscall.O_CREAT|syscall.O_NOFOLLOW|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0600)
	if err != nil {
		return nil, fmt.Errorf("could not open GPU lock")
	}
	f := os.NewFile(uintptr(fd), name)
	st, err := f.Stat()
	if err != nil || st.Mode() != 0600 || int(st.Sys().(*syscall.Stat_t).Uid) != os.Geteuid() || st.Sys().(*syscall.Stat_t).Nlink != 1 {
		f.Close()
		return nil, fmt.Errorf("unsafe GPU lock")
	}
	if syscall.Flock(fd, syscall.LOCK_EX|syscall.LOCK_NB) != nil {
		f.Close()
		return nil, fmt.Errorf("diffusion lock is held by a Modelship worker")
	}
	return f, nil
}

func admitDiffusionGPU(ctx context.Context, p Plan) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	if !diffusionGPUUUID.MatchString(p.GPU.UUID) {
		return fmt.Errorf("invalid physical GPU UUID")
	}
	smi := func(query string) ([]byte, error) {
		cmd := exec.CommandContext(ctx, "/usr/bin/nvidia-smi", "--id="+p.GPU.UUID, "--format=csv,noheader,nounits", query)
		cmd.Env = []string{"PATH=/usr/bin:/bin", "LC_ALL=C"}
		return cmd.Output()
	}
	rows, err := smi("--query-gpu=index,uuid,name,memory.total,memory.free,driver_version,compute_cap")
	gpus, parseErr := ParseGPUs(string(rows))
	if err != nil || parseErr != nil || len(gpus) != 1 || gpus[0].UUID != p.GPU.UUID {
		return fmt.Errorf("could not refresh selected physical GPU")
	}
	if err := checkDiffusionGPU(gpus[0]); err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, "/usr/bin/nvidia-smi", "-i", p.GPU.UUID, "-q", "-x")
	cmd.Env = []string{"PATH=/usr/bin:/bin", "LC_ALL=C"}
	var inventory boundedDiagnostic
	cmd.Stdout = &inventory
	cmd.WaitDelay = time.Second
	if err := cmd.Run(); err != nil || inventory.truncated || !diffusionIdleInventory(inventory.data.Bytes(), p.GPU.UUID) {
		return fmt.Errorf("GGUF diffusion requires a verified idle GPU; graphics or compute processes are active or inventory is unavailable")
	}
	return nil
}

func diffusionIdleInventory(data []byte, uuid string) bool {
	var inventory struct {
		XMLName xml.Name `xml:"nvidia_smi_log"`
		GPUs    []struct {
			UUID      string `xml:"uuid"`
			Processes []struct {
				Inner string `xml:",innerxml"`
			} `xml:"processes"`
		} `xml:"gpu"`
	}
	return xml.Unmarshal(data, &inventory) == nil && len(inventory.GPUs) == 1 &&
		strings.TrimSpace(inventory.GPUs[0].UUID) == uuid && len(inventory.GPUs[0].Processes) == 1 &&
		strings.TrimSpace(inventory.GPUs[0].Processes[0].Inner) == ""
}

func diffusionEnv(dir, uuid, libs string) []string {
	return []string{"PATH=/usr/bin:/bin", "HOME=" + dir, "TMPDIR=" + dir, "LC_ALL=C", "CUDA_VISIBLE_DEVICES=" + uuid, "LD_LIBRARY_PATH=" + libs}
}

func diffusionDirectoryFD(path string) (*os.File, error) {
	if err := diffusionDir(path); err != nil {
		return nil, err
	}
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_DIRECTORY|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("cannot anchor verified diffusion directory")
	}
	return os.NewFile(uintptr(fd), path), nil
}

func diffusionFD(fd uintptr) string { return fmt.Sprintf("/proc/self/fd/%d", fd) }

func diffusionCacheRemaining(cache string, pin ArtifactPin) int64 {
	present, err := os.Lstat(filepath.Join(cache, pin.Name))
	if err == nil && present.Mode() == 0600 && present.Size() == pin.Size {
		return 0 // The downloader verifies digest before accepting a cache hit.
	}
	part, err := os.Lstat(filepath.Join(cache, pin.Name+".partial"))
	if err == nil && part.Mode() == 0600 && part.Size() <= pin.Size {
		return pin.Size - part.Size()
	}
	return pin.Size
}

func probeDiffusionCUDA(ctx context.Context, dir, uuid string, pin ArtifactPin, libs, lock *os.File) error {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	binary, err := diffusionVerifiedFile(ctx, dir, pin, 0500, io.Discard)
	if err != nil {
		return err
	}
	defer binary.Close()
	cmd := exec.CommandContext(ctx, diffusionFD(3), "--cuda-check")
	cmd.ExtraFiles = []*os.File{binary, libs, lock}
	cmd.Env, cmd.Dir = diffusionEnv(dir, uuid, diffusionFD(4)), dir
	release := guardDiffusionChild(cmd)
	defer release()
	var out boundedDiagnostic
	cmd.Stdout = &out
	cmd.WaitDelay = time.Second
	if err := cmd.Run(); err != nil || out.truncated || !diffusionProbePassed(out.data.String()) {
		return fmt.Errorf("pinned diffusion CUDA0 allocation preflight failed; no weights loaded or host changes attempted")
	}
	return nil
}

func diffusionProbePassed(output string) bool {
	for _, line := range strings.Split(output, "\n") {
		if strings.TrimSpace(line) == "MODELSHIP_SD_CUDA0_ALLOCATED" {
			return true
		}
	}
	return false
}

func diffusionHostABI() error {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		return fmt.Errorf("diffusion requires Linux amd64")
	}
	have, err := inspectRuntimeABI()
	if err != nil {
		return err
	}
	if compareABIVersion(have.GLIBC, "2.35") < 0 || compareABIVersion(have.GLIBCXX, "3.4.30") < 0 {
		return fmt.Errorf("diffusion requires GLIBC_2.35 and GLIBCXX_3.4.30 or newer; host libraries remain unchanged")
	}
	return nil
}

func deployDiffusion(ctx context.Context, p Plan, source string, wait time.Duration) error {
	if err := validateDiffusionPlan(p); err != nil {
		return err
	}
	pins, err := diffusionRuntimePins(p.Task)
	if err != nil {
		return err
	}
	if err := validateNativeAccess(p, true); err != nil {
		return err
	}
	if source != "" {
		if err := diffusionDir(source); err != nil {
			return err
		}
	}
	if err := diffusionHostABI(); err != nil {
		return err
	}
	if err := admitDiffusionGPU(ctx, p); err != nil {
		return err
	}
	dir, err := nativeDir(p.Name)
	if err != nil {
		return err
	}
	if err = privateDir(filepath.Dir(dir)); err != nil {
		return err
	}
	if err = os.Mkdir(dir, 0700); err != nil {
		return fmt.Errorf("deployment already exists or cannot be created; existing deployments unchanged")
	}
	retain := false
	defer func() {
		if !retain {
			_ = os.RemoveAll(dir)
		}
	}()
	unlock, err := nativeLock(dir)
	if err != nil {
		return err
	}
	defer unlock()
	if err := diffusionDir(dir); err != nil {
		return err
	}
	stateFS, err := diffusionDisk(dir, diffusionStateHeadroom)
	if err != nil {
		return err
	}
	cache := diffusionCache(p)
	for _, path := range []string{filepath.Dir(cache), cache} {
		if err := privateDir(path); err != nil {
			return err
		}
		if err := diffusionDir(path); err != nil {
			return err
		}
	}
	missing := int64(0)
	for _, f := range diffusionArtifacts(p.Task) {
		missing += diffusionCacheRemaining(cache, f.ArtifactPin)
	}
	cacheFS, err := diffusionDisk(cache, missing+(1<<30))
	if err != nil {
		return err
	}
	if cacheFS.Fsid == stateFS.Fsid {
		if _, err := diffusionDisk(cache, missing+(1<<30)+diffusionStateHeadroom); err != nil {
			return err
		}
	}
	client := artifactHTTPClient()
	defer client.CloseIdleConnections()
	if source == "" {
		asset := diffusionArchivePin(p.Task)
		fmt.Printf("Downloading verified CUDA runtime %s (%d bytes).\n", asset.Name, asset.Size)
		if err := fetchDiffusionRuntime(ctx, client, dir, asset, pins); err != nil {
			return err
		}
	} else {
		for _, pin := range pins {
			if err := diffusionCopyBinary(ctx, source, dir, pin); err != nil {
				return err
			}
		}
	}
	inputs, libs := filepath.Join(dir, "cuda-inputs"), filepath.Join(dir, "cuda")
	if err := os.Mkdir(libs, 0700); err != nil {
		return err
	}
	for _, wheel := range diffusionWheels() {
		if err := downloadVerified(ctx, client, wheel.URL, "", filepath.Join(inputs, wheel.Name), wheel.ArtifactPin); err != nil {
			return err
		}
		if err := diffusionWheelLibraries(ctx, inputs, libs, wheel, true); err != nil {
			return err
		}
	}
	libsFD, err := diffusionDirectoryFD(libs)
	if err != nil {
		return err
	}
	defer libsFD.Close()
	lock, err := lockDiffusionGPU(p.GPU.UUID)
	if err != nil {
		return err
	}
	err = admitDiffusionGPU(ctx, p)
	if err == nil {
		err = probeDiffusionCUDA(ctx, dir, p.GPU.UUID, pins[1], libsFD, lock)
	}
	lock.Close()
	if err != nil {
		return err
	}
	for _, f := range diffusionArtifacts(p.Task) {
		fmt.Printf("Verifying or downloading %s @ %s / %s (%d bytes).\n", f.Repo, f.Revision, f.RemotePath, f.Size)
		url := "https://huggingface.co/" + f.Repo + "/resolve/" + f.Revision + "/" + escapeWeightPath(f.RemotePath)
		if err := downloadVerified(ctx, client, url, "", filepath.Join(cache, f.Name), f.ArtifactPin); err != nil {
			return err
		}
	}
	cfg := NativeConfig{Schema: diffusionSchemaVersion, Plan: p}
	if err := writeJSONFile(filepath.Join(dir, "config.json"), cfg); err != nil {
		return err
	}
	retain = true
	return startNative(ctx, cfg, dir, wait)
}

// Accept only the detached native supervisor whose PID/run identity was
// recorded by the parent. In particular do not launch a child if writing the
// run record fails and the parent has to discard the new supervisor.
func waitDiffusionIdentity(ctx context.Context, dir, runID string) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	for {
		if run, err := loadNativeRun(dir); err == nil && run.RunID == runID && run.PID == os.Getpid() && nativeAlive(dir, run) {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("could not confirm recorded diffusion supervisor identity")
		case <-ticker.C:
		}
	}
}

func runDiffusionNative(ctx context.Context, cfg NativeConfig, dir, runID string) error {
	p := cfg.Plan
	if err := validateDiffusionPlan(p); err != nil {
		return err
	}
	if err := diffusionHostABI(); err != nil {
		return err
	}
	pins, err := diffusionRuntimePins(p.Task)
	if err != nil {
		return err
	}
	cache, inputs, libs := diffusionCache(p), filepath.Join(dir, "cuda-inputs"), filepath.Join(dir, "cuda")
	for _, root := range []string{dir, cache, inputs, libs} {
		if err := diffusionDir(root); err != nil {
			return err
		}
	}
	lifeLock, err := lockDiffusionFile(dir, ".diffusion-child.lock")
	if err != nil {
		return err
	}
	defer lifeLock.Close()
	binary, err := diffusionVerifiedFile(ctx, dir, pins[0], 0500, io.Discard)
	if err != nil {
		return err
	}
	defer binary.Close()
	libsFD, err := diffusionDirectoryFD(libs)
	if err != nil {
		return err
	}
	defer libsFD.Close()
	cacheFD, err := diffusionDirectoryFD(cache)
	if err != nil {
		return err
	}
	defer cacheFD.Close()
	libsPath, cachePath := diffusionFD(libsFD.Fd()), diffusionFD(cacheFD.Fd())
	for _, wheel := range diffusionWheels() {
		if err := diffusionWheelLibraries(ctx, inputs, libsPath, wheel, false); err != nil {
			return err
		}
	}
	// The child loader searches exactly this directory. Reject any extra file
	// rather than allowing an unreviewed library to shadow a host dependency.
	entries, err := libsFD.ReadDir(-1)
	if err != nil || len(entries) != 3 {
		return fmt.Errorf("unexpected contents of private CUDA library directory")
	}
	for _, f := range diffusionArtifacts(p.Task) {
		if err := diffusionCheckFile(ctx, cachePath, f.ArtifactPin, 0600, io.Discard); err != nil {
			return err
		}
	}
	key, err := readAPIKey(p.APIKeyFile)
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp4", fmt.Sprintf("%s:%d", nativeListen(p), p.Port))
	if err != nil {
		return fmt.Errorf("diffusion listener unavailable")
	}
	defer listener.Close()
	lock, err := lockDiffusionGPU(p.GPU.UUID)
	if err != nil {
		return err
	}
	defer lock.Close()
	if err := admitDiffusionGPU(ctx, p); err != nil {
		return err
	}
	if err := probeDiffusionCUDA(ctx, dir, p.GPU.UUID, pins[1], libsFD, lock); err != nil {
		return err
	}
	if err := admitDiffusionGPU(ctx, p); err != nil {
		return err
	}
	mode := "--private-stdio"
	if p.Task == diffusionEditTask {
		mode = "--private-edit-worker-stdio"
	}
	cmd := exec.Command(diffusionFD(3), mode, diffusionFD(5))
	cmd.Env, cmd.Dir = diffusionEnv(dir, p.GPU.UUID, diffusionFD(4)), dir
	cmd.ExtraFiles = []*os.File{binary, libsFD, cacheFD, lock, lifeLock}
	fmt.Printf("Loading reviewed GGUF diffusion weights and running fixed startup %s probe.\n", p.Task)
	return serveDiffusion(ctx, listener, cmd, p.Task, p.Model, p.Revision, runID, key, diffusionStartupTimeout, diffusionRequestTimeout)
}
