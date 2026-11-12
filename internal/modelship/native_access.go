package modelship

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"syscall"
	"time"
)

var apiKeyPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{32,128}$`)

func nativeListen(p Plan) string {
	if p.Listen == "" {
		return "127.0.0.1"
	}
	return p.Listen
}
func validateNativeAccess(p Plan, checkKey bool) error {
	if nativeListen(p) != "127.0.0.1" && nativeListen(p) != "0.0.0.0" {
		return fmt.Errorf("native --listen must be 127.0.0.1 or 0.0.0.0")
	}
	if nativeListen(p) != "127.0.0.1" && p.APIKeyFile == "" {
		return fmt.Errorf("non-loopback listeners require --api-key-file; generate one with modelship auth-keygen")
	}
	if p.APIKeyFile != "" {
		if !filepath.IsAbs(p.APIKeyFile) {
			return fmt.Errorf("API key file path must be absolute")
		}
		if checkKey {
			_, err := readAPIKey(p.APIKeyFile)
			return err
		}
	}
	return nil
}
func accessOptions(o *Options) error {
	if o.Backend != "native" {
		if o.Listen != "" || o.APIKeyFile != "" {
			return fmt.Errorf("--listen and --api-key-file are supported only with --backend native")
		}
		return nil
	}
	if o.Listen == "" {
		o.Listen = "127.0.0.1"
	}
	if o.APIKeyFile != "" {
		path, err := filepath.Abs(o.APIKeyFile)
		if err != nil {
			return fmt.Errorf("invalid API key file path")
		}
		o.APIKeyFile = path
	}
	return validateNativeAccess(Plan{Listen: o.Listen, APIKeyFile: o.APIKeyFile}, true)
}

// Never follow a final symlink or block on a FIFO. No key bytes enter errors.
func readAPIKey(path string) (string, error) {
	invalid := fmt.Errorf("API key file must be an owned regular file with mode 0600, containing 32-128 base64url characters")
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
	if err != nil {
		return "", invalid
	}
	f := os.NewFile(uintptr(fd), path)
	defer f.Close()
	st, err := f.Stat()
	if err != nil || !st.Mode().IsRegular() || st.Mode().Perm() != 0600 || st.Mode()&(os.ModeSetuid|os.ModeSetgid|os.ModeSticky) != 0 || st.Size() < 32 || st.Size() > 130 {
		return "", invalid
	}
	stat, ok := st.Sys().(*syscall.Stat_t)
	if !ok || int(stat.Uid) != os.Geteuid() {
		return "", invalid
	}
	data, err := io.ReadAll(io.LimitReader(f, 131))
	if err != nil {
		return "", invalid
	}
	data = bytes.TrimSuffix(data, []byte("\n"))
	data = bytes.TrimSuffix(data, []byte("\r"))
	if !apiKeyPattern.Match(data) {
		return "", invalid
	}
	return string(data), nil
}
func authKeygenCommand(args []string) error {
	fs := flag.NewFlagSet("auth-keygen", flag.ContinueOnError)
	path := fs.String("file", "", "new private API key file; defaults to XDG_CONFIG_HOME/modelship/api.key")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("auth-keygen accepts --file only; never pass key values as arguments")
	}
	if *path == "" {
		base := os.Getenv("XDG_CONFIG_HOME")
		if base == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				return err
			}
			base = filepath.Join(home, ".config")
		}
		if !filepath.IsAbs(base) {
			return fmt.Errorf("XDG_CONFIG_HOME must be absolute")
		}
		*path = filepath.Join(base, "modelship", "api.key")
	}
	absolute, err := filepath.Abs(*path)
	if err != nil {
		return err
	}
	if err = privateDir(filepath.Dir(absolute)); err != nil {
		return err
	}
	key := make([]byte, 32)
	if _, err = rand.Read(key); err != nil {
		return err
	}
	f, err := os.OpenFile(absolute, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return fmt.Errorf("cannot create API key file; existing files are never overwritten")
	}
	_, err = f.WriteString(base64.RawURLEncoding.EncodeToString(key) + "\n")
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(absolute)
		return fmt.Errorf("could not write API key file")
	}
	fmt.Printf("API key saved privately to %s (key value not printed).\n", absolute)
	return nil
}
func nativeClient(timeout time.Duration) *http.Client {
	return &http.Client{Timeout: timeout, Transport: &http.Transport{Proxy: nil}, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
}
func nativeAuthorize(req *http.Request, p Plan) error {
	if p.APIKeyFile == "" {
		return nil
	}
	key, err := readAPIKey(p.APIKeyFile)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	return nil
}
func configureNativeCommand(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("configure", flag.ContinueOnError)
	backend := fs.String("backend", "docker", "native only")
	name := fs.String("name", "modelship", "deployment name")
	listen := fs.String("listen", "", "127.0.0.1 or 0.0.0.0")
	key := fs.String("api-key-file", "", "private key file; empty disables auth on loopback only")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *backend != "native" {
		return fmt.Errorf("configure requires --backend native and no positional arguments")
	}
	var l, k *string
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "listen" {
			l = listen
		}
		if f.Name == "api-key-file" {
			k = key
		}
	})
	if l == nil && k == nil {
		return fmt.Errorf("configure requires --listen and/or --api-key-file")
	}
	return ConfigureNative(ctx, *name, l, k)
}

// Configuration never interrupts a running worker or changes model dependencies.
func ConfigureNative(ctx context.Context, name string, listen, key *string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	cfg, dir, err := nativeConfig(name)
	if err != nil {
		return err
	}
	unlock, err := nativeLock(dir)
	if err != nil {
		return err
	}
	defer unlock()
	if run, e := loadNativeRun(dir); e == nil {
		if nativeAlive(dir, run) {
			return fmt.Errorf("stop the deployment before changing access settings; no worker was interrupted")
		}
	} else if !os.IsNotExist(e) {
		return fmt.Errorf("cannot verify stopped worker identity; access settings unchanged")
	}
	if listen != nil {
		cfg.Plan.Listen = *listen
	}
	if key != nil {
		cfg.Plan.APIKeyFile = *key
		if *key != "" {
			cfg.Plan.APIKeyFile, err = filepath.Abs(*key)
			if err != nil {
				return fmt.Errorf("invalid API key path")
			}
		}
	}
	if err = validateNativeAccess(cfg.Plan, true); err != nil {
		return err
	}
	// Auth module first, entrypoint second, metadata last. A partial failure leaves
	// legacy configs loopback-only. Existing inference modules and venv stay intact.
	for _, name := range []string{"api_auth.py", "native_entry.py"} {
		data, e := assets.ReadFile("assets/" + name)
		if e != nil {
			return e
		}
		f, e := os.CreateTemp(dir, ".access-")
		if e != nil {
			return e
		}
		temp := f.Name()
		_, e = f.Write(data)
		closeErr := f.Close()
		if e == nil {
			e = closeErr
		}
		if e == nil {
			e = os.Rename(temp, filepath.Join(dir, name))
		}
		_ = os.Remove(temp)
		if e != nil {
			return e
		}
	}
	cfg.Schema = nativeSchemaVersion
	if err = writeJSONFile(filepath.Join(dir, "config.json"), cfg); err != nil {
		return err
	}
	fmt.Printf("Updated access for %s: listen=%s, authentication=%t. Worker remains stopped; start explicitly.\n", name, nativeListen(cfg.Plan), cfg.Plan.APIKeyFile != "")
	return nil
}

func nativeHealth(ctx context.Context, cfg NativeConfig, run NativeRun) (bool, string) {
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("http://127.0.0.1:%d/health", cfg.Plan.Port), nil)
	if err != nil {
		return false, "readiness unavailable"
	}
	if err = nativeAuthorize(req, cfg.Plan); err != nil {
		return false, "API key unavailable or insecure"
	}
	client := nativeClient(3 * time.Second)
	defer client.CloseIdleConnections()
	resp, err := client.Do(req)
	if err != nil {
		return false, "worker not ready or unreachable"
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Sprintf("readiness HTTP %d", resp.StatusCode)
	}
	var health struct {
		Ready    bool   `json:"ready"`
		Model    string `json:"model"`
		Revision string `json:"revision"`
		RunID    string `json:"run_id"`
	}
	if err = json.NewDecoder(io.LimitReader(resp.Body, 65536)).Decode(&health); err != nil {
		return false, "invalid readiness response"
	}
	if !health.Ready || health.Model != cfg.Plan.Model || health.Revision != cfg.Plan.Revision || health.RunID != run.RunID {
		return false, "worker identity/readiness mismatch"
	}
	return true, ""
}
