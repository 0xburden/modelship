package modelship

import (
	"encoding/json"
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

const llamaSourceCommit = "cd959bae99ff4bed22bef2604f45241c09bbb8c8"
const llamaLegacyRuntime = "llama-b12110-cuda12.8"
const llamaJammyRuntime = "llama-b12110-cuda12.8-ubuntu22.04-r1"

type llamaRuntimeSpec struct {
	ID             string        `json:"id"`
	SourceCommit   string        `json:"source_commit"`
	MinimumGLIBC   string        `json:"minimum_glibc"`
	MinimumGLIBCXX string        `json:"minimum_glibcxx"`
	Description    string        `json:"description"`
	Artifacts      []ArtifactPin `json:"artifacts"`
}
type llamaRuntimeCatalog struct {
	Schema   int                `json:"schema"`
	Default  string             `json:"default"`
	Runtimes []llamaRuntimeSpec `json:"runtimes"`
}

var abiVersionPattern = regexp.MustCompile(`^[0-9]{1,3}(\.[0-9]{1,3}){1,3}$`)
var artifactDigestPattern = regexp.MustCompile(`^[a-f0-9]{64}$`)

func parseLlamaRuntimes(data []byte) (llamaRuntimeCatalog, error) {
	var c llamaRuntimeCatalog
	if len(data) > 64<<10 || json.Unmarshal(data, &c) != nil || c.Schema != 1 || len(c.Runtimes) < 1 || len(c.Runtimes) > 8 {
		return c, fmt.Errorf("invalid embedded llama.cpp runtime catalog")
	}
	ids := map[string]bool{}
	for _, r := range c.Runtimes {
		if (r.ID != llamaLegacyRuntime && r.ID != llamaJammyRuntime) || ids[r.ID] || r.SourceCommit != llamaSourceCommit || !abiVersionPattern.MatchString(r.MinimumGLIBC) || !abiVersionPattern.MatchString(r.MinimumGLIBCXX) || len(r.Artifacts) != 2 {
			return c, fmt.Errorf("invalid pinned llama.cpp runtime specification")
		}
		ids[r.ID] = true
		names := map[string]bool{}
		for _, pin := range r.Artifacts {
			u, e := url.Parse(pin.URL)
			if e != nil || u.Scheme != "https" || u.Host != "github.com" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.RawPath != "" || pin.Size <= 0 || pin.Size > 2<<30 || !artifactDigestPattern.MatchString(pin.SHA256) || (pin.Name != "core.tar.gz" && pin.Name != "cuda.tar.gz") || names[pin.Name] {
				return c, fmt.Errorf("invalid pinned llama.cpp runtime artifact")
			}
			upstream := strings.HasPrefix(u.Path, "/ggml-org/llama.cpp/releases/download/b12110/")
			rebuilt := strings.HasPrefix(u.Path, "/0xburden/modelship/releases/download/runtime-llama-b12110-u2204-r1/")
			if !upstream && !rebuilt {
				return c, fmt.Errorf("unapproved llama.cpp artifact origin/revision")
			}
			names[pin.Name] = true
		}
	}
	if !ids[c.Default] {
		return c, fmt.Errorf("default llama.cpp runtime is not pinned")
	}
	return c, nil
}
func loadLlamaRuntime(id string) (llamaRuntimeSpec, error) {
	data, e := assets.ReadFile("assets/llama_runtimes.json")
	if e != nil {
		return llamaRuntimeSpec{}, e
	}
	c, e := parseLlamaRuntimes(data)
	if e != nil {
		return llamaRuntimeSpec{}, e
	}
	if id == "" {
		id = c.Default
	}
	for _, r := range c.Runtimes {
		if r.ID == id {
			return r, nil
		}
	}
	return llamaRuntimeSpec{}, fmt.Errorf("unsupported saved GGUF runtime identity; no implicit runtime substitution")
}
