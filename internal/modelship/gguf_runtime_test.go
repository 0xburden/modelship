package modelship

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGGUFRuntimeCatalogKeepsLegacyPinsAndABI(t *testing.T) {
	r, e := loadLlamaRuntime(llamaLegacyRuntime)
	if e != nil {
		t.Fatal(e)
	}
	if r.MinimumGLIBC != "2.38" || r.MinimumGLIBCXX != "3.4.32" || len(r.Artifacts) != 2 || r.Artifacts[0].SHA256 != "4af1e1da23661db4a5b84b29aae16ebc053df689c5b8b1427c4670f2c6406d8c" {
		t.Fatal("legacy deployment runtime was silently replaced")
	}
	if _, e = loadLlamaRuntime("latest"); e == nil {
		t.Fatal("unknown runtime identity accepted")
	}
}
func TestGGUFRuntimeCatalogRejectsUnverifiedOrCredentialURLs(t *testing.T) {
	bytes, e := assets.ReadFile("assets/llama_runtimes.json")
	if e != nil {
		t.Fatal(e)
	}
	for _, name := range []string{"default", "source", "digest", "http", "userinfo", "query", "origin", "size", "names"} {
		t.Run(name, func(t *testing.T) {
			c, e := parseLlamaRuntimes(bytes)
			if e != nil {
				t.Fatal(e)
			}
			p := &c.Runtimes[0].Artifacts[0]
			switch name {
			case "default":
				c.Default = "unknown"
			case "source":
				c.Runtimes[0].SourceCommit = "main"
			case "digest":
				p.SHA256 = "unknown"
			case "http":
				p.URL = strings.Replace(p.URL, "https:", "http:", 1)
			case "userinfo":
				p.URL = strings.Replace(p.URL, "github.com", "credential@github.com", 1)
			case "query":
				p.URL += "?token=must-not-accept"
			case "origin":
				p.URL = strings.Replace(p.URL, "ggml-org/llama.cpp", "unknown/other", 1)
			case "size":
				p.Size = 0
			case "names":
				c.Runtimes[0].Artifacts[1].Name = p.Name
			}
			b, _ := json.Marshal(c)
			if _, e = parseLlamaRuntimes(b); e == nil {
				t.Fatal("invalid runtime pin catalog accepted")
			}
		})
	}
}
func TestGGUFRebuiltIdentityHasIndependentABIPolicy(t *testing.T) {
	legacy, e := loadLlamaRuntime(llamaLegacyRuntime)
	if e != nil {
		t.Fatal(e)
	}
	// Synthetic pin: verifies catalog policy/selection only, not artifact bytes.
	rebuilt := legacy
	rebuilt.ID = llamaJammyRuntime
	rebuilt.MinimumGLIBC = "2.35"
	rebuilt.MinimumGLIBCXX = "3.4.30"
	rebuilt.Artifacts = append([]ArtifactPin(nil), legacy.Artifacts...)
	rebuilt.Artifacts[0] = ArtifactPin{Name: "core.tar.gz", Size: 1, SHA256: strings.Repeat("a", 64), URL: "https://github.com/0xburden/modelship/releases/download/runtime-llama-b12110-u2204-r1/fixture.tar.gz"}
	c := llamaRuntimeCatalog{Schema: 1, Default: llamaJammyRuntime, Runtimes: []llamaRuntimeSpec{legacy, rebuilt}}
	b, _ := json.Marshal(c)
	got, e := parseLlamaRuntimes(b)
	if e != nil || got.Default != llamaJammyRuntime {
		t.Fatal(e)
	}
	have := runtimeABI{GLIBC: "2.35", GLIBCXX: "3.4.30"}
	if validateRuntimeABI(have, runtimeABI{rebuilt.MinimumGLIBC, rebuilt.MinimumGLIBCXX}) != nil {
		t.Fatal("confirmed host ABI rejected")
	}
	if validateRuntimeABI(have, runtimeABI{legacy.MinimumGLIBC, legacy.MinimumGLIBCXX}) == nil {
		t.Fatal("incompatible legacy runtime allowed")
	}
}
