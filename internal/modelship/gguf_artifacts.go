package modelship

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

type ggufArtifacts struct {
	Schema     int           `json:"schema"`
	Repo       string        `json:"repo"`
	Revision   string        `json:"revision"`
	Default    string        `json:"default"`
	Models     []ArtifactPin `json:"models"`
	Projectors []ArtifactPin `json:"projectors"`
}

func loadGGUFArtifacts() (ggufArtifacts, error) {
	var c ggufArtifacts
	b, e := assets.ReadFile("assets/gguf_artifacts.json")
	if e != nil {
		return c, e
	}
	if json.Unmarshal(b, &c) != nil || c.Schema != 1 || c.Repo != ggufRepo || c.Revision != ggufRevision || c.Default != ggufFilename || len(c.Models) < 1 || len(c.Models) > 64 || len(c.Projectors) != 1 {
		return c, fmt.Errorf("invalid reviewed GGUF artifact catalog")
	}
	seen := map[string]bool{}
	for _, group := range [][]ArtifactPin{c.Models, c.Projectors} {
		for _, p := range group {
			if p.Size <= 0 || p.Size > 100<<30 || !artifactDigestPattern.MatchString(p.SHA256) || strings.ContainsAny(p.Name, "/\\\x00") || !strings.HasSuffix(p.Name, ".gguf") || p.URL != "" || seen[p.Name] {
				return c, fmt.Errorf("invalid reviewed GGUF artifact pin")
			}
			seen[p.Name] = true
		}
	}
	if !seen[c.Default] || c.Projectors[0].Name != "mmproj-F16.gguf" {
		return c, fmt.Errorf("missing default GGUF/projector pairing")
	}
	return c, nil
}
func ggufArtifact(repo, revision, file string, projector bool) (ArtifactPin, error) {
	c, e := loadGGUFArtifacts()
	if e != nil {
		return ArtifactPin{}, e
	}
	if repo != c.Repo || revision != c.Revision {
		return ArtifactPin{}, fmt.Errorf("GGUF profile requires %s @ %s; no model/revision substitution", c.Repo, c.Revision)
	}
	list := c.Models
	if projector {
		list = c.Projectors
	}
	for _, p := range list {
		if p.Name == file {
			return p, nil
		}
	}
	return ArtifactPin{}, fmt.Errorf("select a reviewed standalone GGUF %s; auxiliary, split, calibration and unknown files are not enabled", map[bool]string{true: "projector", false: "quantization"}[projector])
}
func ggufChoices(m Model) ([]ArtifactPin, error) {
	c, e := loadGGUFArtifacts()
	if e != nil {
		return nil, e
	}
	if m.Hub.ID != c.Repo || m.Hub.SHA != c.Revision {
		return nil, fmt.Errorf("no reviewed quantization catalog for this revision")
	}
	available := map[string]int64{}
	for _, f := range m.Hub.Siblings {
		available[f.Name] = f.Bytes()
	}
	out := []ArtifactPin{}
	for _, p := range c.Models {
		if available[p.Name] == p.Size {
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Name == c.Default {
			return true
		}
		if out[j].Name == c.Default {
			return false
		}
		return out[i].Size < out[j].Size
	})
	return out, nil
}
