package modelship

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

var repoPart = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9_.-]*$`)
var commitPattern = regexp.MustCompile(`^[a-f0-9]{40}$`)

type ModelFile struct {
	Name string `json:"rfilename"`
	Size int64  `json:"size"`
	LFS  *struct {
		Size int64 `json:"size"`
	} `json:"lfs,omitempty"`
}

func (f ModelFile) Bytes() int64 {
	if f.Size > 0 {
		return f.Size
	}
	if f.LFS != nil {
		return f.LFS.Size
	}
	return 0
}

type HubModel struct {
	ID          string      `json:"id"`
	SHA         string      `json:"sha"`
	PipelineTag string      `json:"pipeline_tag"`
	Library     string      `json:"library_name"`
	Gated       any         `json:"gated"`
	Siblings    []ModelFile `json:"siblings"`
	Safetensors struct {
		Total int64 `json:"total"`
	} `json:"safetensors"`
	CardData struct {
		License any `json:"license"`
	} `json:"cardData"`
}
type ModelConfig struct {
	Architectures []string        `json:"architectures"`
	ModelType     string          `json:"model_type"`
	Quantization  json.RawMessage `json:"quantization_config"`
	Layers        int             `json:"num_hidden_layers"`
	Heads         int             `json:"num_attention_heads"`
	KVHeads       int             `json:"num_key_value_heads"`
	Hidden        int             `json:"hidden_size"`
	HeadDim       int             `json:"head_dim"`
	MaxPosition   int             `json:"max_position_embeddings"`
}
type ModelIndex struct {
	Class string `json:"_class_name"`
}
type Model struct {
	Hub    HubModel
	Config ModelConfig
	Index  ModelIndex
}
type Hub struct {
	BaseURL string
	Token   string
	Client  *http.Client
}

func NewHub() *Hub {
	return &Hub{BaseURL: "https://huggingface.co", Client: &http.Client{Timeout: 45 * time.Second}}
}
func ParseModel(input string) (string, error) {
	id := strings.TrimSuffix(input, "/")
	if strings.Contains(id, "://") {
		u, err := url.Parse(id)
		if err != nil || u.Scheme != "https" || u.Host != "huggingface.co" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
			return "", fmt.Errorf("use an HTTPS huggingface.co model URL without query parameters, or a repo ID")
		}
		id = strings.TrimPrefix(u.Path, "/")
	}
	parts := strings.Split(id, "/")
	if len(parts) < 1 || len(parts) > 2 {
		return "", fmt.Errorf("use a model repository root, not a file/tree URL; select a revision with --revision")
	}
	for _, p := range parts {
		if !repoPart.MatchString(p) || strings.Contains(p, "..") || strings.HasSuffix(p, ".git") {
			return "", fmt.Errorf("invalid model repository ID")
		}
	}
	return id, nil
}
func (h *Hub) get(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, h.BaseURL+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "modelship/0.1")
	if h.Token != "" {
		req.Header.Set("Authorization", "Bearer "+h.Token)
	}
	resp, err := h.Client.Do(req)
	if err != nil {
		return fmt.Errorf("Hugging Face request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		switch resp.StatusCode {
		case 401, 403:
			return fmt.Errorf("Hugging Face denied access (%d): set HF_TOKEN and accept any gated model license on its model page", resp.StatusCode)
		case 404:
			return fmt.Errorf("Hugging Face repository, revision, or required configuration not found (404)")
		default:
			return fmt.Errorf("Hugging Face returned HTTP %d", resp.StatusCode)
		}
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 16*1024*1024+1))
	if err != nil {
		return err
	}
	if len(data) > 16*1024*1024 {
		return fmt.Errorf("Hugging Face metadata exceeds 16 MiB")
	}
	if err = json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("invalid Hugging Face metadata: %w", err)
	}
	return nil
}
func (h *Hub) Inspect(ctx context.Context, id, revision string) (Model, error) {
	var m Model
	if revision == "" || len(revision) > 200 || strings.ContainsAny(revision, "\r\n?#") {
		return m, fmt.Errorf("invalid revision")
	}
	err := h.get(ctx, "/api/models/"+id+"/revision/"+url.PathEscape(revision)+"?blobs=true", &m.Hub)
	if err != nil {
		return m, err
	}
	if !commitPattern.MatchString(m.Hub.SHA) {
		return m, fmt.Errorf("repository response has no valid immutable commit SHA")
	}
	// Use the requested, validated ID, not an arbitrary ID supplied by metadata.
	m.Hub.ID = id
	has := func(name string) bool {
		for _, f := range m.Hub.Siblings {
			if f.Name == name {
				return true
			}
		}
		return false
	}
	base := "/" + id + "/resolve/" + m.Hub.SHA + "/"
	if has("model_index.json") {
		if err = h.get(ctx, base+"model_index.json", &m.Index); err != nil {
			return m, err
		}
	} else if has("config.json") {
		if err = h.get(ctx, base+"config.json", &m.Config); err != nil {
			return m, err
		}
	} else if has("adapter_config.json") {
		return m, fmt.Errorf("this is an adapter/LoRA, not a standalone model; base-model composition is not implemented")
	} else {
		return m, fmt.Errorf("no supported config.json or Diffusers model_index.json; GGUF and single-file checkpoints are not supported yet")
	}
	return m, nil
}
