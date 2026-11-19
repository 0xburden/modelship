package modelship

// Private GGUF diffusion generation gateway, behind the verified supervisor.
import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

const (
	diffusionSide         = 512
	diffusionSteps        = 40
	diffusionCFG          = 6
	diffusionPromptBytes  = 1024
	diffusionMaxBody      = 8 << 10
	diffusionRuntime      = "stable-diffusion.cpp"
	diffusionTask         = "text-to-image"
	diffusionProfileID    = "qwen-image21-gguf-generation-v1"
	diffusionWarmupPrompt = "A watercolor painting of a red sailboat on a calm alpine lake, with pine trees and distant mountains."
	diffusionWarmupSeed   = 43
)

var errDiffusionPrompt = errors.New("invalid diffusion prompt")
var errDiffusionProtocol = errors.New("private diffusion worker protocol failed")

func validDiffusionPrompt(prompt string) bool {
	if len(prompt) == 0 || len(prompt) > diffusionPromptBytes || !utf8.ValidString(prompt) ||
		strings.ContainsAny(prompt, "<>()[\\]") || strings.Contains(prompt, "BREAK") {
		return false
	}
	nonblank := false
	for _, r := range prompt {
		if (r < 0x20 && r != '\t' && r != '\n') || (r >= 0x7f && r <= 0x9f) {
			return false
		}
		nonblank = nonblank || r != ' ' && r != '\t' && r != '\n'
	}
	return nonblank
}

// diffusionWire speaks only protocol v1. Its owner supplies pipe deadlines and
// must terminate/reap the child on any protocol failure. Never resynchronize
// after partial IO, retry a frame, or call it from concurrent requests.
type diffusionWire struct {
	in  io.Reader
	out io.Writer
}

func (d diffusionWire) loaded() error {
	var frame [6]byte
	if _, err := io.ReadFull(d.in, frame[:]); err != nil || frame != [6]byte{0, 0, 0, 2, 'L', 1} {
		return errDiffusionProtocol
	}
	return nil
}

func (d diffusionWire) generate(prompt string, seed uint32) (*image.NRGBA, error) {
	if !validDiffusionPrompt(prompt) {
		return nil, errDiffusionPrompt
	}
	var request [8 + diffusionPromptBytes]byte
	binary.BigEndian.PutUint32(request[:4], uint32(4+len(prompt)))
	binary.BigEndian.PutUint32(request[4:8], seed)
	copy(request[8:], prompt)
	if n, err := d.out.Write(request[:8+len(prompt)]); err != nil || n != 8+len(prompt) {
		return nil, errDiffusionProtocol
	}
	var header [5]byte
	if _, err := io.ReadFull(d.in, header[:]); err != nil {
		return nil, errDiffusionProtocol
	}
	length := binary.BigEndian.Uint32(header[:4])
	if header[4] == 'V' && length == 1 {
		return nil, errDiffusionPrompt
	}
	const pixels = diffusionSide * diffusionSide
	if header[4] != 'I' || length != 2+3*pixels && length != 2+4*pixels {
		return nil, errDiffusionProtocol
	}
	var channels [1]byte
	if _, err := io.ReadFull(d.in, channels[:]); err != nil || uint32(channels[0]) != (length-2)/pixels {
		return nil, errDiffusionProtocol
	}
	out := image.NewNRGBA(image.Rect(0, 0, diffusionSide, diffusionSide))
	if channels[0] == 4 {
		if _, err := io.ReadFull(d.in, out.Pix); err != nil {
			return nil, errDiffusionProtocol
		}
		return out, nil
	}
	rgb := make([]byte, pixels*3)
	if _, err := io.ReadFull(d.in, rgb); err != nil {
		return nil, errDiffusionProtocol
	}
	for i := 0; i < pixels; i++ {
		copy(out.Pix[4*i:4*i+3], rgb[i*3:i*3+3])
		out.Pix[i*4+3] = 255
	}
	return out, nil
}

type diffusionRequest struct {
	prompt string
	seed   *uint32
}

// Only exact field names and scalar types. Assert fixed options when provided
// rather than silently substituting the profile's dimensions, steps, or CFG.
func parseDiffusionRequest(body []byte, model string) (diffusionRequest, error) {
	var req diffusionRequest
	bad := fmt.Errorf("invalid request: use a plain-text prompt, 512×512, 40 steps, guidance 6, and optional uint32 seed")
	d := json.NewDecoder(bytes.NewReader(body))
	token, err := d.Token()
	if err != nil || token != json.Delim('{') || !utf8.Valid(body) {
		return req, bad
	}
	seen := map[string]bool{}
	for d.More() {
		token, err = d.Token()
		field, ok := token.(string)
		if err != nil || !ok || seen[field] {
			return req, bad
		}
		seen[field] = true
		var raw json.RawMessage
		if err = d.Decode(&raw); err != nil || bytes.Equal(raw, []byte("null")) {
			return req, bad
		}
		switch field {
		case "prompt":
			if json.Unmarshal(raw, &req.prompt) != nil || !validDiffusionPrompt(req.prompt) {
				return req, bad
			}
		case "model":
			var s string
			if json.Unmarshal(raw, &s) != nil || s != model {
				return req, bad
			}
		case "seed":
			var v uint32
			if json.Unmarshal(raw, &v) != nil {
				return req, bad
			}
			req.seed = &v
		case "width", "height", "steps":
			var v int
			want := diffusionSide
			if field == "steps" {
				want = diffusionSteps
			}
			if json.Unmarshal(raw, &v) != nil || v != want {
				return req, bad
			}
		case "guidance_scale":
			var v float64
			if json.Unmarshal(raw, &v) != nil || v != diffusionCFG {
				return req, bad
			}
		default:
			return req, bad
		}
	}
	if token, err = d.Token(); err != nil || token != json.Delim('}') || !seen["prompt"] || d.Decode(new(any)) != io.EOF {
		return req, bad
	}
	return req, nil
}

type diffusionGateway struct {
	model, revision, runID string
	digest                 [sha256.Size]byte
	// Lifecycle is monotonic: warming 0 -> ready 1 -> unavailable 2.
	state atomic.Uint32
	busy  chan struct{}
	// Bounded by the supervisor, not the client request's context. Once sent,
	// the request retains the GPU slot until synchronized output or failure.
	generate func(string, uint32) (*image.NRGBA, error)
	poison   func()
}

func newDiffusionGateway(model, revision, runID, key string, generate func(string, uint32) (*image.NRGBA, error), poison func()) (*diffusionGateway, error) {
	if !commitPattern.MatchString(revision) || !runIDPattern.MatchString(runID) || !apiKeyPattern.MatchString(key) || generate == nil || poison == nil {
		return nil, fmt.Errorf("invalid private diffusion gateway configuration")
	}
	return &diffusionGateway{model: model, revision: revision, runID: runID, digest: sha256.Sum256([]byte(key)), busy: make(chan struct{}, 1), generate: generate, poison: poison}, nil
}

func (g *diffusionGateway) ready() bool { return g.state.Load() == 1 }

// The supervisor calls this only after a verified synchronized warm-up frame.
func (g *diffusionGateway) warmed() { g.state.CompareAndSwap(0, 1) }

func (g *diffusionGateway) unavailable() { g.state.Store(2) }

func (g *diffusionGateway) authorized(values []string) bool {
	if len(values) != 1 || len(values[0]) > 512 {
		return false
	}
	scheme, key, ok := strings.Cut(values[0], " ")
	if !ok || !strings.EqualFold(scheme, "Bearer") || !apiKeyPattern.MatchString(key) {
		return false
	}
	digest := sha256.Sum256([]byte(key))
	return subtle.ConstantTimeCompare(digest[:], g.digest[:]) == 1
}

func diffusionJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func diffusionError(w http.ResponseWriter, status int, message string) {
	diffusionJSON(w, status, map[string]string{"detail": message})
}

func (g *diffusionGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if !g.authorized(r.Header.Values("Authorization")) {
		w.Header().Set("WWW-Authenticate", `Bearer realm="modelship"`)
		diffusionError(w, 401, "Authentication required")
		return
	}
	method := "GET"
	switch r.URL.Path {
	case "/v1/images/generations", "/v1/images/edits":
		method = "POST"
	case "/health", "/v1/models", "/v1/capabilities":
	default:
		diffusionError(w, 404, "route not available on this diffusion worker")
		return
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		diffusionError(w, 405, "method not allowed")
		return
	}
	switch r.URL.Path {
	case "/health":
		ready := g.ready()
		status := 503
		if ready {
			status = 200
		}
		diffusionJSON(w, status, map[string]any{"ready": ready, "model": g.model, "revision": g.revision, "task": diffusionTask,
			"recipe": diffusionProfileID, "runtime": diffusionRuntime, "run_id": g.runID})
	case "/v1/models":
		diffusionJSON(w, 200, map[string]any{"object": "list", "data": []map[string]string{{"id": g.model, "object": "model", "owned_by": "modelship"}}})
	case "/v1/capabilities":
		diffusionJSON(w, 200, map[string]any{"schema_version": 1, "model": g.model, "revision": g.revision, "runtime": diffusionRuntime,
			"run_id": g.runID, "image_generation": map[string]any{"width": diffusionSide, "height": diffusionSide, "steps": diffusionSteps,
				"guidance_scale": diffusionCFG, "max_prompt_bytes": diffusionPromptBytes, "max_seed": uint32(1<<32 - 1)}})
	case "/v1/images/edits":
		diffusionError(w, 409, "this deployment serves /v1/images/generations only")
	case "/v1/images/generations":
		g.generation(w, r)
	}
}

func (g *diffusionGateway) generation(w http.ResponseWriter, r *http.Request) {
	if !g.ready() {
		diffusionError(w, 503, "image worker unavailable; check readiness and service logs")
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, diffusionMaxBody))
	if err != nil {
		var limit *http.MaxBytesError
		if errors.As(err, &limit) {
			diffusionError(w, 413, "image request too large")
		} else {
			diffusionError(w, 422, "could not read image request")
		}
		return
	}
	req, err := parseDiffusionRequest(body, g.model)
	if err != nil {
		diffusionError(w, 422, err.Error())
		return
	}
	select {
	case g.busy <- struct{}{}:
		defer func() { <-g.busy }()
	default:
		diffusionError(w, 503, "GPU busy; retry after the current request finishes")
		return
	}
	if !g.ready() {
		diffusionError(w, 503, "image worker unavailable; check readiness and service logs")
		return
	}
	if r.Context().Err() != nil {
		return
	}
	var seed uint32
	if req.seed != nil {
		seed = *req.seed
	} else {
		var random [4]byte
		if _, err := rand.Read(random[:]); err != nil {
			diffusionError(w, 500, "could not select a seed")
			return
		}
		seed = binary.BigEndian.Uint32(random[:])
	}
	pixels, err := g.generate(req.prompt, seed)
	if errors.Is(err, errDiffusionPrompt) {
		diffusionError(w, 422, "invalid diffusion prompt")
		return
	}
	if err != nil || pixels == nil || pixels.Bounds() != image.Rect(0, 0, diffusionSide, diffusionSide) {
		g.unavailable()
		g.poison()
		diffusionError(w, 503, "image worker failed; restart before submitting again")
		return
	}
	if r.Context().Err() != nil {
		return
	}
	var buf bytes.Buffer
	encoder := png.Encoder{CompressionLevel: png.BestSpeed}
	if err := encoder.Encode(&buf, pixels); err != nil {
		diffusionError(w, 500, "could not encode image")
		return
	}
	diffusionJSON(w, 200, map[string]any{"created": time.Now().Unix(), "model": g.model, "seed": seed,
		"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(buf.Bytes())}}})
}
