package modelship

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func diffusionFixtureFrame(tag, field byte, data []byte) []byte {
	b := make([]byte, 4)
	n := len(data) + 1
	if tag == 'L' || tag == 'I' {
		n++
	}
	binary.BigEndian.PutUint32(b, uint32(n))
	b = append(b, tag)
	if tag == 'L' || tag == 'I' {
		b = append(b, field)
	}
	return append(b, data...)
}

func TestDiffusionWire(t *testing.T) {
	for _, channels := range []byte{3, 4} {
		pixels := make([]byte, diffusionSide*diffusionSide*int(channels))
		copy(pixels, []byte{1, 2, 3, 4})
		reader := bytes.NewReader(append(diffusionFixtureFrame('L', 1, nil), diffusionFixtureFrame('I', channels, pixels)...))
		var in bytes.Buffer
		wire := diffusionWire{reader, &in}
		if err := wire.loaded(); err != nil {
			t.Fatal(err)
		}
		prompt := strings.Repeat("x", 1024)
		out, err := wire.generate(prompt, 0xffffffff)
		alpha := channels
		if channels == 3 {
			alpha = 255
		}
		if err != nil || out.NRGBAAt(0, 0) != (color.NRGBA{R: 1, G: 2, B: 3, A: alpha}) || reader.Len() != 0 {
			t.Fatalf("bad response: %v", err)
		}
		req := in.Bytes()
		if binary.BigEndian.Uint32(req[:4]) != 1028 || binary.BigEndian.Uint32(req[4:8]) != 0xffffffff || string(req[8:]) != prompt {
			t.Fatal("wrong encoded request")
		}
	}
	if err := (diffusionWire{in: bytes.NewReader(diffusionFixtureFrame('L', 2, nil))}).loaded(); err != errDiffusionProtocol {
		t.Fatalf("unrecognized protocol version: %v", err)
	}
	for _, b := range [][]byte{
		nil, {0xff, 0xff, 0xff, 0xff, 'I'}, {0, 0, 0, 1, 'F'}, {0, 0, 0, 2, 'V', 0},
		diffusionFixtureFrame('I', 4, make([]byte, 3*diffusionSide*diffusionSide)),
		diffusionFixtureFrame('I', 3, make([]byte, 3*diffusionSide*diffusionSide))[:7],
	} {
		if _, err := (diffusionWire{bytes.NewReader(b), new(bytes.Buffer)}).generate("a lake", 0); err != errDiffusionProtocol {
			t.Fatalf("malformed response accepted: %v", err)
		}
	}
	if _, err := (diffusionWire{bytes.NewReader(diffusionFixtureFrame('V', 0, nil)), new(bytes.Buffer)}).generate("a lake", 0); err != errDiffusionPrompt {
		t.Fatalf("valid invalid-prompt frame classified as: %v", err)
	}
}

func TestDiffusionRequestPolicy(t *testing.T) {
	ok, err := parseDiffusionRequest([]byte(`{"prompt":"山の湖","width":512,"height":512,"steps":40,"guidance_scale":6.0,"seed":0,"model":"fixture"}`), "fixture")
	if err != nil || ok.seed == nil || *ok.seed != 0 || ok.prompt != "山の湖" {
		t.Fatalf("fixed generation request: %+v, %v", ok, err)
	}
	for _, s := range []string{
		`null`, `[]`, `{"prompt":null}`, `{"prompt":" \t\n"}`, `{"prompt":"hi\u007f"}`, `{"prompt":"hi\u0091"}`,
		`{"prompt":"a (weighted) prompt"}`, `{"prompt":"[a]"}`, `{"prompt":"<|vision_start|>"}`, `{"prompt":"a BREAK b"}`,
		`{"prompt":"bad\\escape"}`, `{"prompt":"ok","prompt":"duplicate"}`, `{"prompt":"ok","Prompt":"alias"}`,
		`{"prompt":"ok","model":"other"}`, `{"prompt":"ok","seed":-1}`, `{"prompt":"ok","seed":4294967296}`,
		`{"prompt":"ok","seed":"1"}`, `{"prompt":"ok","seed":null}`, `{"prompt":"ok","seed":1.0}`,
		`{"prompt":"ok","width":1024}`, `{"prompt":"ok","height":256}`, `{"prompt":"ok","steps":28}`,
		`{"prompt":"ok","guidance_scale":1}`, `{"prompt":"ok","negative_prompt":"bad"}`, `{"prompt":"ok","image_b64":"data"}`,
		`{"prompt":"ok","lora_strength":0}`, `{"prompt":"ok","sampler":"euler"}`, `{"prompt":"ok"} {}`,
		`{"prompt":"` + strings.Repeat("é", 513) + `"}`, "{\"prompt\":\"\xff\"}",
	} {
		if _, err := parseDiffusionRequest([]byte(s), "fixture"); err == nil {
			t.Fatalf("accepted unsupported request %q", s)
		}
	}
}

const diffusionFixtureKey = "synthetic_diffusion_test_bearer_0123456789"

func diffusionFixtureGateway(t *testing.T, gen func(string, uint32) (*image.NRGBA, error), poison func()) *diffusionGateway {
	t.Helper()
	g, err := newDiffusionGateway("fixture", strings.Repeat("a", 40), strings.Repeat("b", 32), diffusionFixtureKey, gen, poison)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func diffusionFixtureHTTP(ctx context.Context, g *diffusionGateway, method, path, body string, keys ...string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, strings.NewReader(body)).WithContext(ctx)
	for _, key := range keys {
		r.Header.Add("Authorization", "Bearer "+key)
	}
	w := httptest.NewRecorder()
	g.ServeHTTP(w, r)
	return w
}

func TestDiffusionGatewayAuthAndShape(t *testing.T) {
	ctx := context.Background()
	calls := 0
	g := diffusionFixtureGateway(t, func(prompt string, seed uint32) (*image.NRGBA, error) {
		calls++
		if prompt != "a lake" || seed != 0 {
			t.Fatalf("changed request: %q %d", prompt, seed)
		}
		return image.NewNRGBA(image.Rect(0, 0, 512, 512)), nil
	}, func() { t.Fatal("unexpected poisoning") })
	for _, tc := range []struct {
		method, path, body string
		status             int
		keys               []string
	}{
		{"GET", "/health?api_key=" + diffusionFixtureKey, "", 401, nil},
		{"GET", "/health", "", 401, []string{diffusionFixtureKey, diffusionFixtureKey}},
		{"POST", "/v1/images/generations", "bad", 401, nil},
		{"GET", "/health", "", 503, []string{diffusionFixtureKey}},
		{"GET", "/v1/models", "", 200, []string{diffusionFixtureKey}},
		{"GET", "/v1/capabilities", "", 200, []string{diffusionFixtureKey}},
		{"POST", "/v1/images/edits", "", 409, []string{diffusionFixtureKey}},
		{"POST", "/v1/chat/completions", "", 404, []string{diffusionFixtureKey}},
		{"GET", "/v1/images/generations", "", 405, []string{diffusionFixtureKey}},
	} {
		w := diffusionFixtureHTTP(ctx, g, tc.method, tc.path, tc.body, tc.keys...)
		if w.Code != tc.status || w.Header().Get("Cache-Control") != "no-store" || strings.Contains(w.Body.String(), diffusionFixtureKey) {
			t.Fatalf("%s: %d %s", tc.path, w.Code, w.Body.String())
		}
	}
	g.warmed()
	if w := diffusionFixtureHTTP(ctx, g, "GET", "/health", "", diffusionFixtureKey); w.Code != 200 || !strings.Contains(w.Body.String(), `"task":"text-to-image"`) {
		t.Fatalf("health: %s", w.Body.String())
	}
	for body, status := range map[string]int{`{"prompt":"a lake","width":1024}`: 422, strings.Repeat("x", diffusionMaxBody+1): 413} {
		if w := diffusionFixtureHTTP(ctx, g, "POST", "/v1/images/generations", body, diffusionFixtureKey); w.Code != status {
			t.Fatalf("invalid request: %d", w.Code)
		}
	}
	w := diffusionFixtureHTTP(ctx, g, "POST", "/v1/images/generations", `{"prompt":"a lake","seed":0}`, diffusionFixtureKey)
	var result struct {
		Seed *uint32
		Data []struct {
			PNG string `json:"b64_json"`
		}
	}
	if w.Code != 200 || calls != 1 || json.Unmarshal(w.Body.Bytes(), &result) != nil || result.Seed == nil || *result.Seed != 0 || len(result.Data) != 1 {
		t.Fatalf("generation status=%d calls=%d", w.Code, calls)
	}
	p, err := base64.StdEncoding.DecodeString(result.Data[0].PNG)
	if err != nil {
		t.Fatal(err)
	}
	i, err := png.Decode(bytes.NewReader(p))
	if err != nil || i.Bounds() != image.Rect(0, 0, 512, 512) {
		t.Fatalf("image decode: %v", err)
	}
}

func TestDiffusionGatewayDisconnectBusyAndPoison(t *testing.T) {
	started, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var calls, poisoned atomic.Int32
	g := diffusionFixtureGateway(t, func(string, uint32) (*image.NRGBA, error) {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
		return nil, errDiffusionProtocol
	}, func() { poisoned.Add(1) })
	g.warmed()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	post := func(ctx context.Context) *httptest.ResponseRecorder {
		return diffusionFixtureHTTP(ctx, g, "POST", "/v1/images/generations", `{"prompt":"a lake"}`, diffusionFixtureKey)
	}
	go func() { defer close(done); post(ctx) }()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("first generation never entered")
	}
	cancel()
	if w := post(context.Background()); w.Code != 503 || !strings.Contains(w.Body.String(), "GPU busy") {
		t.Fatalf("disconnect released active slot: %d %s", w.Code, w.Body.String())
	}
	close(release)
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("first generation did not complete")
	}
	g.warmed() // late readiness must not revive a failed process
	if w := post(context.Background()); w.Code != 503 || calls.Load() != 1 || poisoned.Load() != 1 || g.ready() {
		t.Fatalf("unavailable worker accepted work: %d calls=%d poison=%d", w.Code, calls.Load(), poisoned.Load())
	}
}
