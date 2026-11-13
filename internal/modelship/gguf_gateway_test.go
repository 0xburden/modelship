package modelship

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func gatewayFixture(t *testing.T, handler http.Handler) (*ggufGateway, string) {
	t.Helper()
	m, h, o := ggufPlanFixture(t)
	p, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	p.APIKeyFile = fixtureAPIKey(t)
	key, e := readAPIKey(p.APIKeyFile)
	if e != nil {
		t.Fatal(e)
	}
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	g, e := newGGUFGateway(p, "fixture-run", server.URL, "private-backend-key")
	if e != nil {
		t.Fatal(e)
	}
	g.ready.Store(true)
	t.Cleanup(g.client.CloseIdleConnections)
	return g, key
}
func gatewayCall(g *ggufGateway, key, method, path, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	if key != "" {
		r.Header.Set("Authorization", "Bearer "+key)
	}
	w := httptest.NewRecorder()
	g.ServeHTTP(w, r)
	return w
}
func TestGGUFGatewayAuthenticatesBeforeRoutingAndBody(t *testing.T) {
	calls := 0
	g, key := gatewayFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++ }))
	for _, path := range []string{"/health", "/v1/models", "/v1/chat/completions", "/v1/completions", "/v1/videos/generations", "/docs", "/openapi.json", "/props", "/slots", "/unknown"} {
		w := gatewayCall(g, "", "POST", path, strings.Repeat("x", 2<<20))
		if w.Code != 401 || w.Header().Get("Cache-Control") != "no-store" {
			t.Fatal("auth routing/body bypass", path, w.Code)
		}
	}
	for _, credential := range []string{"Bearer wrong", "Basic " + key, "Bearer " + key + ", Bearer " + key, "Bearer  " + key} {
		r := httptest.NewRequest("GET", "/health", nil)
		r.Header.Set("Authorization", credential)
		w := httptest.NewRecorder()
		g.ServeHTTP(w, r)
		if w.Code != 401 {
			t.Fatal("invalid authorization accepted")
		}
	}
	r := httptest.NewRequest("GET", "/health", nil)
	r.Header.Add("Authorization", "Bearer "+key)
	r.Header.Add("Authorization", "Bearer "+key)
	w := httptest.NewRecorder()
	g.ServeHTTP(w, r)
	if w.Code != 401 {
		t.Fatal("duplicate authorization accepted")
	}
	w = gatewayCall(g, "", "GET", "/health?api_key="+key, "")
	if w.Code != 401 {
		t.Fatal("query credentials accepted")
	}
	for _, path := range []string{"/health", "/v1/models"} {
		w = gatewayCall(g, key, "GET", path, "")
		if w.Code != 200 {
			t.Fatal("authorized metadata unavailable", w.Body.String())
		}
		if strings.Contains(w.Body.String(), key) || strings.Contains(w.Body.String(), "private-backend-key") {
			t.Fatal("credential leak")
		}
	}
	w = gatewayCall(g, key, "GET", "/props", "")
	if w.Code != 404 || calls != 0 {
		t.Fatal("private runtime API exposed")
	}
}
func TestGGUFGatewayTextContractShieldsBackendAndUsesTemplates(t *testing.T) {
	var payload map[string]any
	g, key := gatewayFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer private-backend-key" {
			t.Error("wrong backend route/credential")
		}
		if e := json.NewDecoder(r.Body).Decode(&payload); e != nil {
			t.Error(e)
		}
		w.Header().Set("X-Private", "must-not-forward")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-fixture","choices":[{"message":{"role":"assistant","content":"Paris"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10,"private_field":"must-not-forward"}}`)
	}))
	for _, chat := range []bool{false, true} {
		path, body := "/v1/completions", `{"prompt":"Capital of France?","max_tokens":12,"temperature":0,"seed":42}`
		if chat {
			path = "/v1/chat/completions"
			body = `{"messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":12}`
		}
		w := gatewayCall(g, key, "POST", path, body)
		if w.Code != 200 || !strings.Contains(w.Body.String(), "Paris") || strings.Contains(w.Body.String(), "must-not-forward") || w.Header().Get("X-Private") != "" {
			t.Fatal("invalid public response", w.Code, w.Body.String())
		}
		if payload["stream"] != false || payload["n"] != float64(1) || payload["model"] != ggufRepo || payload["max_tokens"] != float64(12) {
			t.Fatal("wrong controlled backend request", payload)
		}
		kwargs := payload["chat_template_kwargs"].(map[string]any)
		if kwargs["enable_thinking"] != false {
			t.Fatal("thinking override escaped")
		}
		messages := payload["messages"].([]any)
		if len(messages) != 1 || messages[0].(map[string]any)["role"] != "user" {
			t.Fatal("raw prompt bypassed model template")
		}
	}
}
func TestGGUFGatewayRejectsMediaToolsStreamingAndInvalidBounds(t *testing.T) {
	var calls atomic.Int32
	g, key := gatewayFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1) }))
	for _, body := range []string{
		`{"prompt":"test","stream":true}`, `{"prompt":"test","max_tokens":0}`, `{"prompt":"test","max_tokens":1025}`, `{"prompt":"test","seed":-1}`, `{"prompt":"test","seed":4294967295}`, `{"prompt":"test","temperature":3}`, `{"prompt":"test","top_p":0}`,
		`{"prompt":"test","image":"media"}`, `{"prompt":"test","tools":[]}`, `{"prompt":"test","chat_template_kwargs":{"enable_thinking":true}}`, `{"prompt":[]}`, `{"prompt":""}`, `{"prompt":"test"} {}`, `null`,
	} {
		w := gatewayCall(g, key, "POST", "/v1/completions", body)
		if w.Code != 422 {
			t.Fatal("invalid request accepted", body, w.Code)
		}
	}
	for _, body := range []string{`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com"}}]}]}`, `{"messages":[{"role":"tool","content":"x"}]}`, `{"messages":[]}`} {
		w := gatewayCall(g, key, "POST", "/v1/chat/completions", body)
		if w.Code != 422 {
			t.Fatal("media/tool message accepted", w.Code)
		}
	}
	if w := gatewayCall(g, key, "POST", "/v1/completions", `{"prompt":"test","model":"unloaded/model"}`); w.Code != 404 {
		t.Fatal("wrong model accepted")
	}
	if w := gatewayCall(g, key, "POST", "/v1/completions", strings.Repeat("x", (1<<20)+1)); w.Code != 413 {
		t.Fatal("oversized request accepted")
	}
	if calls.Load() != 0 {
		t.Fatal("invalid request reached backend")
	}
}
func TestGGUFGatewaySingleFlightSurvivesClientDisconnect(t *testing.T) {
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	g, key := gatewayFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(entered)
		<-release
		_, _ = io.WriteString(w, `{"id":"fixture","choices":[{"message":{"content":"Paris"},"finish_reason":"stop"}]}`)
	}))
	ctx, cancel := context.WithCancel(context.Background())
	r := httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"prompt":"first"}`)).WithContext(ctx)
	r.Header.Set("Authorization", "Bearer "+key)
	go func() { g.ServeHTTP(httptest.NewRecorder(), r); close(done) }()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("backend not entered")
	}
	cancel()
	w := gatewayCall(g, key, "POST", "/v1/completions", `{"prompt":"second"}`)
	if w.Code != 503 {
		t.Fatal("active GPU slot released on client disconnect")
	}
	close(release)
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("generation did not drain")
	}
	if len(g.busy) != 0 {
		t.Fatal("GPU slot leaked")
	}
}
func TestGGUFGatewayUncertainBackendRequiresRestartAndSanitizesErrors(t *testing.T) {
	g, key := gatewayFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		_, _ = io.WriteString(w, "sensitive backend path and secret")
	}))
	w := gatewayCall(g, key, "POST", "/v1/completions", `{"prompt":"first"}`)
	if w.Code != 502 || strings.Contains(w.Body.String(), "sensitive") {
		t.Fatal("backend error leaked")
	}
	g.client.Transport = roundTripFailure{}
	w = gatewayCall(g, key, "POST", "/v1/completions", `{"prompt":"second"}`)
	if w.Code != 502 || g.ready.Load() || !g.failed.Load() {
		t.Fatal("uncertain backend admitted more work")
	}
	if w = gatewayCall(g, key, "POST", "/v1/completions", `{"prompt":"third"}`); w.Code != 503 {
		t.Fatal("GPU retried after uncertain backend failure")
	}
}

type roundTripFailure struct{}

func (roundTripFailure) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("secret URL/token must not leak")
}
func TestGGUFLoopbackCanRunWithoutOwnerKeyButBackendKeyIsSeparate(t *testing.T) {
	m, h, o := ggufPlanFixture(t)
	p, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	g, e := newGGUFGateway(p, "run", "http://127.0.0.1:1", "private-backend-key")
	if e != nil {
		t.Fatal(e)
	}
	defer g.client.CloseIdleConnections()
	g.ready.Store(true)
	w := gatewayCall(g, "", "GET", "/health", "")
	if w.Code != 200 || !bytes.Contains(w.Body.Bytes(), []byte(`"run_id":"run"`)) {
		t.Fatal("loopback health failed")
	}
}

func TestGGUFWizardExplicitSelectionWithoutTorchOrEarlySideEffects(t *testing.T) {
	for _, final := range []string{"n", "y"} {
		t.Run(final, func(t *testing.T) {
			m, h, _ := ggufPlanFixture(t)
			d, calls, got := testWizardDeps(t, m)
			d.detect = func(context.Context, string) (Host, error) { return h, nil }
			d.probe = func(context.Context, string, Plan) (TorchInfo, error) {
				t.Fatal("GGUF wizard invoked PyTorch")
				return TorchInfo{}, nil
			}
			probes := 0
			d.gguf = func(_ context.Context, model *Model, file string) error {
				probes++
				if file != ggufFilename {
					t.Fatal("implicit/wrong artifact")
				}
				model.GGUF = m.GGUF
				return nil
			}
			d.createKey = func(string) error { t.Fatal("local wizard created an unrequested key"); return nil }
			input := strings.Join([]string{ggufRepo, "", ggufFilename, "", "", "", "", "", "y", "", final}, "\n") + "\n"
			var output bytes.Buffer
			e := runWizard(context.Background(), strings.NewReader(input), &output, d)
			if final == "n" {
				if e == nil || *calls != 0 {
					t.Fatal("declined GGUF deployment started")
				}
			} else if e != nil || *calls != 1 || got.GGUF == nil {
				t.Fatal("GGUF wizard failed", e, output.String())
			}
			if probes != 1 || !strings.Contains(output.String(), "--gguf-file "+ggufFilename) || strings.Contains(output.String(), "--python") {
				t.Fatal("missing explicit artifact/reproduction or unexpected Python", output.String())
			}
		})
	}
}
