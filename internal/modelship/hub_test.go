package modelship

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseModel(t *testing.T) {
	for _, input := range []string{"org/model", "https://huggingface.co/org/model", "https://huggingface.co/org/model/", "gpt2"} {
		if _, err := ParseModel(input); err != nil {
			t.Errorf("%s: %v", input, err)
		}
	}
	for _, input := range []string{"https://evil.example/org/model", "https://huggingface.co.evil.example/org/model", "http://huggingface.co/org/model", "https://huggingface.co/org/model?token=foo", "org/model/tree/main", "../model", "org/../../x", "--model", "org/model.git", "https://user@huggingface.co/org/model"} {
		if _, err := ParseModel(input); err == nil {
			t.Errorf("accepted %q", input)
		}
	}
}
func TestHubPinnedConfigAndToken(t *testing.T) {
	sha := strings.Repeat("a", 40)
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Error("missing auth")
		}
		switch r.URL.Path {
		case "/api/models/org/model/revision/main":
			if r.URL.Query().Get("blobs") != "true" {
				t.Error("missing sizes")
			}
			fmt.Fprintf(w, `{"id":"malicious/metadata","sha":%q,"siblings":[{"rfilename":"config.json"},{"rfilename":"model.safetensors","lfs":{"size":100}}]}`, sha)
		case "/org/model/resolve/" + sha + "/config.json":
			fmt.Fprint(w, `{"architectures":["LlamaForCausalLM"]}`)
		default:
			t.Error("unexpected path", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	hub := &Hub{BaseURL: srv.URL, Token: "secret", Client: srv.Client()}
	m, err := hub.Inspect(context.Background(), "org/model", "main")
	if err != nil || calls != 2 || m.Hub.ID != "org/model" || m.Hub.Siblings[1].Bytes() != 100 {
		t.Fatalf("%+v %v calls=%d", m, err, calls)
	}
}
func TestHubFailures(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		status     int
	}{
		{"gated", "token should not be printed", 403},
		{"missing", "", 404},
		{"bad-json", "oops", 200},
		{"unversioned", `{"sha":"main"}`, 200},
		{"adapter", `{"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","siblings":[{"rfilename":"adapter_config.json"}]}`, 200},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(tc.status); fmt.Fprint(w, tc.body) }))
			defer srv.Close()
			h := &Hub{BaseURL: srv.URL, Client: srv.Client()}
			_, err := h.Inspect(context.Background(), "org/model", "main")
			if err == nil {
				t.Fatal("expected failure")
			}
			if strings.Contains(err.Error(), "token should not be printed") {
				t.Fatal("response body leaked")
			}
		})
	}
}
func TestDriverMinimum(t *testing.T) {
	for _, s := range []string{"570.26", "570.26.01", "575.57.08"} {
		if !driverAtLeast(s, 570, 26) {
			t.Fatal(s)
		}
	}
	for _, s := range []string{"550.99", "570.25", "N/A", "570"} {
		if driverAtLeast(s, 570, 26) {
			t.Fatal(s)
		}
	}
}
