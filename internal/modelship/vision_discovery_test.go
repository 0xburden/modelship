package modelship

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// Exercise the real Hub inspector, not a stub that copies the input spelling.
// All responses are small local metadata fixtures; no weights/GPU/install work.
func visionDiscoveryHub(t *testing.T, requested, reported string, m Model) (*Hub, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var paths []string
	canonical := requested
	if id, e := ParseModel(reported); e == nil && id == reported && strings.EqualFold(id, requested) {
		canonical = id
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasPrefix(r.URL.Path, "/api/models/"+requested+"/revision/"):
			if r.URL.Query().Get("blobs") != "true" {
				t.Error("metadata sizes not requested")
			}
			metadata := m.Hub
			// Planner fixtures retain weights only; real Hub metadata also lists
			// the config needed by the shared inspection path.
			metadata.Siblings = append(append([]ModelFile(nil), metadata.Siblings...), ModelFile{Name: "config.json"})
			metadata.ID = reported
			json.NewEncoder(w).Encode(metadata)
		case r.URL.Path == "/"+canonical+"/resolve/"+m.Hub.SHA+"/config.json":
			json.NewEncoder(w).Encode(m.Config)
		default:
			t.Error("unexpected request (must be metadata only)", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return &Hub{BaseURL: srv.URL, Client: srv.Client()}, func() []string { mu.Lock(); defer mu.Unlock(); return append([]string(nil), paths...) }
}

func TestHubCanonicalizesOnlyValidatedCasingAliases(t *testing.T) {
	for _, tc := range []struct{ name, requested, reported, want string }{
		{"lowercase", "google/gemma-4-e4b-it", gemmaVisionRepo, gemmaVisionRepo},
		{"mixed-case", "GOOGLE/GEMMA-4-e4b-IT", gemmaVisionRepo, gemmaVisionRepo},
		{"canonical", gemmaVisionRepo, gemmaVisionRepo, gemmaVisionRepo},
		{"other-repository", "org/model", "org/other", "org/model"},
		{"other-owner", "org/model", "other/model", "org/model"},
		{"missing-id", "org/model", "", "org/model"},
		{"not-an-id", "org/model", "https://huggingface.co/org/model", "org/model"},
		{"tree-path", "org/model", "org/model/tree/main", "org/model"},
		{"credentials", "org/model", "org/model?token=fixture", "org/model"},
		{"terminal-control", "org/model", "org/model\x1b[31m", "org/model"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := targetFixture(t, gemmaVisionRepo)
			hub, requests := visionDiscoveryHub(t, tc.requested, tc.reported, fixture)
			m, e := hub.Inspect(context.Background(), tc.requested, "main")
			if e != nil {
				t.Fatal(e)
			}
			paths := requests()
			if m.Hub.ID != tc.want || m.Hub.SHA != gemmaVisionRevision || len(paths) != 2 || paths[1] != "/"+tc.want+"/resolve/"+gemmaVisionRevision+"/config.json" {
				t.Fatal("repository/revision changed beyond casing", m.Hub.ID, m.Hub.SHA, paths)
			}
		})
	}
}

func TestGemmaVisionWizardAcceptsDirectURLCasingAndFloatingMain(t *testing.T) {
	for _, tc := range []struct{ input, revision string }{
		{"https://huggingface.co/google/gemma-4-e4b-it/", ""},
		{"https://huggingface.co/GOOGLE/GEMMA-4-e4b-IT", "main"},
		{"https://huggingface.co/" + gemmaVisionRepo, "main"},
		{"google/gemma-4-e4b-it", gemmaVisionRevision},
	} {
		t.Run(tc.input+"@"+tc.revision, func(t *testing.T) {
			requested, e := ParseModel(tc.input)
			if e != nil {
				t.Fatal(e)
			}
			fixture := targetFixture(t, gemmaVisionRepo)
			hub, requests := visionDiscoveryHub(t, requested, gemmaVisionRepo, fixture)
			d, calls, got := testWizardDeps(t, fixture)
			d.inspect = hub.Inspect
			d.detect = func(context.Context, string) (Host, error) {
				h := hostFixture()
				h.GPUs[0].TotalMiB = 49152
				h.GPUs[0].FreeMiB = 48000
				return h, nil
			}
			input := []string{tc.input, tc.revision, "vision-images", "", "", "", "", "1", "", "y", "", "y"}
			var out strings.Builder
			if e = runWizard(context.Background(), strings.NewReader(strings.Join(input, "\n")+"\n"), &out, d); e != nil {
				t.Fatal(e, out.String())
			}
			if *calls != 1 || got.Vision == nil || got.Model != gemmaVisionRepo || got.Revision != gemmaVisionRevision || got.Vision.Repo != got.Model || !strings.Contains(visionWizardRepro(out.String(), d.executable), " --vision") || !strings.Contains(out.String(), "Input mode") {
				t.Fatal("direct URL lost reviewed image profile", got, out.String())
			}
			if requested != gemmaVisionRepo && !strings.Contains(out.String(), "Hub canonical spelling") {
				t.Fatal("casing normalization was not explained", out.String())
			}
			expected := tc.revision
			if expected == "" {
				expected = gemmaVisionRevision
			}
			paths := requests()
			if len(paths) != 2 || !strings.HasSuffix(paths[0], "/revision/"+expected) {
				t.Fatal("revision silently substituted", paths)
			}
		})
	}
}

func TestGemmaWizardExplainsUnavailableVisionWithoutSubstitution(t *testing.T) {
	for _, tc := range []struct {
		name, requested, reported string
		change                    func(*Model)
	}{
		{"unreviewed-revision", gemmaVisionRepo, gemmaVisionRepo, func(m *Model) { m.Hub.SHA = strings.Repeat("b", 40) }},
		{"different-official-size", "google/gemma-4-E2B-it", "google/gemma-4-E2B-it", func(*Model) {}},
		{"community-variant", "community/gemma-variant", "community/gemma-variant", func(*Model) {}},
		{"foreign-metadata-cannot-grant-profile", "google/gemma-4-e4b-it", "other/model", func(*Model) {}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := targetFixture(t, gemmaVisionRepo)
			tc.change(&m)
			hub, _ := visionDiscoveryHub(t, tc.requested, tc.reported, m)
			d, calls, _ := testWizardDeps(t, m)
			d.inspect = hub.Inspect
			stopped := errors.New("stop before hardware detection")
			d.detect = func(context.Context, string) (Host, error) { return Host{}, stopped }
			var out strings.Builder
			e := runWizard(context.Background(), strings.NewReader(tc.requested+"\nmain\n"), &out, d)
			if !errors.Is(e, stopped) || *calls != 0 {
				t.Fatal("unexpected side effect", e, *calls, out.String())
			}
			transcript := out.String()
			if !strings.Contains(transcript, "Image input unavailable") || !strings.Contains(transcript, tc.requested) || !strings.Contains(transcript, m.Hub.SHA) || !strings.Contains(transcript, "No model/revision is substituted") || strings.Contains(transcript, "Input mode (") {
				t.Fatal("missing explicit no-vision reason", transcript)
			}
		})
	}
}

func TestGemmaURLVisionCancellationBeforeApprovalHasNoSideEffects(t *testing.T) {
	m := targetFixture(t, gemmaVisionRepo)
	hub, _ := visionDiscoveryHub(t, "google/gemma-4-e4b-it", gemmaVisionRepo, m)
	d, calls, _ := testWizardDeps(t, m)
	d.inspect = hub.Inspect
	probes, keys, hardware := 0, 0, 0
	d.detect = func(context.Context, string) (Host, error) {
		hardware++
		return Host{}, errors.New("unexpected hardware")
	}
	d.probe = func(context.Context, string, Plan) (TorchInfo, error) { probes++; return TorchInfo{}, nil }
	d.createKey = func(string) error { keys++; return nil }
	var out strings.Builder
	// EOF while choosing the now-visible image mode, before even GPU detection.
	e := runWizard(context.Background(), strings.NewReader("https://huggingface.co/google/gemma-4-e4b-it/\nmain\n"), &out, d)
	if e == nil || *calls != 0 || keys != 0 || probes != 0 || hardware != 0 || !strings.Contains(out.String(), "Input mode") {
		t.Fatal("canceled menu caused side effects", e, out.String())
	}
}

func visionWizardRepro(output, executable string) string {
	for _, line := range strings.Split(output, "\n") {
		if strings.HasPrefix(line, shellWord(executable)+" deploy ") {
			return line
		}
	}
	return ""
}

func TestGemmaURLWizardCanonicalizationDoesNotEnableVisionByDefault(t *testing.T) {
	fixture := targetFixture(t, gemmaVisionRepo)
	hub, _ := visionDiscoveryHub(t, "google/gemma-4-e4b-it", gemmaVisionRepo, fixture)
	d, calls, got := testWizardDeps(t, fixture)
	d.inspect = hub.Inspect
	d.detect = func(context.Context, string) (Host, error) {
		h := hostFixture()
		h.GPUs[0].TotalMiB = 49152
		h.GPUs[0].FreeMiB = 48000
		return h, nil
	}
	input := []string{"https://huggingface.co/google/gemma-4-e4b-it/", "main", "", "", "", "", "", "1", "", "y", "", "y"}
	var out strings.Builder
	if e := runWizard(context.Background(), strings.NewReader(strings.Join(input, "\n")+"\n"), &out, d); e != nil {
		t.Fatal(e, out.String())
	}
	if *calls != 1 || got.Vision != nil || got.Model != gemmaVisionRepo || got.Revision != gemmaVisionRevision || strings.Contains(visionWizardRepro(out.String(), d.executable), " --vision") || !strings.Contains(out.String(), "text-only (default)") {
		t.Fatal("case-only URL implicitly enabled image execution", got, out.String())
	}
}
