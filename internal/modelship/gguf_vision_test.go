package modelship

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

// Synthetic metadata fixtures reproduce reviewed header facts, not tensors or
// inference. No test here downloads or loads a model/projector.
func visionFixture(t *testing.T) *GGUFVisionSpec {
	t.Helper()
	pin, e := ggufArtifact(ggufRepo, ggufRevision, ggufProjectorFile, true)
	if e != nil {
		t.Fatal(e)
	}
	return &GGUFVisionSpec{File: pin, Metadata: GGUFMetadata{Architecture: "clip", TensorCount: 334, TensorBytes: 927587264, HeaderBytes: 20224, Types: map[uint32]int{0: 222, 1: 112}, Attributes: map[string]string{"general.type": "mmproj", "clip.projector_type": "qwen3vl_merger", "clip.has_vision_encoder": "1", "clip.vision.projection_dim": "5120", "clip.vision.patch_size": "16", "clip.vision.spatial_merge_size": "2"}}}
}
func visionURL(t *testing.T, mime string) string {
	t.Helper()
	im := image.NewRGBA(image.Rect(0, 0, 2, 2))
	im.Set(0, 0, color.RGBA{255, 0, 0, 255})
	var b bytes.Buffer
	if mime == "image/png" {
		if e := png.Encode(&b, im); e != nil {
			t.Fatal(e)
		}
	} else {
		if e := jpeg.Encode(&b, im, nil); e != nil {
			t.Fatal(e)
		}
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(b.Bytes())
}
func visionBody(url string) string {
	b, _ := json.Marshal(map[string]any{"messages": []any{map[string]any{"role": "user", "content": []any{map[string]string{"type": "text", "text": "Describe this image."}, map[string]any{"type": "image_url", "image_url": map[string]string{"url": url}}}}}})
	return string(b)
}
func TestGGUFReviewedArtifactCatalogAndCodecSizes(t *testing.T) {
	c, e := loadGGUFArtifacts()
	if e != nil || len(c.Models) < 20 || len(c.Projectors) != 1 {
		t.Fatal(e, c)
	}
	for _, p := range c.Models {
		got, e := ggufTarget(c.Repo, c.Revision, p.Name)
		if e != nil || got != p {
			t.Fatal(e, p)
		}
	}
	for _, file := range []string{"BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", "Qwen3.8-27B-MTP.gguf", "mmproj-F16.gguf", "unknown.gguf"} {
		if _, e := ggufTarget(c.Repo, c.Revision, file); e == nil {
			t.Fatal("non-language artifact accepted", file)
		}
	}
	for typ, block := range ggufBlocks {
		header, _ := ggufHeaderFixture(256, typ)
		if _, e := parseHeaderFixture(header, int64(len(header))+256/block[0]*block[1]); e != nil {
			t.Fatal(typ, e)
		}
	}
}
func TestGGUFVisionCombinedBudgetsAndExplicitPairing(t *testing.T) {
	m, h, o := ggufPlanFixture(t)
	m.GGUF.Runtime = llamaJammyRuntime
	m.GGUF.Vision = visionFixture(t)
	o.Vision = true
	o.MMProjFile = ggufProjectorFile
	p, e := BuildPlan(m, h, o)
	if e != nil || len(p.Blockers) > 0 {
		t.Fatal(e, p.Blockers)
	}
	report, e := inspectModel(m)
	if e != nil || report.SelectedWeightBytes != p.WeightBytes {
		t.Fatal("inspection excludes selected projector bytes", e, report.SelectedWeightBytes)
	}
	if len(p.Files) != 2 || p.WeightBytes != m.GGUF.File.Size+m.GGUF.Vision.File.Size || p.WeightResidentBytes != m.GGUF.Metadata.TensorBytes+m.GGUF.Vision.Metadata.TensorBytes {
		t.Fatal("combined artifact budget missing", p)
	}
	o.AllowRisk = true
	h.GPUs[0].FreeMiB = (m.GGUF.Metadata.TensorBytes + (1 << 20) - 1) / (1 << 20)
	p, e = BuildPlan(m, h, o)
	if e != nil || p.WeightsFitFree || len(p.Blockers) == 0 {
		t.Fatal("projector physical budget bypassed", e, p.Blockers)
	}
	for _, name := range []string{"no-optin", "no-projector", "wrong-projector", "wrong-shape", "wrong-runtime"} {
		t.Run(name, func(t *testing.T) {
			mm := copyWizardModel(m)
			oo := o
			switch name {
			case "no-optin":
				oo.Vision = false
			case "no-projector":
				oo.MMProjFile = ""
			case "wrong-projector":
				oo.MMProjFile = "mmproj-BF16.gguf"
			case "wrong-shape":
				mm.GGUF.Vision.Metadata.Attributes = map[string]string{}
			case "wrong-runtime":
				mm.GGUF.Runtime = llamaLegacyRuntime
			}
			if _, e := BuildPlan(mm, h, oo); e == nil {
				t.Fatal("unsafe pairing accepted")
			}
		})
	}
}
func TestGGUFVisionArtifactCacheIsSeparate(t *testing.T) {
	m, h, _ := ggufPlanFixture(t)
	m.GGUF.Vision = visionFixture(t)
	t.Setenv("HF_HUB_CACHE", h.ModelCacheDir)
	m.Hub.Siblings = append(m.Hub.Siblings, ModelFile{Name: ggufProjectorFile, Size: m.GGUF.Vision.File.Size})
	InspectModelCache(&h, m)
	if h.CachedWeightBytes != 0 {
		t.Fatal("projector excluded from cache/disk budget", h.CachedWeightBytes)
	}
	path := ggufArtifactPath(Plan{ModelCacheDir: h.ModelCacheDir, Model: m.Hub.ID, Revision: m.Hub.SHA}, ggufProjectorFile)
	if e := privateDir(strings.TrimSuffix(path, "/"+ggufProjectorFile)); e != nil {
		t.Fatal(e)
	}
	f, e := os.Create(path)
	if e != nil {
		t.Fatal(e)
	}
	if e = f.Truncate(m.GGUF.Vision.File.Size); e != nil {
		t.Fatal(e)
	}
	f.Close()
	InspectModelCache(&h, m)
	if h.CachedWeightBytes != m.GGUF.Vision.File.Size {
		t.Fatal("projector and language cache paths mixed", h.CachedWeightBytes)
	}
}
func TestGGUFVisionImageContractAndCapabilities(t *testing.T) {
	calls := 0
	g, key := gatewayFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body map[string]any
		if e := json.NewDecoder(r.Body).Decode(&body); e != nil {
			t.Error(e)
		}
		if !strings.Contains(string(mustJSON(body)), "data:image/") {
			t.Error("image stripped from controlled backend request")
		}
		io.WriteString(w, `{"choices":[{"message":{"content":"A red pixel."},"finish_reason":"stop"}]}`)
	}))
	g.plan.GGUF.Vision = visionFixture(t)
	if w := gatewayCall(g, "", "GET", "/v1/capabilities", ""); w.Code != 401 {
		t.Fatal("capabilities unauthenticated")
	}
	w := gatewayCall(g, key, "GET", "/v1/capabilities", "")
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"image_input":true`) || !strings.Contains(w.Body.String(), `"max_image_tokens":1024`) {
		t.Fatal(w.Body.String())
	}
	for _, mime := range []string{"image/png", "image/jpeg"} {
		w = gatewayCall(g, key, "POST", "/v1/chat/completions", visionBody(visionURL(t, mime)))
		if w.Code != 200 {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	before := calls
	for _, url := range []string{"https://example.com/image.png", "http://169.254.169.254/metadata", "file:///etc/passwd", "data:image/gif;base64,R0lGODlh", "data:image/png;base64,invalid!", strings.Replace(visionURL(t, "image/png"), "image/png", "image/jpeg", 1)} {
		w = gatewayCall(g, key, "POST", "/v1/chat/completions", visionBody(url))
		if w.Code != 422 {
			t.Fatal("invalid image reached backend", w.Code, w.Body.String())
		}
	}
	body := visionBody(visionURL(t, "image/png"))
	body = strings.Replace(body, `"role":"user"`, `"role":"assistant"`, 1)
	if w = gatewayCall(g, key, "POST", "/v1/chat/completions", body); w.Code != 422 {
		t.Fatal("assistant image accepted")
	}
	if calls != before {
		t.Fatal("invalid images reached backend")
	}
	g.plan.GGUF.Vision = nil
	if w = gatewayCall(g, key, "POST", "/v1/chat/completions", visionBody(visionURL(t, "image/png"))); w.Code != 422 {
		t.Fatal("text-only deployment accepted image")
	}
}
func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }
func TestGGUFVisionLocalArgsOnlyAndStateRoundTrip(t *testing.T) {
	m, h, o := ggufPlanFixture(t)
	m.GGUF.Runtime = llamaJammyRuntime
	m.GGUF.Vision = visionFixture(t)
	o.Vision = true
	o.MMProjFile = ggufProjectorFile
	p, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	args := llamaArgs(p, 18001, "/private/key")
	joined := strings.Join(args, " ")
	for _, value := range []string{"--mmproj " + ggufArtifactPath(p, ggufProjectorFile), "--mmproj-device CUDA0", "--mmproj-offload", "--image-max-tokens 1024", "--mtmd-batch-max-tokens 256"} {
		if !strings.Contains(joined, value) {
			t.Fatal("missing explicit vision policy", value)
		}
	}
	if contains(args, "--no-mmproj") || contains(args, "--mmproj-url") {
		t.Fatal("vision disabled or remote loading enabled")
	}
	b, _ := json.Marshal(p)
	var saved Plan
	if e = json.Unmarshal(b, &saved); e != nil {
		t.Fatal(e)
	}
	if e = validateGGUFPlan(saved); e != nil {
		t.Fatal("vision state lost typed metadata", e)
	}
}
func TestWizardEscBackReusesBufferedInputAndDiscardsApproval(t *testing.T) {
	d, calls, got := testWizardDeps(t, textFixture())
	inspects := 0
	original := d.inspect
	d.inspect = func(c context.Context, id, rev string) (Model, error) { inspects++; return original(c, id, rev) }
	input := []string{"org/model", "", "a", "old-name", ":back", "new-name", "", "", "", "a", "", "", ":back", "", "y"}
	var out bytes.Buffer
	if e := runWizard(context.Background(), strings.NewReader(strings.Join(input, "\n")+"\n"), &out, d); e != nil {
		t.Fatal(e, out.String())
	}
	if *calls != 1 || got.Name != "new-name" || inspects != 1 {
		t.Fatal("back lost input, repeated metadata, or kept stale selection", *calls, got.Name, inspects, out.String())
	}
	if !strings.Contains(out.String(), "Later selections/approvals were discarded") {
		t.Fatal("back not explained")
	}
}

func TestWizardGGUFDefaultQuantAndNumericVisionMenu(t *testing.T) {
	m, h, _ := ggufPlanFixture(t)
	m.GGUF.Runtime = llamaJammyRuntime
	pin, _ := ggufArtifact(ggufRepo, ggufRevision, ggufProjectorFile, true)
	m.Hub.Siblings = append(m.Hub.Siblings, ModelFile{Name: pin.Name, Size: pin.Size})
	d, calls, got := testWizardDeps(t, m)
	d.detect = func(context.Context, string) (Host, error) { return h, nil }
	d.gguf = func(_ context.Context, mm *Model, file string) error {
		if file != ggufFilename {
			t.Fatal("hidden quantization substitution", file)
		}
		return nil
	}
	d.vision = func(_ context.Context, mm *Model, file string) error {
		if file != ggufProjectorFile {
			t.Fatal(file)
		}
		mm.GGUF.Vision = visionFixture(t)
		return nil
	}
	d.probe = func(context.Context, string, Plan) (TorchInfo, error) {
		t.Fatal("GGUF invoked Python probe")
		return TorchInfo{}, nil
	}
	input := []string{ggufRepo, "", "", "2", "", "", "", "1", "", "y", "", "y"}
	var out bytes.Buffer
	if e := runWizard(context.Background(), strings.NewReader(strings.Join(input, "\n")+"\n"), &out, d); e != nil {
		t.Fatal(e, out.String())
	}
	if *calls != 1 || got.GGUF == nil || got.GGUF.Vision == nil || len(got.Files) != 2 {
		t.Fatal("vision/default selection not deployed exactly once", out.String())
	}
	for _, hint := range []string{"(default)", "15.43 GiB download", "--vision --mmproj-file mmproj-F16.gguf", "1 / a)", "2 / b)"} {
		if !strings.Contains(out.String(), hint) {
			t.Fatal("missing guided selection/reproduction", hint, out.String())
		}
	}
}
