package modelship

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"image"
	"image/png"
	"io"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func ggufImageMessage(urls ...string) ggufMessage {
	parts := []any{map[string]string{"type": "text", "text": "Compare these in order."}}
	for _, url := range urls {
		parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]string{"url": url}})
	}
	return ggufMessage{Role: "user", Content: mustJSON(parts)}
}
func ggufImagesBody(messages ...ggufMessage) string {
	return string(mustJSON(map[string]any{"messages": messages}))
}
func ggufPNGChunk(kind string, payload []byte) []byte {
	b := make([]byte, 12+len(payload))
	binary.BigEndian.PutUint32(b[:4], uint32(len(payload)))
	copy(b[4:8], kind)
	copy(b[8:], payload)
	binary.BigEndian.PutUint32(b[len(b)-4:], crc32.ChecksumIEEE(b[4:len(b)-4]))
	return b
}
func ggufPNGURL(b []byte) string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(b)
}
func ggufPNGBytes(t *testing.T, width, height int) []byte {
	t.Helper()
	var b bytes.Buffer
	if e := png.Encode(&b, image.NewGray(image.Rect(0, 0, width, height))); e != nil {
		t.Fatal(e)
	}
	return b.Bytes()
}
func ggufPaddedPNG(t *testing.T, size int) string {
	t.Helper()
	b := ggufPNGBytes(t, 2, 2)
	// A valid ancillary chunk adds encoded file bytes without decoded pixels.
	chunk := ggufPNGChunk("tEXt", append([]byte("padding\x00"), bytes.Repeat([]byte("x"), size-len(b)-20)...))
	return ggufPNGURL(append(append(b[:len(b)-12:len(b)-12], chunk...), b[len(b)-12:]...))
}

func TestGGUFNewVisionPlanSavesPolicyWithoutUpgradingLegacy(t *testing.T) {
	m, h, o := ggufPlanFixture(t)
	m.GGUF.Runtime, m.GGUF.Vision = llamaJammyRuntime, visionFixture(t)
	o.Vision, o.MMProjFile = true, ggufProjectorFile
	p, e := BuildPlan(m, h, o)
	if e != nil || !reviewedGGUFImages(p.GGUF.Vision.Images) || m.GGUF.Vision.Images != nil {
		t.Fatal("new policy missing or inspected metadata mutated", e)
	}
	if !strings.Contains(strings.Join(p.Warnings, " "), "extra 4 GiB activation allowance is an unmeasured heuristic") {
		t.Fatal("multi-image unmeasured budget not identified", p.Warnings)
	}
	textModel := m
	textSpec := *m.GGUF
	textSpec.Vision = nil
	textModel.GGUF = &textSpec
	textOptions := o
	textOptions.Vision, textOptions.MMProjFile = false, ""
	textPlan, e := BuildPlan(textModel, h, textOptions)
	projectorMiB := (m.GGUF.Vision.Metadata.TensorBytes + (1 << 20) - 1) / (1 << 20)
	// Rounding the combined resident budget can differ by one MiB.
	delta := p.EstimatedVRAMMiB - textPlan.EstimatedVRAMMiB
	if e != nil || delta < projectorMiB+4095 || delta > projectorMiB+4096 {
		t.Fatal("new vision activation heuristic is not +4 GiB", delta, e)
	}
	for _, legacy := range []bool{false, true} {
		if legacy {
			p.GGUF.Vision.Images = nil
		}
		var saved Plan
		if e := json.Unmarshal(mustJSON(p), &saved); e != nil {
			t.Fatal(e)
		}
		if e := validateGGUFPlan(saved); e != nil || (saved.GGUF.Vision.Images == nil) != legacy {
			t.Fatal("saved contract changed", legacy, e)
		}
		g := &ggufGateway{plan: saved}
		chat := g.capabilities()["chat"].(map[string]any)
		_, totals := chat["max_total_image_bytes"]
		if legacy && (chat["max_images"] != 1 || totals) || !legacy && (chat["max_images"] != 4 || !totals) {
			t.Fatal("capabilities did not follow saved policy", chat)
		}
	}
	for _, policy := range []ImagePolicy{{}, {MaxImages: 5, MaxTotalBytes: 8 << 20, MaxTotalPixels: 8 * 1024 * 1024}, {MaxImages: 4, MaxTotalBytes: 9 << 20, MaxTotalPixels: 8 * 1024 * 1024}, {MaxImages: 4, MaxTotalBytes: 8 << 20, MaxTotalPixels: 9 * 1024 * 1024}} {
		v := visionFixture(t)
		v.Images = &policy
		if validateGGUFVision(v) == nil {
			t.Fatal("non-reviewed saved policy accepted", policy)
		}
		p.GGUF.Vision = v
		chat := (&ggufGateway{plan: p}).capabilities()["chat"].(map[string]any)
		if chat["max_images"] != 1 || chat["max_total_image_bytes"] != nil {
			t.Fatal("unreviewed policy advertised multi-image", chat)
		}
	}
}

func TestGGUFMultiImageConversationCountOrderingAndLegacy(t *testing.T) {
	policy := reviewedImagePolicy()
	pngURL, jpegURL := visionURL(t, "image/png"), visionURL(t, "image/jpeg")
	messages := []ggufMessage{
		{Role: "system", Content: mustJSON("Be precise.")},
		ggufImageMessage(pngURL, jpegURL),
		{Role: "assistant", Content: mustJSON("First pair.")},
		ggufImageMessage(jpegURL, pngURL),
	}
	got, e := validateGGUFMessagesWithPolicy(messages, true, &policy)
	if e != nil {
		t.Fatal(e)
	}
	var wantJSON, gotJSON any
	_ = json.Unmarshal(mustJSON(messages), &wantJSON)
	_ = json.Unmarshal(mustJSON(got), &gotJSON)
	if !reflect.DeepEqual(wantJSON, gotJSON) {
		t.Fatal("normalization changed roles, turns, part order or URLs", string(mustJSON(got)))
	}
	if _, e := validateGGUFMessages(messages, true); e == nil {
		t.Fatal("nil-policy legacy helper accepted multiple images")
	}
	if _, e := validateGGUFMessages([]ggufMessage{ggufImageMessage(pngURL)}, true); e != nil {
		t.Fatal("legacy single-image request changed", e)
	}
	if _, e := validateGGUFMessagesWithPolicy([]ggufMessage{ggufImageMessage(pngURL, jpegURL, pngURL, jpegURL)}, true, &policy); e != nil {
		t.Fatal("four images in one turn rejected", e)
	}
	messages = append(messages, ggufImageMessage(pngURL))
	if _, e := validateGGUFMessagesWithPolicy(messages, true, &policy); e == nil || !strings.Contains(e.Error(), "at most 4") {
		t.Fatal("conversation count not enforced", e)
	}
	// Count admission precedes even URL inspection or decoding of earlier images.
	bad := []ggufMessage{ggufImageMessage("invalid", "invalid", "invalid", "invalid"), ggufImageMessage("invalid")}
	if _, e := validateGGUFMessagesWithPolicy(bad, true, &policy); e == nil || !strings.Contains(e.Error(), "at most 4") {
		t.Fatal("count was not admitted before decoding", e)
	}
}

func TestGGUFMultiImageAggregateAdmissionBeforePixelDecode(t *testing.T) {
	policy := reviewedImagePolicy()
	// Valid PNG container/header but deliberately invalid compressed pixels.
	// Header dimensions are inspected before heavy pixel decode across all turns.
	b := ggufPNGBytes(t, 2, 2)
	binary.BigEndian.PutUint32(b[16:20], 1536)
	binary.BigEndian.PutUint32(b[20:24], 1536)
	binary.BigEndian.PutUint32(b[29:33], crc32.ChecksumIEEE(b[12:29]))
	url := ggufPNGURL(b)
	messages := []ggufMessage{ggufImageMessage(url, url), ggufImageMessage(url, url)}
	if _, e := validateGGUFMessagesWithPolicy(messages, true, &policy); e == nil || !strings.Contains(e.Error(), "aggregate") {
		t.Fatal("pixel aggregate not rejected before full decode", e)
	}
	if _, e := validateGGUFMessagesWithPolicy(messages[:1], true, &policy); e == nil || !strings.Contains(e.Error(), "decoded safely") {
		t.Fatal("valid budgets bypassed full image decode", e)
	}
	large := ggufPaddedPNG(t, (3<<20)-4096)
	messages = []ggufMessage{ggufImageMessage(large), ggufImageMessage(large), ggufImageMessage(large)}
	if _, e := validateGGUFMessagesWithPolicy(messages, true, &policy); e == nil || !strings.Contains(e.Error(), "aggregate") {
		t.Fatal("encoded image byte aggregate not rejected across turns", e)
	}
	boundary := ggufPNGURL(ggufPNGBytes(t, 2048, 2048))
	if _, e := validateGGUFMessagesWithPolicy([]ggufMessage{ggufImageMessage(boundary, boundary)}, true, &policy); e != nil {
		t.Fatal("exact 8-megapixel aggregate rejected", e)
	}
}

func TestGGUFMultiImageStaticPNGAndPerImageAdmission(t *testing.T) {
	b := ggufPNGBytes(t, 2, 2)
	for _, kind := range []string{"acTL", "fcTL", "fdAT"} {
		animated := append(append(append([]byte(nil), b[:33]...), ggufPNGChunk(kind, make([]byte, 8))...), b[33:]...)
		if _, e := checkedVisionDataURL(ggufPNGURL(animated)); e == nil || !strings.Contains(e.Error(), "animated PNG") {
			t.Fatal("APNG chunk passed static image contract", kind, e)
		}
	}
	for _, size := range [][2]int{{4097, 1}, {2049, 2048}} {
		url := ggufPNGURL(ggufPNGBytes(t, size[0], size[1]))
		if _, e := checkedVisionDataURL(url); e == nil || !strings.Contains(e.Error(), "dimensions") {
			t.Fatal("per-image dimension/pixel bound missing", size, e)
		}
	}
	for _, invalid := range [][]byte{append(append([]byte(nil), b...), 'x'), b[:len(b)-1]} {
		if _, e := checkedVisionDataURL(ggufPNGURL(invalid)); e == nil {
			t.Fatal("incomplete or trailing PNG accepted")
		}
	}
	if _, e := checkedVisionDataURL(ggufPaddedPNG(t, visionMaxBytes+1)); e == nil {
		t.Fatal("per-image encoded byte cap missing")
	}
}

func TestGGUFMultiImageGatewayForwardsOrderedConversation(t *testing.T) {
	pngURL, jpegURL := visionURL(t, "image/png"), visionURL(t, "image/jpeg")
	messages := []ggufMessage{ggufImageMessage(pngURL, jpegURL), {Role: "assistant", Content: mustJSON("First pair.")}, ggufImageMessage(jpegURL, pngURL)}
	calls := 0
	g, key := gatewayFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var request struct {
			Messages []ggufMessage `json:"messages"`
		}
		if e := json.NewDecoder(r.Body).Decode(&request); e != nil || !reflect.DeepEqual(request.Messages, messages) {
			t.Error("child did not receive ordered conversation", request.Messages, e)
		}
		io.WriteString(w, `{"choices":[{"message":{"content":"Four images in order."},"finish_reason":"stop"}]}`)
	}))
	g.plan.GGUF.Vision = visionFixture(t)
	policy := reviewedImagePolicy()
	g.plan.GGUF.Vision.Images = &policy
	if w := gatewayCall(g, key, "POST", "/v1/chat/completions", ggufImagesBody(messages...)); w.Code != 200 || calls != 1 {
		t.Fatal("valid four-image request did not reach child once", w.Code, w.Body.String(), calls)
	}
}

func TestGGUFMultiImageGatewayChildAdmissionAndBodyLimits(t *testing.T) {
	calls := 0
	g, key := gatewayFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		// CPU fake child: emulate the reviewed runtime's expanded-context error.
		// This tests gateway propagation, not real runtime tokenization/inference.
		if r.Header.Get("X-Fake") != "" {
			t.Error("unexpected client header propagation")
		}
		w.WriteHeader(400)
		io.WriteString(w, `{"error":{"message":"expanded prompt exceeds context; private path"}}`)
	}))
	g.plan.GGUF.Vision = visionFixture(t)
	url := visionURL(t, "image/png")
	body := ggufImagesBody(ggufImageMessage(url, url, url, url))
	if w := gatewayCall(g, key, "POST", "/v1/chat/completions", body); w.Code != 422 || calls != 0 {
		t.Fatal("legacy image count reached child", w.Code, calls)
	}
	policy := reviewedImagePolicy()
	g.plan.GGUF.Vision.Images = &policy
	if w := gatewayCall(g, key, "POST", "/v1/chat/completions", body); w.Code != 422 || calls != 1 || strings.Contains(w.Body.String(), "private path") || !g.ready.Load() {
		t.Fatal("child context rejection was not sanitized/recoverable", w.Code, w.Body.String(), calls)
	}
	for _, messages := range [][]ggufMessage{
		{ggufImageMessage(url, url, url, url), ggufImageMessage(url)},
		{ggufImageMessage("https://example.com/image.png")},
		{{Role: "user", Content: mustJSON([]any{map[string]string{"type": "audio", "text": "x"}})}},
	} {
		if w := gatewayCall(g, key, "POST", "/v1/chat/completions", ggufImagesBody(messages...)); w.Code != 422 || calls != 1 {
			t.Fatal("invalid image request reached child", w.Code, calls)
		}
	}
	large := ggufPaddedPNG(t, (3<<20)-4096)
	aggregateBody := ggufImagesBody(ggufImageMessage(large), ggufImageMessage(large), ggufImageMessage(large))
	if len(aggregateBody) > 12<<20 {
		t.Fatal("fixture exceeds body limit instead of aggregate")
	}
	if w := gatewayCall(g, key, "POST", "/v1/chat/completions", aggregateBody); w.Code != 422 || calls != 1 || !strings.Contains(w.Body.String(), "aggregate") {
		t.Fatal("aggregate admission reached child", w.Code, w.Body.String(), calls)
	}
	// Two individually valid images below 8 MiB encoded bytes need >8 MiB of
	// JSON/base64. Only the explicit policy gets the 12 MiB request envelope.
	nearMax := ggufPaddedPNG(t, visionMaxBytes-1024)
	body = ggufImagesBody(ggufImageMessage(nearMax, nearMax))
	if len(body) <= 8<<20 || len(body) > 12<<20 {
		t.Fatal("wrong body fixture envelope", len(body))
	}
	if w := gatewayCall(g, key, "POST", "/v1/chat/completions", body); w.Code != 422 || calls != 2 {
		t.Fatal("new body envelope failed before child", w.Code, calls)
	}
	g.plan.GGUF.Vision.Images = nil
	if w := gatewayCall(g, key, "POST", "/v1/chat/completions", body); w.Code != 413 || calls != 2 {
		t.Fatal("legacy body limit changed", w.Code, calls)
	}
	g.plan.GGUF.Vision.Images = &policy
	if w := gatewayCall(g, key, "POST", "/v1/chat/completions", strings.Repeat("x", (12<<20)+1)); w.Code != 413 || calls != 2 {
		t.Fatal("new body hard limit missing", w.Code, calls)
	}
	args := strings.Join(llamaArgs(g.plan, 18001, "/private/key"), " ")
	for _, want := range []string{"--ctx-size " + strconv.Itoa(g.plan.Context), "--no-context-shift", "--parallel 1", "--fit off", "--image-max-tokens 1024", "--mtmd-batch-max-tokens 256"} {
		if !strings.Contains(args, want) {
			t.Fatal("multi-image changed fixed runtime policy", want, args)
		}
	}
}
