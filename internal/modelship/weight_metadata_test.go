package modelship

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

type virtualWeight struct {
	header []byte
	size   int64
}

func virtualTensor(dtype string, elements, width int64) virtualWeight {
	header, _ := json.Marshal(map[string]any{"weight": map[string]any{"dtype": dtype, "shape": []int64{elements}, "data_offsets": []int64{0, elements * width}}})
	return virtualWeight{header: header, size: int64(len(header)) + 8 + elements*width}
}
func weightTestHub(t *testing.T, m *Model, files map[string]virtualWeight) *Hub {
	t.Helper()
	m.Hub.Siblings = nil
	for name, file := range files {
		m.Hub.Siblings = append(m.Hub.Siblings, ModelFile{Name: name, Size: file.size})
	}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		prefix := "/" + m.Hub.ID + "/resolve/" + m.Hub.SHA + "/"
		if !strings.HasPrefix(r.URL.Path, prefix) {
			t.Errorf("request did not use immutable SHA: %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		file, ok := files[strings.TrimPrefix(r.URL.Path, prefix)]
		if !ok {
			http.NotFound(w, r)
			return
		}
		start, end := int64(-1), int64(-1)
		if _, err := fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &start, &end); err != nil {
			t.Errorf("missing Range")
			http.Error(w, "range required", 400)
			return
		}
		if start == 0 && end == 7 {
			data := make([]byte, 8)
			binary.LittleEndian.PutUint64(data, uint64(len(file.header)))
			w.Header().Set("Content-Range", fmt.Sprintf("bytes 0-7/%d", file.size))
			w.Header().Set("Content-Length", "8")
			w.WriteHeader(206)
			_, _ = w.Write(data)
			return
		}
		if start != 8 || end != int64(len(file.header))+7 {
			t.Errorf("attempt to download tensor payload: %d-%d", start, end)
			http.Error(w, "payload forbidden", 400)
			return
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, file.size))
		w.Header().Set("Content-Length", strconv.Itoa(len(file.header)))
		w.WriteHeader(206)
		_, _ = w.Write(file.header)
	}))
	t.Cleanup(s.Close)
	return &Hub{BaseURL: s.URL, Client: s.Client()}
}
func TestWeightHeadersGiantBF16TextHardBlock(t *testing.T) {
	m := textFixture()
	m.Hub.Safetensors.Total = 1 // Do not trust a misleading Hub parameter projection.
	h := weightTestHub(t, &m, map[string]virtualWeight{"model.safetensors": virtualTensor("BF16", 27_000_000_000, 2)})
	if err := h.InspectWeightMetadata(context.Background(), &m); err != nil {
		t.Fatal(err)
	}
	o := optsFixture()
	o.AllowRisk = true
	p, err := BuildPlan(m, hostFixture(), o)
	if err != nil {
		t.Fatal(err)
	}
	if p.WeightResidentBytes != 54_000_000_000 || p.WeightsFitTotal || p.WeightsFitFree || len(p.Blockers) != 2 {
		t.Fatalf("%+v", p)
	}
	if !strings.Contains(p.WeightMemoryBasis, "validated safetensors") {
		t.Fatal(p.WeightMemoryBasis)
	}
}
func TestWeightHeadersFP32ImageLoadsHalfPrecision(t *testing.T) {
	m := imageFixture()
	h := weightTestHub(t, &m, map[string]virtualWeight{"unet/diffusion_pytorch_model.safetensors": virtualTensor("F32", 4_000_000_000, 4)})
	if err := h.InspectWeightMetadata(context.Background(), &m); err != nil {
		t.Fatal(err)
	}
	p, err := BuildPlan(m, hostFixture(), optsFixture())
	if err != nil {
		t.Fatal(err)
	}
	if p.WeightResidentBytes != 8_000_000_000 || p.WeightBytes <= 16_000_000_000 || !p.WeightsFitTotal || !p.WeightsFitFree || len(p.Blockers) != 0 {
		t.Fatalf("%+v", p)
	}
}
func TestMixedFloatingAndIntegerResidentBytes(t *testing.T) {
	header := []byte(`{"bf":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"fp":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},"index":{"dtype":"I64","shape":[3],"data_offsets":[16,40]}}`)
	meta, err := parseWeightHeader(header, int64(len(header))+48)
	if err != nil {
		t.Fatal(err)
	}
	meta.Name = "vae/diffusion_pytorch_model.safetensors"
	m := Model{WeightMetadataChecked: true, WeightMetadata: []WeightFileMetadata{meta}}
	count, basis, err := residentWeightBytes(m, []string{meta.Name}, meta.Size, "bfloat16", map[string]string{"vae": "float32"})
	if err != nil || count != 48 || !strings.Contains(basis, "integer buffers") {
		t.Fatalf("%d %s %v", count, basis, err)
	}
	count, _, err = residentWeightBytes(m, []string{meta.Name}, meta.Size, "bfloat16", nil)
	if err != nil || count != 36 {
		t.Fatalf("%d %v", count, err)
	}
}
func TestFullWanDualExpertWeightSelection(t *testing.T) {
	m := Model{Hub: HubModel{ID: "Wan-AI/Wan2.2-T2V-A14B-Diffusers", SHA: strings.Repeat("a", 40)}, Index: ModelIndex{Class: "WanPipeline"}}
	files := map[string]virtualWeight{
		"transformer/diffusion_pytorch_model-00001-of-00002.safetensors":   virtualTensor("BF16", 7_000_000_000, 2),
		"transformer/diffusion_pytorch_model-00002-of-00002.safetensors":   virtualTensor("BF16", 7_000_000_000, 2),
		"transformer_2/diffusion_pytorch_model-00001-of-00002.safetensors": virtualTensor("BF16", 7_000_000_000, 2),
		"transformer_2/diffusion_pytorch_model-00002-of-00002.safetensors": virtualTensor("BF16", 7_000_000_000, 2),
		"text_encoder/model.safetensors":                                   virtualTensor("BF16", 4_000_000_000, 2),
		"vae/diffusion_pytorch_model.safetensors":                          virtualTensor("F32", 100_000_000, 4),
	}
	h := weightTestHub(t, &m, files)
	if err := h.InspectWeightMetadata(context.Background(), &m); err != nil {
		t.Fatal(err)
	}
	names, download := weightFiles(m.Hub.Siblings, true)
	resident, _, err := residentWeightBytes(m, names, download, "bfloat16", map[string]string{"vae": "float32"})
	if err != nil || len(names) != 6 || resident != 64_400_000_000 {
		t.Fatalf("files=%v resident=%d err=%v", names, resident, err)
	}
	o := optsFixture()
	o.Backend = "native"
	o.Experimental = true
	o.Research = true
	o.AllowRisk = true
	host := hostFixture()
	host.GPUs[0].TotalMiB = 49152
	host.GPUs[0].FreeMiB = 49152
	p, err := BuildPlan(m, host, o)
	if err != nil || p.Task != "text-to-video" || p.Runtime != "diffusers" || p.NativeLoader != "wan-pipeline" || p.ComponentDTypes["vae"] != "float32" || p.WeightResidentBytes != resident || p.WeightsFitTotal || p.WeightsFitFree || len(p.Blockers) != 2 {
		t.Fatalf("%+v %v", p, err)
	}
	host.GPUs[0].TotalMiB = 98304
	host.GPUs[0].FreeMiB = 98304
	p, err = BuildPlan(m, host, o)
	if err != nil || !p.WeightsFitTotal || !p.WeightsFitFree || len(p.Blockers) != 0 {
		t.Fatalf("%+v %v", p, err)
	}
	host.GPUs[0].Compute = "7.5"
	p, err = BuildPlan(m, host, o)
	if err != nil || !strings.Contains(strings.Join(p.Blockers, ";"), "recipe requires compute capability >= 8.0") {
		t.Fatalf("%+v %v", p, err)
	}
}
func TestWanMissingEntireExpertFailsClosed(t *testing.T) {
	m := Model{Hub: HubModel{ID: "Wan-AI/Wan2.2-T2V-A14B-Diffusers", SHA: strings.Repeat("a", 40), Siblings: []ModelFile{
		{Name: "transformer/diffusion_pytorch_model.safetensors", Size: 100},
		{Name: "text_encoder/model.safetensors", Size: 100},
		{Name: "vae/diffusion_pytorch_model.safetensors", Size: 100},
	}}, Index: ModelIndex{Class: "WanPipeline"}}
	if err := NewHub().InspectWeightMetadata(context.Background(), &m); err == nil || !strings.Contains(m.WeightMetadataError, "both denoising experts") {
		t.Fatalf("%+v %v", m, err)
	}
}
func TestMalformedWeightHeaders(t *testing.T) {
	tests := map[string]string{
		"duplicate-tensor":     `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`,
		"duplicate-descriptor": `{"x":{"dtype":"F32","dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`,
		"unknown-dtype":        `{"x":{"dtype":"Q4","shape":[1],"data_offsets":[0,2]}}`,
		"negative-shape":       `{"x":{"dtype":"BF16","shape":[-1],"data_offsets":[0,2]}}`,
		"fractional-shape":     `{"x":{"dtype":"BF16","shape":[1.0],"data_offsets":[0,2]}}`,
		"overflow-shape":       `{"x":{"dtype":"BF16","shape":[9223372036854775807,2],"data_offsets":[0,2]}}`,
		"overflow-width":       `{"x":{"dtype":"BF16","shape":[9223372036854775807],"data_offsets":[0,2]}}`,
		"missing-shape":        `{"x":{"dtype":"BF16","data_offsets":[0,2]}}`,
		"null-shape":           `{"x":{"dtype":"BF16","shape":null,"data_offsets":[0,2]}}`,
		"missing-offset":       `{"x":{"dtype":"BF16","shape":[1]}}`,
		"negative-offset":      `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[-1,1]}}`,
		"reversed-offset":      `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[2,0]}}`,
		"wrong-width":          `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,2]}}`,
		"gap":                  `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[1,3]}}`,
		"overlap":              `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"y":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`,
		"trailing":             `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}} {}`,
		"unknown-field":        `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2],"extra":true}}`,
		"bad-metadata":         `{"__metadata__":{"value":null},"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`,
		"empty":                `{}`,
	}
	for name, header := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := parseWeightHeader([]byte(header), int64(len(header))+10); err == nil {
				t.Fatal("accepted malformed header")
			}
		})
	}
}
func TestIntegerAndBooleanBuffersKeepStorageWidth(t *testing.T) {
	for dtype, width := range map[string]int64{"BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8} {
		file := virtualTensor(dtype, 10, width)
		meta, err := parseWeightHeader(file.header, file.size)
		if err != nil || meta.IntegerBytes != 10*width || meta.FloatingElements != 0 {
			t.Fatalf("%s %+v %v", dtype, meta, err)
		}
	}
}
func TestWeightHeaderScalarAndZeroTensor(t *testing.T) {
	header := []byte(`{"empty":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"scalar":{"dtype":"I16","shape":[],"data_offsets":[0,2]}}`)
	m, err := parseWeightHeader(header, int64(len(header))+10)
	if err != nil || m.IntegerBytes != 2 {
		t.Fatalf("%+v %v", m, err)
	}
}
func TestWeightProbeMissingShardAndSizeFailClosed(t *testing.T) {
	for _, test := range []struct {
		name  string
		files []ModelFile
	}{
		{"missing-shard", []ModelFile{{Name: "model-00001-of-00002.safetensors", Size: 100}}},
		{"unknown-size", []ModelFile{{Name: "model.safetensors"}}},
		{"overflow-size", []ModelFile{{Name: "model-00001-of-00002.safetensors", Size: math.MaxInt64}, {Name: "model-00002-of-00002.safetensors", Size: 100}}},
		{"duplicate-file", []ModelFile{{Name: "model.safetensors", Size: 100}, {Name: "model.safetensors", Size: 100}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			m := textFixture()
			m.Hub.Siblings = test.files
			if err := NewHub().InspectWeightMetadata(context.Background(), &m); err == nil || m.WeightMetadataError == "" || len(m.WeightMetadata) != 0 {
				t.Fatalf("%+v %v", m, err)
			}
			o := optsFixture()
			o.AllowRisk = true
			p, err := BuildPlan(m, hostFixture(), o)
			if err != nil || !strings.Contains(strings.Join(p.Blockers, ";"), "weights-alone VRAM preflight failed") || p.WeightsFitTotal || p.WeightsFitFree {
				t.Fatalf("%+v %v", p, err)
			}
		})
	}
}

type headerTransport func(*http.Request) (*http.Response, error)

func (f headerTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type forbiddenBody struct{ read bool }

func (b *forbiddenBody) Read([]byte) (int, error) { b.read = true; return 0, io.EOF }
func (*forbiddenBody) Close() error               { return nil }
func TestWeightProbeManifestMissingShard(t *testing.T) {
	for _, manifest := range []string{
		`{"weight_map":{"x":"model-00002-of-00002.safetensors"}}`,
		`{"weight_map":{}}`,
		`{"weight_map":{"x":"model-00001-of-00001.safetensors","x":"missing.safetensors"}}`,
		`{"weight_map":{"x":"../model.safetensors"}}`,
	} {
		m := textFixture()
		m.Hub.Siblings = []ModelFile{{Name: "model-00001-of-00001.safetensors", Size: 100}, {Name: "model.safetensors.index.json", Size: int64(len(manifest))}}
		h := &Hub{BaseURL: "https://example.invalid", Client: &http.Client{Transport: headerTransport(func(r *http.Request) (*http.Response, error) {
			if !strings.HasSuffix(r.URL.Path, ".index.json") {
				t.Errorf("weight request before validating manifest: %s", r.URL.Path)
			}
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(manifest)), Header: make(http.Header)}, nil
		})}}
		if err := h.InspectWeightMetadata(context.Background(), &m); err == nil || m.WeightMetadataError == "" {
			t.Fatalf("accepted manifest %s", manifest)
		}
	}
}
func TestWeightProbeNoPayloadAndNoSecretErrors(t *testing.T) {
	body := &forbiddenBody{}
	m := textFixture()
	h := &Hub{BaseURL: "https://example.invalid", Token: "SECRET", Client: &http.Client{Transport: headerTransport(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: body, Header: make(http.Header)}, nil
	})}}
	if err := h.InspectWeightMetadata(context.Background(), &m); err == nil || body.read {
		t.Fatalf("read payload=%v err=%v", body.read, err)
	}
	h.Client.Transport = headerTransport(func(r *http.Request) (*http.Response, error) {
		return nil, fmt.Errorf("https://signed.invalid/weights?token=SECRET")
	})
	err := h.InspectWeightMetadata(context.Background(), &m)
	if err == nil || strings.Contains(err.Error(), "SECRET") || strings.Contains(err.Error(), "https://") || strings.Contains(m.WeightMetadataError, "SECRET") {
		t.Fatalf("unsafe error: %v", err)
	}
}
func TestWeightProbeRejectsPartialAndOversizedRanges(t *testing.T) {
	for _, mode := range []string{"truncated", "wrong-size", "oversized", "bad-length", "encoded"} {
		t.Run(mode, func(t *testing.T) {
			m := textFixture()
			h := &Hub{BaseURL: "https://example.invalid", Client: &http.Client{Transport: headerTransport(func(r *http.Request) (*http.Response, error) {
				data := make([]byte, 8)
				binary.LittleEndian.PutUint64(data, 100)
				headers := http.Header{"Content-Range": []string{fmt.Sprintf("bytes 0-7/%d", m.Hub.Siblings[0].Size)}}
				length := int64(8)
				switch mode {
				case "truncated":
					data = data[:4]
				case "wrong-size":
					headers.Set("Content-Range", "bytes 0-7/42")
				case "oversized":
					binary.LittleEndian.PutUint64(data, maxWeightHeader+1)
				case "bad-length":
					length = 9
				case "encoded":
					headers.Set("Content-Encoding", "gzip")
				}
				return &http.Response{StatusCode: 206, Body: io.NopCloser(strings.NewReader(string(data))), Header: headers, ContentLength: length}, nil
			})}}
			if err := h.InspectWeightMetadata(context.Background(), &m); err == nil || !m.WeightMetadataChecked || m.WeightMetadataError == "" {
				t.Fatalf("%+v %v", m, err)
			}
		})
	}
}
func TestProbedWeightsUseSelectedGPUFreeMemory(t *testing.T) {
	m := textFixture()
	h := weightTestHub(t, &m, map[string]virtualWeight{"model.safetensors": virtualTensor("BF16", 1_000_000_000, 2)})
	if err := h.InspectWeightMetadata(context.Background(), &m); err != nil {
		t.Fatal(err)
	}
	host := hostFixture()
	host.GPUs[0].FreeMiB = 1000
	other := host.GPUs[0]
	other.Index = 1
	other.TotalMiB = 98304
	other.FreeMiB = 98304
	host.GPUs = append(host.GPUs, other)
	o := optsFixture()
	o.AllowRisk = true
	p, err := BuildPlan(m, host, o)
	if err != nil || !p.WeightsFitTotal || p.WeightsFitFree || len(p.Blockers) != 1 || !strings.Contains(p.Blockers[0], "currently free") {
		t.Fatalf("%+v %v", p, err)
	}
	o.GPU = 1
	p, err = BuildPlan(m, host, o)
	if err != nil || !p.WeightsFitTotal || !p.WeightsFitFree || len(p.Blockers) != 0 {
		t.Fatalf("%+v %v", p, err)
	}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"weight_download_bytes":`, `"weight_resident_bytes":`, `"weight_memory_basis":`, `"weights_fit_total":true`, `"weights_fit_free":true`} {
		if !strings.Contains(string(data), field) {
			t.Fatalf("missing %s: %s", field, data)
		}
	}
}
func TestImageRecipeMinimumComputeIsHardBlocker(t *testing.T) {
	m := imageFixture()
	m.Index.Class = "QwenImage21Pipeline"
	host := hostFixture()
	host.GPUs[0].Compute = "7.5"
	o := optsFixture()
	o.Backend = "native"
	o.Experimental = true
	o.Research = true
	o.AllowRisk = true
	p, err := BuildPlan(m, host, o)
	if err != nil || !strings.Contains(strings.Join(p.Blockers, ";"), "recipe requires compute capability >= 8.0") {
		t.Fatalf("%+v %v", p, err)
	}
}
func TestResidentCoverageAndOverflowFailClosed(t *testing.T) {
	for _, metas := range [][]WeightFileMetadata{
		nil,
		{{Name: "other.safetensors", FloatingElements: 1}},
		{{Name: "model.safetensors", FloatingElements: math.MaxInt64}},
		{{Name: "model.safetensors", FloatingElements: 1, IntegerBytes: -1}},
	} {
		m := Model{WeightMetadataChecked: true, WeightMetadata: metas}
		if _, _, err := residentWeightBytes(m, []string{"model.safetensors"}, 100, "float16", nil); err == nil {
			t.Fatalf("accepted %+v", metas)
		}
	}
}
