package modelship

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func ggufU32(b *[]byte, n uint32)    { *b = binary.LittleEndian.AppendUint32(*b, n) }
func ggufU64(b *[]byte, n uint64)    { *b = binary.LittleEndian.AppendUint64(*b, n) }
func ggufString(b *[]byte, s string) { ggufU64(b, uint64(len(s))); *b = append(*b, []byte(s)...) }

// Synthetic header only. There is deliberately no tensor payload in this fixture.
func ggufHeaderFixture(dim uint64, typ uint32) ([]byte, int) {
	b := []byte("GGUF")
	ggufU32(&b, 3)
	ggufU64(&b, 1)
	ggufU64(&b, 9)
	for _, kv := range [][2]string{{"general.architecture", "qwen35"}, {"general.type", "model"}} {
		ggufString(&b, kv[0])
		ggufU32(&b, 8)
		ggufString(&b, kv[1])
	}
	for _, kv := range []struct {
		key string
		n   uint32
	}{{"qwen35.context_length", 262144}, {"qwen35.block_count", 65}, {"qwen35.attention.head_count_kv", 4}, {"qwen35.attention.key_length", 256}, {"qwen35.attention.value_length", 256}, {"general.alignment", 32}} {
		ggufString(&b, kv.key)
		ggufU32(&b, 4)
		ggufU32(&b, kv.n)
	}
	ggufString(&b, "tokenizer.ggml.tokens")
	ggufU32(&b, 9)
	ggufU32(&b, 8)
	ggufU64(&b, 3)
	for _, v := range []string{"hello", "世界", "token"} {
		ggufString(&b, v)
	}
	ggufString(&b, "blk.0.weight")
	ggufU32(&b, 1)
	dimOffset := len(b)
	ggufU64(&b, dim)
	ggufU32(&b, typ)
	ggufU64(&b, 0)
	for len(b)%32 != 0 {
		b = append(b, 0)
	}
	return b, dimOffset
}
func parseHeaderFixture(b []byte, size int64) (GGUFMetadata, error) {
	return parseGGUF(size, func(offset, n int64) ([]byte, error) {
		if offset < 0 || offset+n > int64(len(b)) {
			return nil, fmt.Errorf("attempted tensor payload read")
		}
		return b[offset : offset+n], nil
	})
}
func TestGGUFHeaderNeverReadsTensorPayload(t *testing.T) {
	for _, typ := range []uint32{0, 1, 2, 8, 12, 14, 20, 21, 23, 30} {
		t.Run(fmt.Sprint(typ), func(t *testing.T) {
			b, _ := ggufHeaderFixture(256, typ)
			block := ggufBlocks[typ]
			bytes := 256 / block[0] * block[1]
			meta, e := parseHeaderFixture(b, int64(len(b))+bytes)
			if e != nil {
				t.Fatal(e)
			}
			if meta.TensorBytes != bytes || meta.HeaderBytes != int64(len(b)) || meta.Architecture != "qwen35" || meta.Context != 262144 || meta.Layers != 65 || meta.Types[typ] != 1 {
				t.Fatalf("wrong metadata: %+v", meta)
			}
		})
	}
}
func TestGGUFHeaderRejectsMalformedAndUnsupported(t *testing.T) {
	cases := map[string]func([]byte, int) []byte{
		"magic":                   func(b []byte, _ int) []byte { b[0] = '!'; return b },
		"version":                 func(b []byte, _ int) []byte { binary.LittleEndian.PutUint32(b[4:], 2); return b },
		"tensor-count":            func(b []byte, _ int) []byte { binary.LittleEndian.PutUint64(b[8:], 100001); return b },
		"metadata-count":          func(b []byte, _ int) []byte { binary.LittleEndian.PutUint64(b[16:], math.MaxUint64); return b },
		"string-overflow":         func(b []byte, _ int) []byte { binary.LittleEndian.PutUint64(b[24:], math.MaxUint64); return b },
		"unknown-architecture":    func(b []byte, _ int) []byte { return []byte(strings.Replace(string(b), "qwen35", "otherx", 1)) },
		"auxiliary-type":          func(b []byte, _ int) []byte { return []byte(strings.Replace(string(b), "model", "mmprj", 1)) },
		"shape-overflow":          func(b []byte, i int) []byte { binary.LittleEndian.PutUint64(b[i:], math.MaxUint64); return b },
		"zero-shape":              func(b []byte, i int) []byte { binary.LittleEndian.PutUint64(b[i:], 0); return b },
		"unknown-tensor-type":     func(b []byte, i int) []byte { binary.LittleEndian.PutUint32(b[i+8:], 999); return b },
		"misaligned-offset":       func(b []byte, i int) []byte { binary.LittleEndian.PutUint64(b[i+12:], 1); return b },
		"out-of-file-offset":      func(b []byte, i int) []byte { binary.LittleEndian.PutUint64(b[i+12:], 1<<40); return b },
		"invalid-dimension-count": func(b []byte, i int) []byte { binary.LittleEndian.PutUint32(b[i-4:], 5); return b },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			b, i := ggufHeaderFixture(32, 1)
			size := int64(len(b) + 64)
			b = change(b, i)
			if _, e := parseHeaderFixture(b, size); e == nil {
				t.Fatal("unsafe GGUF accepted")
			}
		})
	}
	for _, length := range []int{0, 4, 23, 40, 100, 400} {
		b, _ := ggufHeaderFixture(32, 1)
		if _, e := parseHeaderFixture(b[:length], int64(length)); e == nil {
			t.Fatal("truncated header accepted")
		}
	}
	b, _ := ggufHeaderFixture(32, 12)
	if _, e := parseHeaderFixture(b, int64(len(b)+64)); e == nil {
		t.Fatal("quant block row mismatch accepted")
	}
}
func TestGGUFRangeInspectorExactResponseAndNoTensorBytes(t *testing.T) {
	pin, _ := ggufTarget(ggufRepo, ggufRevision, ggufFilename)
	b, dimOffset := ggufHeaderFixture(32, 1)
	binary.LittleEndian.PutUint64(b[dimOffset:], uint64((pin.Size-int64(len(b)))/2))
	for _, mode := range []string{"ok", "ignored-range", "wrong-range", "encoded", "short"} {
		t.Run(mode, func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if r.URL.Path != "/"+ggufRepo+"/resolve/"+ggufRevision+"/"+ggufFilename {
					t.Error("unselected artifact requested")
				}
				var start, end int64
				if _, e := fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &start, &end); e != nil || start < 0 || end >= int64(len(b)) {
					t.Error("tensor payload requested")
					w.WriteHeader(400)
					return
				}
				if mode == "ignored-range" {
					w.WriteHeader(200)
					return
				}
				rangeSize := pin.Size
				if mode == "wrong-range" {
					rangeSize--
				}
				if mode == "encoded" {
					w.Header().Set("Content-Encoding", "gzip")
				}
				w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, rangeSize))
				w.WriteHeader(206)
				if mode == "short" {
					return
				}
				_, _ = w.Write(b[start : end+1])
			}))
			defer server.Close()
			hub := &Hub{BaseURL: server.URL, Client: server.Client()}
			m := Model{Hub: HubModel{ID: ggufRepo, SHA: ggufRevision, Siblings: []ModelFile{{Name: ggufFilename, Size: pin.Size}}}}
			e := hub.inspectGGUF(context.Background(), &m, ggufFilename)
			if mode == "ok" {
				if e != nil || m.GGUF == nil || calls > 20 {
					t.Fatal("bounded header inspection failed", e, calls)
				}
			} else if e == nil || m.GGUF != nil {
				t.Fatal("invalid server response accepted")
			}
		})
	}
}
func ggufPlanFixture(t *testing.T) (Model, Host, Options) {
	t.Helper()
	pin, _ := ggufTarget(ggufRepo, ggufRevision, ggufFilename)
	m := Model{Hub: HubModel{ID: ggufRepo, SHA: ggufRevision, Siblings: []ModelFile{{Name: ggufFilename, Size: pin.Size}, {Name: "mmproj.gguf", Size: 2 << 30}, {Name: "another-quant.gguf", Size: 30 << 30}}}, GGUF: &GGUFSpec{File: pin, Runtime: "llama-b12110-cuda12.8", Metadata: GGUFMetadata{Architecture: "qwen35", TensorBytes: pin.Size - 4<<20, TensorCount: 866, HeaderBytes: 4 << 20, Context: 262144, Layers: 65, KVHeads: 4, KeyLength: 256, ValueLength: 256}}}
	h := hostFixture()
	h.GPUs[0].TotalMiB = 81920
	h.GPUs[0].FreeMiB = 80000
	h.ModelCacheInspected = true
	h.ModelCacheDir = t.TempDir()
	h.ModelCacheFreeMiB = 100000
	o := optsFixture()
	o.Backend = "native"
	o.Experimental = true
	o.GGUFFile = ggufFilename
	return m, h, o
}
func TestGGUFPlanPinsOneArtifactAndKeepsHardBudgets(t *testing.T) {
	m, h, o := ggufPlanFixture(t)
	p, e := BuildPlan(m, h, o)
	if e != nil || len(p.Blockers) != 0 {
		t.Fatal(e, p.Blockers)
	}
	if p.Runtime != "llama.cpp" || p.GGUF == nil || len(p.Files) != 1 || p.Files[0] != ggufFilename || p.WeightBytes != m.GGUF.File.Size || p.WeightResidentBytes != m.GGUF.Metadata.TensorBytes || len(p.NativeRequirements) != 0 {
		t.Fatal("GGUF plan mixed formats/auxiliary files/Python requirements")
	}
	o.AllowRisk = true
	h.GPUs[0].FreeMiB = 8000
	p, e = BuildPlan(m, h, o)
	if e != nil || len(p.Blockers) == 0 || p.WeightsFitFree {
		t.Fatal("physical budget overridden", e)
	}
	h.GPUs[0].FreeMiB = 80000
	h.ModelCacheFreeMiB = 1
	p, e = BuildPlan(m, h, o)
	if e != nil || len(p.Blockers) == 0 {
		t.Fatal("disk budget overridden", e)
	}
}
func TestGGUFNoImplicitSelectionFormatFallbackOrCPUFallback(t *testing.T) {
	for _, name := range []string{"no-file", "other-revision", "other-file", "projector", "unknown-repository", "docker", "video", "no-header", "oversize-context"} {
		t.Run(name, func(t *testing.T) {
			m, h, o := ggufPlanFixture(t)
			switch name {
			case "no-file":
				o.GGUFFile = ""
			case "other-revision":
				m.Hub.SHA = strings.Repeat("a", 40)
			case "other-file":
				o.GGUFFile = "Qwen3.8-27B-Q8_0.gguf"
			case "projector":
				o.GGUFFile = "mmproj.gguf"
			case "unknown-repository":
				m.Hub.ID = "community/unknown"
			case "docker":
				o.Backend = "docker"
			case "video":
				o.Task = "text-to-video"
			case "no-header":
				m.GGUF = nil
			case "oversize-context":
				o.Context = 32769
			}
			if _, e := BuildPlan(m, h, o); e == nil {
				t.Fatal("unsupported selection accepted")
			}
		})
	}
	m, _, _ := ggufPlanFixture(t)
	m.GGUF = nil
	m.Hub.ID = "unknown/qwen-gguf"
	m.Config.Architectures = []string{"Qwen3_5ForConditionalGeneration"}
	if _, ok, _ := findRecipe(m); ok {
		t.Fatal("GGUF fell through to Transformers architecture matching")
	}
}
func TestGGUFCacheOnlyCountsSelectedFile(t *testing.T) {
	m, _, _ := ggufPlanFixture(t)
	root := t.TempDir()
	t.Setenv("HF_HUB_CACHE", root)
	h := Host{}
	InspectModelCache(&h, m)
	if !h.ModelCacheInspected || h.CachedWeightBytes != 0 {
		t.Fatal("empty cache miscounted")
	}
	path := ggufModelPath(Plan{Model: m.Hub.ID, Revision: m.Hub.SHA, ModelCacheDir: root, GGUF: m.GGUF})
	if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
		t.Fatal(e)
	}
	f, e := os.Create(path)
	if e != nil {
		t.Fatal(e)
	}
	if e = f.Truncate(m.GGUF.File.Size); e != nil {
		t.Fatal(e)
	}
	f.Close()
	InspectModelCache(&h, m)
	if h.CachedWeightBytes != m.GGUF.File.Size {
		t.Fatal("selected exact-size file not counted")
	}
}
func FuzzGGUFHeader(f *testing.F) {
	seed, _ := ggufHeaderFixture(32, 1)
	f.Add(seed)
	f.Add([]byte("GGUF"))
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 2<<20 {
			t.Skip()
		}
		_, _ = parseHeaderFixture(b, int64(len(b)+64))
	})
}
