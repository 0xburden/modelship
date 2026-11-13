package modelship

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

const ggufMaxHeader int64 = 64 << 20

type GGUFMetadata struct {
	Attributes map[string]string `json:"attributes,omitempty"`

	Architecture string         `json:"architecture"`
	TensorCount  int            `json:"tensor_count"`
	TensorBytes  int64          `json:"tensor_bytes"`
	HeaderBytes  int64          `json:"header_bytes"`
	Context      int64          `json:"context_length"`
	Layers       int64          `json:"layers"`
	KVHeads      int64          `json:"kv_heads"`
	KeyLength    int64          `json:"key_length"`
	ValueLength  int64          `json:"value_length"`
	Types        map[uint32]int `json:"tensor_types"`
}
type ggufRead struct {
	size, pos, fetched, start int64
	buf                       []byte
	read                      func(int64, int64) ([]byte, error)
}

func (g *ggufRead) take(n, tail int64) ([]byte, error) {
	if n < 0 || tail < 0 || n > ggufMaxHeader || g.pos > ggufMaxHeader-n || g.pos > g.size-n {
		return nil, fmt.Errorf("invalid/oversized GGUF header")
	}
	if g.pos < g.start || g.pos+n > g.start+int64(len(g.buf)) {
		length := n
		// Read-ahead is limited to a known lower bound of remaining metadata/table
		// bytes, never a blind window into tensor payloads. Large tokenizer arrays
		// can thus be skipped with bounded Range requests rather than per-token IO.
		if extra := min(int64(256<<10), n+tail); extra > length {
			length = extra
		}
		length = min(length, ggufMaxHeader-g.pos, g.size-g.pos)
		if g.fetched > 96<<20-length {
			return nil, fmt.Errorf("GGUF aggregate header budget exceeded")
		}
		data, err := g.read(g.pos, length)
		if err != nil {
			return nil, err
		}
		if int64(len(data)) != length {
			return nil, fmt.Errorf("short GGUF header read")
		}
		g.start = g.pos
		g.buf = data
		g.fetched += length
	}
	data := g.buf[g.pos-g.start : g.pos-g.start+n]
	g.pos += n
	return data, nil
}
func (g *ggufRead) number(n, tail int64) (uint64, error) {
	b, e := g.take(n, tail)
	if e != nil {
		return 0, e
	}
	switch n {
	case 1:
		return uint64(b[0]), nil
	case 2:
		return uint64(binary.LittleEndian.Uint16(b)), nil
	case 4:
		return uint64(binary.LittleEndian.Uint32(b)), nil
	case 8:
		return binary.LittleEndian.Uint64(b), nil
	}
	return 0, fmt.Errorf("invalid numeric width")
}
func (g *ggufRead) str(limit, tail int64, keep bool) (string, error) {
	n, e := g.number(8, tail)
	if e != nil {
		return "", e
	}
	if n > uint64(limit) {
		return "", fmt.Errorf("GGUF string exceeds bounds")
	}
	data, e := g.take(int64(n), tail)
	if e != nil {
		return "", e
	}
	if !utf8.Valid(data) {
		return "", fmt.Errorf("invalid GGUF UTF-8")
	}
	if keep {
		return string(data), nil
	}
	return "", nil
}

var ggufWidths = map[uint32]int64{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

func ggufMinValue(t uint32) int64 {
	if t == 8 {
		return 8
	}
	if t == 9 {
		return 12
	}
	return ggufWidths[t]
}
func (g *ggufRead) value(t uint32, tail int64, keep bool) (any, error) {
	if width := ggufWidths[t]; width > 0 {
		v, e := g.number(width, tail)
		if e != nil {
			return nil, e
		}
		if t == 7 && v > 1 {
			return nil, fmt.Errorf("invalid GGUF boolean")
		}
		if t == 6 && (!isFinite(float64(math.Float32frombits(uint32(v))))) {
			return nil, fmt.Errorf("invalid GGUF float")
		}
		if t == 12 && !isFinite(math.Float64frombits(v)) {
			return nil, fmt.Errorf("invalid GGUF float")
		}
		return v, nil
	}
	if t == 8 {
		return g.str(8<<20, tail, keep)
	}
	if t != 9 {
		return nil, fmt.Errorf("unsupported GGUF metadata type")
	}
	typ, e := g.number(4, 8+tail)
	if e != nil {
		return nil, e
	}
	count, e := g.number(8, tail)
	if e != nil {
		return nil, e
	}
	if count > 1_000_000 || typ == 9 || ggufMinValue(uint32(typ)) == 0 {
		return nil, fmt.Errorf("invalid/oversized GGUF array")
	}
	for i := uint64(0); i < count; i++ {
		if _, e = g.value(uint32(typ), int64(count-i-1)*ggufMinValue(uint32(typ))+tail, false); e != nil {
			return nil, e
		}
	}
	return nil, nil
}
func isFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// GGML block sizes are pinned to llama.cpp b12110. Unknown types fail closed.
var ggufBlocks = map[uint32][2]int64{0: {1, 4}, 1: {1, 2}, 2: {32, 18}, 3: {32, 20}, 6: {32, 22}, 7: {32, 24}, 8: {32, 34}, 9: {32, 36}, 10: {256, 84}, 11: {256, 110}, 12: {256, 144}, 13: {256, 176}, 14: {256, 210}, 15: {256, 292}, 16: {256, 66}, 17: {256, 74}, 18: {256, 98}, 19: {256, 50}, 22: {256, 82}, 29: {256, 56}, 20: {32, 18}, 21: {256, 110}, 23: {256, 136}, 30: {1, 2}}

func parseGGUF(size int64, read func(int64, int64) ([]byte, error)) (GGUFMetadata, error) {
	return parseGGUFFor(size, read, "qwen35")
}
func parseGGUFFor(size int64, read func(int64, int64) ([]byte, error), wanted string) (GGUFMetadata, error) {
	var out GGUFMetadata
	g := ggufRead{size: size, read: read}
	head, e := g.take(24, 0)
	if e != nil {
		return out, e
	}
	if string(head[:4]) != "GGUF" || binary.LittleEndian.Uint32(head[4:8]) != 3 {
		return out, fmt.Errorf("only GGUF version 3 is supported")
	}
	tensors, entries := binary.LittleEndian.Uint64(head[8:16]), binary.LittleEndian.Uint64(head[16:24])
	if tensors == 0 || tensors > 100000 || entries == 0 || entries > 4096 {
		return out, fmt.Errorf("invalid GGUF table counts")
	}
	fields := map[string]any{}
	keys := map[string]bool{}
	for i := uint64(0); i < entries; i++ {
		tail := int64(entries-i-1)*13 + int64(tensors)*32
		key, e := g.str(256, 5+tail, true)
		if e != nil {
			return out, e
		}
		if key == "" || keys[key] || strings.IndexFunc(key, unicode.IsControl) >= 0 {
			return out, fmt.Errorf("duplicate/invalid GGUF metadata key")
		}
		keys[key] = true
		typ, e := g.number(4, 1+tail)
		if e != nil {
			return out, e
		}
		keep := key == "general.architecture" || key == "general.type" || key == "general.alignment" || strings.HasPrefix(key, "qwen35.") || strings.HasPrefix(key, "clip.") || strings.HasPrefix(key, "split.")
		value, e := g.value(uint32(typ), tail, keep)
		if e != nil {
			return out, e
		}
		if keep {
			fields[key] = value
		}
	}
	arch, ok := fields["general.architecture"].(string)
	if !ok || arch != wanted || (wanted == "qwen35" && fields["general.type"] != "model") {
		return out, fmt.Errorf("GGUF is not a qwen35 standalone model")
	}
	if _, ok = fields["split.count"]; ok {
		return out, fmt.Errorf("split GGUF is not enabled for this first adapter")
	}
	align := int64(32)
	if v, exists := fields["general.alignment"]; exists {
		n, ok := v.(uint64)
		if !ok || n < 16 || n > 4096 || n&(n-1) != 0 {
			return out, fmt.Errorf("invalid GGUF alignment")
		}
		align = int64(n)
	}
	type span struct{ offset, bytes int64 }
	spans := make([]span, 0, tensors)
	names := map[string]bool{}
	out.Types = map[uint32]int{}
	for i := uint64(0); i < tensors; i++ {
		tail := int64(tensors-i-1) * 32
		name, e := g.str(256, 24+tail, true)
		if e != nil {
			return out, e
		}
		if name == "" || names[name] || strings.IndexFunc(name, unicode.IsControl) >= 0 {
			return out, fmt.Errorf("duplicate/invalid GGUF tensor")
		}
		names[name] = true
		dims, e := g.number(4, 20+tail)
		if e != nil {
			return out, e
		}
		if dims < 1 || dims > 4 {
			return out, fmt.Errorf("invalid GGUF tensor dimensions")
		}
		count, first := int64(1), int64(0)
		for j := uint64(0); j < dims; j++ {
			v, e := g.number(8, int64(dims-j-1)*8+12+tail)
			if e != nil {
				return out, e
			}
			if v == 0 || v > 1<<40 || count > math.MaxInt64/int64(v) {
				return out, fmt.Errorf("GGUF shape overflow")
			}
			count *= int64(v)
			if j == 0 {
				first = int64(v)
			}
		}
		typ, e := g.number(4, 8+tail)
		if e != nil {
			return out, e
		}
		block, known := ggufBlocks[uint32(typ)]
		if !known || first%block[0] != 0 || count/block[0] > math.MaxInt64/block[1] {
			return out, fmt.Errorf("unsupported GGUF tensor type %d or row shape %d", typ, first)
		}
		bytes := count / block[0] * block[1]
		offset, e := g.number(8, tail)
		if e != nil {
			return out, e
		}
		if offset > uint64(size) || offset%uint64(align) != 0 || bytes > size-int64(offset) {
			return out, fmt.Errorf("invalid GGUF tensor offset")
		}
		if out.TensorBytes > math.MaxInt64-bytes {
			return out, fmt.Errorf("GGUF tensor-byte overflow")
		}
		out.TensorBytes += bytes
		out.Types[uint32(typ)]++
		spans = append(spans, span{int64(offset), bytes})
	}
	start := (g.pos + align - 1) / align * align
	sort.Slice(spans, func(i, j int) bool { return spans[i].offset < spans[j].offset })
	end := int64(0)
	for _, s := range spans {
		if s.offset < end || s.offset-end >= align || s.offset > size-start || s.bytes > size-start-s.offset {
			return out, fmt.Errorf("overlapping/out-of-file GGUF tensor ranges")
		}
		end = s.offset + s.bytes
	}
	if size-start-end < 0 || size-start-end >= align {
		return out, fmt.Errorf("unexpected GGUF trailing payload")
	}
	number := func(key string) int64 {
		v, ok := fields[key].(uint64)
		if !ok || v > math.MaxInt64 {
			return 0
		}
		return int64(v)
	}
	out.Architecture = arch
	out.TensorCount = int(tensors)
	out.HeaderBytes = start
	out.Context = number("qwen35.context_length")
	out.Layers = number("qwen35.block_count")
	out.KVHeads = number("qwen35.attention.head_count_kv")
	out.KeyLength = number("qwen35.attention.key_length")
	out.ValueLength = number("qwen35.attention.value_length")
	if wanted == "clip" {
		out.Attributes = map[string]string{}
		for _, key := range []string{"general.architecture", "general.type", "clip.projector_type", "clip.has_vision_encoder", "clip.use_gelu", "clip.vision.attention.head_count", "clip.vision.block_count", "clip.vision.embedding_length", "clip.vision.feed_forward_length", "clip.vision.image_size", "clip.vision.patch_size", "clip.vision.projection_dim", "clip.vision.spatial_merge_size"} {
			value := fields[key]
			switch v := value.(type) {
			case string:
				out.Attributes[key] = v
			case uint64:
				out.Attributes[key] = strconv.FormatUint(v, 10)
			}
		}
		return out, nil
	}
	if out.Context < 128 || out.Context > 10_000_000 || out.Layers < 1 || out.Layers > 512 || out.KVHeads < 1 || out.KVHeads > 256 || out.KeyLength < 1 || out.KeyLength > 4096 || out.ValueLength < 1 || out.ValueLength > 4096 {
		return out, fmt.Errorf("GGUF context/cache metadata is missing or unsupported")
	}
	return out, nil
}
func (h *Hub) inspectGGUF(ctx context.Context, m *Model, file string) error {
	pin, e := ggufTarget(m.Hub.ID, m.Hub.SHA, file)
	if e != nil {
		return e
	}
	meta, e := h.inspectGGUFHeader(ctx, m, pin, "qwen35")
	if e != nil {
		return e
	}
	runtimeSpec, e := loadLlamaRuntime("")
	if e != nil {
		return e
	}
	m.GGUF = &GGUFSpec{File: pin, Metadata: meta, Runtime: runtimeSpec.ID}
	return nil
}
func (h *Hub) inspectGGUFProjector(ctx context.Context, m *Model, file string) error {
	if m.GGUF == nil {
		return fmt.Errorf("select and inspect the language GGUF first")
	}
	pin, e := ggufArtifact(m.Hub.ID, m.Hub.SHA, file, true)
	if e != nil {
		return e
	}
	meta, e := h.inspectGGUFHeader(ctx, m, pin, "clip")
	if e != nil {
		return e
	}
	v := &GGUFVisionSpec{File: pin, Metadata: meta}
	if e := validateGGUFVision(v); e != nil {
		return e
	}
	m.GGUF.Vision = v
	return nil
}
func (h *Hub) inspectGGUFHeader(ctx context.Context, m *Model, pin ArtifactPin, architecture string) (GGUFMetadata, error) {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	file := pin.Name
	found := false
	for _, f := range m.Hub.Siblings {
		if f.Name == file {
			if f.Bytes() != pin.Size {
				return GGUFMetadata{}, fmt.Errorf("selected GGUF size differs from approved artifact")
			}
			found = true
		}
	}
	if !found {
		return GGUFMetadata{}, fmt.Errorf("selected GGUF file is absent")
	}
	read := func(offset, n int64) ([]byte, error) {
		req, e := http.NewRequestWithContext(ctx, "GET", h.BaseURL+"/"+m.Hub.ID+"/resolve/"+m.Hub.SHA+"/"+escapeWeightPath(file), nil)
		if e != nil {
			return nil, e
		}
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", offset, offset+n-1))
		req.Header.Set("Accept-Encoding", "identity")
		if h.Token != "" {
			req.Header.Set("Authorization", "Bearer "+h.Token)
		}
		client := h.Client
		if client == nil {
			client = NewHub().Client
		}
		resp, e := client.Do(req)
		if e != nil {
			return nil, fmt.Errorf("GGUF header transport failed")
		}
		defer resp.Body.Close()
		want := fmt.Sprintf("bytes %d-%d/%d", offset, offset+n-1, pin.Size)
		if resp.StatusCode != 206 || resp.Header.Get("Content-Range") != want || resp.Header.Get("Content-Encoding") != "" {
			return nil, fmt.Errorf("GGUF server did not honor exact bounded Range")
		}
		if value := resp.Header.Get("Content-Length"); value != "" {
			length, e := strconv.ParseInt(value, 10, 64)
			if e != nil || length != n {
				return nil, fmt.Errorf("invalid GGUF range size")
			}
		}
		b, e := io.ReadAll(io.LimitReader(resp.Body, n+1))
		if e != nil || int64(len(b)) != n {
			return nil, fmt.Errorf("invalid GGUF header body")
		}
		return b, nil
	}
	return parseGGUFFor(pin.Size, read, architecture)
}
