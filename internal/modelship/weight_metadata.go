package modelship

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// WeightFileMetadata describes validated tensor contents, not download size.
// Floating tensors are cast to the recipe's requested component dtype; integer
// and boolean buffers retain their original storage width.
type WeightFileMetadata struct {
	Name             string
	Size             int64
	FloatingElements int64
	IntegerBytes     int64
}

const maxWeightHeader = 16 * 1024 * 1024
const maxWeightHeadersTotal = 64 * 1024 * 1024

var shardName = regexp.MustCompile(`^(.*)-([0-9]+)-of-([0-9]+)\.safetensors$`)

// InspectWeightMetadata reads only bounded safetensors headers at the immutable
// revision. Errors are also recorded on m so BuildPlan can present a fail-closed
// blocker instead of silently falling back to a heuristic.
func (h *Hub) InspectWeightMetadata(ctx context.Context, m *Model) (err error) {
	if m == nil {
		return fmt.Errorf("weight metadata requires a model")
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	m.WeightMetadataChecked = true
	m.WeightMetadataError = ""
	m.WeightMetadata = nil
	defer func() {
		if err != nil {
			m.WeightMetadata = nil
			m.WeightMetadataError = err.Error()
		}
	}()
	if !commitPattern.MatchString(m.Hub.SHA) {
		return fmt.Errorf("weight metadata requires an immutable commit SHA")
	}
	if _, e := ParseModel(m.Hub.ID); e != nil || strings.Contains(m.Hub.ID, "://") {
		return fmt.Errorf("invalid weight repository ID")
	}
	image := m.Index.Class != ""
	names, total := weightFiles(m.Hub.Siblings, image)
	if len(names) == 0 || total <= 0 {
		return fmt.Errorf("weight metadata: no complete, sized default safetensors weights")
	}
	if len(names) > 512 {
		return fmt.Errorf("weight metadata exceeds 512-file inspection limit")
	}
	if e := validateShardNames(names); e != nil {
		return e
	}
	recipe, known, e := findRecipe(*m)
	if e != nil {
		return fmt.Errorf("weight metadata: cannot validate selected recipe")
	}
	if known && recipe.ID == "wan22-t2v-diffusers" {
		for _, component := range []string{"transformer", "transformer_2", "text_encoder", "vae"} {
			present := false
			for _, name := range names {
				if strings.HasPrefix(name, component+"/") {
					present = true
					break
				}
			}
			if !present {
				return fmt.Errorf("weight metadata: WAN 2.2 requires both denoising experts, text encoder and VAE weights")
			}
		}
	}
	sizes := make(map[string]int64)
	for _, f := range m.Hub.Siblings {
		sizes[f.Name] = f.Bytes()
	}
	// A manifest can reveal a shard missing entirely from the repository listing.
	manifests := 0
	var inspected int64
	for _, f := range m.Hub.Siblings {
		if !strings.HasSuffix(f.Name, ".safetensors.index.json") {
			continue
		}
		base := strings.TrimSuffix(f.Name, ".index.json")
		candidates, _ := weightFiles([]ModelFile{{Name: base, Size: 1}}, image)
		if len(candidates) == 0 {
			continue
		}
		if !safeWeightPath(f.Name) {
			return fmt.Errorf("weight metadata: invalid manifest path")
		}
		manifests++
		if manifests > 16 {
			return fmt.Errorf("weight metadata exceeds shard manifest inspection limit")
		}
		var raw json.RawMessage
		reader := *h
		if reader.Client == nil {
			reader.Client = NewHub().Client
		}
		if e := reader.get(ctx, "/"+m.Hub.ID+"/resolve/"+m.Hub.SHA+"/"+escapeWeightPath(f.Name), &raw); e != nil {
			return fmt.Errorf("weight metadata: cannot read required shard manifest")
		}
		inspected += int64(len(raw))
		if inspected > maxWeightHeadersTotal {
			return fmt.Errorf("weight metadata exceeds total metadata inspection limit")
		}
		fields, e := uniqueObject(raw)
		if e != nil {
			return fmt.Errorf("weight metadata: malformed shard manifest")
		}
		weightMap, e := uniqueObject(fields["weight_map"])
		if e != nil || len(weightMap) == 0 {
			return fmt.Errorf("weight metadata: empty or malformed shard manifest")
		}
		referenced := map[string]bool{}
		dir := f.Name[:strings.LastIndex(f.Name, "/")+1]
		for _, value := range weightMap {
			var name string
			if json.Unmarshal(value, &name) != nil || name == "" {
				return fmt.Errorf("weight metadata: invalid shard manifest path")
			}
			full := dir + name
			if !safeWeightPath(full) || !contains(names, full) {
				return fmt.Errorf("weight metadata: manifest references a missing or unselected shard")
			}
			referenced[full] = true
		}
		for _, name := range names {
			if match := shardName.FindStringSubmatch(name); match != nil && match[1]+".safetensors" == base && !referenced[name] {
				return fmt.Errorf("weight metadata: shard absent from manifest")
			}
		}
	}
	for _, name := range names {
		if !safeWeightPath(name) {
			return fmt.Errorf("weight metadata: invalid weight path")
		}
		size := sizes[name]
		if size < 10 {
			return fmt.Errorf("weight metadata: invalid safetensors file size")
		}
		prefix, e := h.weightRange(ctx, m, name, 0, 7, size)
		if e != nil {
			return e
		}
		n := binary.LittleEndian.Uint64(prefix)
		if n < 2 || n > maxWeightHeader || n > uint64(size-8) {
			return fmt.Errorf("weight metadata: invalid or oversized safetensors header")
		}
		inspected += int64(n) + 8
		if inspected > maxWeightHeadersTotal {
			return fmt.Errorf("weight metadata exceeds total header inspection limit")
		}
		header, e := h.weightRange(ctx, m, name, 8, int64(n)+7, size)
		if e != nil {
			return e
		}
		meta, e := parseWeightHeader(header, size)
		if e != nil {
			return e
		}
		meta.Name = name
		m.WeightMetadata = append(m.WeightMetadata, meta)
	}
	return nil
}

func safeWeightPath(name string) bool {
	for _, part := range strings.Split(name, "/") {
		if !repoPart.MatchString(part) || strings.Contains(part, "..") {
			return false
		}
	}
	return true
}
func escapeWeightPath(name string) string {
	parts := strings.Split(name, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}
func validateShardNames(names []string) error {
	groups := map[string]map[int]bool{}
	totals := map[string]int{}
	for _, name := range names {
		match := shardName.FindStringSubmatch(name)
		if match == nil {
			continue
		}
		i, e1 := strconv.Atoi(match[2])
		n, e2 := strconv.Atoi(match[3])
		if e1 != nil || e2 != nil || i < 1 || n < 1 || i > n || n > 512 {
			return fmt.Errorf("weight metadata: invalid shard numbering")
		}
		key := match[1]
		if groups[key] == nil {
			groups[key] = map[int]bool{}
			totals[key] = n
		}
		if totals[key] != n || groups[key][i] {
			return fmt.Errorf("weight metadata: inconsistent shard numbering")
		}
		groups[key][i] = true
	}
	for key, entries := range groups {
		if len(entries) != totals[key] {
			return fmt.Errorf("weight metadata: missing safetensors shard")
		}
	}
	return nil
}

func (h *Hub) weightRange(ctx context.Context, m *Model, name string, start, end, size int64) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimSuffix(h.BaseURL, "/")+"/"+m.Hub.ID+"/resolve/"+m.Hub.SHA+"/"+escapeWeightPath(name), nil)
	if err != nil {
		return nil, fmt.Errorf("weight metadata: cannot create header request")
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start, end))
	req.Header.Set("Accept-Encoding", "identity")
	req.Header.Set("User-Agent", "modelship/0.1")
	if h.Token != "" {
		req.Header.Set("Authorization", "Bearer "+h.Token)
	}
	client := h.Client
	if client == nil {
		client = NewHub().Client
	}
	resp, err := client.Do(req)
	// Do not wrap transport errors: they can contain signed redirect URLs/tokens.
	if err != nil {
		return nil, fmt.Errorf("weight metadata: header request failed (check network and model access)")
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusPartialContent {
		return nil, fmt.Errorf("weight metadata: server must honor header-only HTTP Range (HTTP %d)", resp.StatusCode)
	}
	want := fmt.Sprintf("bytes %d-%d/%d", start, end, size)
	if resp.Header.Get("Content-Range") != want || (resp.ContentLength >= 0 && resp.ContentLength != end-start+1) || (resp.Header.Get("Content-Encoding") != "" && resp.Header.Get("Content-Encoding") != "identity") {
		return nil, fmt.Errorf("weight metadata: inconsistent header range or file size")
	}
	data := make([]byte, end-start+1)
	if _, err = io.ReadFull(resp.Body, data); err != nil {
		return nil, fmt.Errorf("weight metadata: incomplete safetensors header range")
	}
	return data, nil
}

// uniqueObject rejects duplicate JSON keys, which encoding/json otherwise
// silently overwrites. It is used for both the header and tensor descriptors.
func uniqueObject(data []byte) (map[string]json.RawMessage, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	token, err := d.Token()
	if err != nil || token != json.Delim('{') {
		return nil, fmt.Errorf("invalid object")
	}
	result := map[string]json.RawMessage{}
	for d.More() {
		token, err = d.Token()
		if err != nil {
			return nil, err
		}
		key, ok := token.(string)
		if !ok {
			return nil, fmt.Errorf("invalid key")
		}
		if _, exists := result[key]; exists {
			return nil, fmt.Errorf("duplicate key")
		}
		var raw json.RawMessage
		if err = d.Decode(&raw); err != nil {
			return nil, err
		}
		result[key] = raw
	}
	if _, err = d.Token(); err != nil {
		return nil, err
	}
	if _, err = d.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing data")
	}
	return result, nil
}

func parseWeightHeader(header []byte, fileSize int64) (WeightFileMetadata, error) {
	bad := func() (WeightFileMetadata, error) {
		return WeightFileMetadata{}, fmt.Errorf("weight metadata: malformed safetensors shapes, dtypes, offsets or sizes")
	}
	result := WeightFileMetadata{Size: fileSize}
	if len(header) == 0 || len(header) > maxWeightHeader || fileSize < 8 || int64(len(header)) > fileSize-8 || header[0] != '{' || !utf8.Valid(header) {
		return bad()
	}
	entries, err := uniqueObject(header)
	if err != nil {
		return bad()
	}
	type span struct{ start, end int64 }
	var spans []span
	for name, raw := range entries {
		if name == "__metadata__" {
			values, e := uniqueObject(raw)
			if e != nil {
				return bad()
			}
			for _, value := range values {
				var text string
				if json.Unmarshal(value, &text) != nil || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
					return bad()
				}
			}
			continue
		}
		fields, e := uniqueObject(raw)
		if e != nil || len(fields) != 3 || fields["dtype"] == nil || fields["shape"] == nil || fields["data_offsets"] == nil {
			return bad()
		}
		var dtype string
		var shape []uint64
		var offsets []uint64
		if json.Unmarshal(fields["dtype"], &dtype) != nil || json.Unmarshal(fields["shape"], &shape) != nil || shape == nil || len(shape) > 64 || json.Unmarshal(fields["data_offsets"], &offsets) != nil || len(offsets) != 2 {
			return bad()
		}
		var width uint64
		floating := false
		switch dtype {
		case "BOOL", "I8", "U8":
			width = 1
		case "I16", "U16":
			width = 2
		case "I32", "U32":
			width = 4
		case "I64", "U64":
			width = 8
		case "F16", "BF16":
			width = 2
			floating = true
		case "F32":
			width = 4
			floating = true
		case "F64":
			width = 8
			floating = true
		default:
			return bad()
		}
		elements := uint64(1)
		for _, dim := range shape {
			if dim > math.MaxInt64 || (dim != 0 && elements > math.MaxInt64/dim) {
				return bad()
			}
			elements *= dim
		}
		if elements > math.MaxInt64/width {
			return bad()
		}
		count := elements * width
		payload := fileSize - int64(len(header)) - 8
		if payload < 0 || offsets[0] > offsets[1] || offsets[1] > uint64(payload) || offsets[1]-offsets[0] != count {
			return bad()
		}
		spans = append(spans, span{int64(offsets[0]), int64(offsets[1])})
		if floating {
			if result.FloatingElements > math.MaxInt64-int64(elements) {
				return bad()
			}
			result.FloatingElements += int64(elements)
		} else {
			if result.IntegerBytes > math.MaxInt64-int64(count) {
				return bad()
			}
			result.IntegerBytes += int64(count)
		}
	}
	if len(spans) == 0 || result.FloatingElements == 0 && result.IntegerBytes == 0 {
		return bad()
	}
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].start == spans[j].start {
			return spans[i].end < spans[j].end
		}
		return spans[i].start < spans[j].start
	})
	var end int64
	for _, s := range spans {
		if s.start != end {
			return bad()
		}
		end = s.end
	}
	if end != fileSize-int64(len(header))-8 {
		return bad()
	}
	return result, nil
}

func residentWeightBytes(m Model, names []string, download int64, dtype string, overrides map[string]string) (int64, string, error) {
	if m.WeightMetadataChecked {
		if m.WeightMetadataError != "" {
			return 0, "unavailable: safetensors header inspection failed", fmt.Errorf("%s", m.WeightMetadataError)
		}
		if len(m.WeightMetadata) != len(names) || len(names) == 0 {
			return 0, "unavailable", fmt.Errorf("weight metadata: incomplete selected weight coverage")
		}
		sizes := map[string]int64{}
		for _, file := range m.Hub.Siblings {
			sizes[file.Name] = file.Bytes()
		}
		seen := map[string]bool{}
		var total int64
		for _, meta := range m.WeightMetadata {
			if !contains(names, meta.Name) || seen[meta.Name] || meta.FloatingElements < 0 || meta.IntegerBytes < 0 || (len(m.Hub.Siblings) > 0 && sizes[meta.Name] != meta.Size) {
				return 0, "unavailable", fmt.Errorf("weight metadata: inconsistent selected weight coverage")
			}
			seen[meta.Name] = true
			component := strings.Split(meta.Name, "/")[0]
			target := dtype
			if value := overrides[component]; value != "" {
				target = value
			}
			width := int64(2)
			if target == "float32" {
				width = 4
			} else if target != "float16" && target != "bfloat16" {
				return 0, "unavailable", fmt.Errorf("weight metadata: unsupported requested dtype")
			}
			if meta.FloatingElements > (math.MaxInt64-meta.IntegerBytes)/width {
				return 0, "unavailable", fmt.Errorf("weight metadata: resident byte overflow")
			}
			size := meta.FloatingElements*width + meta.IntegerBytes
			if total > math.MaxInt64-size {
				return 0, "unavailable", fmt.Errorf("weight metadata: resident byte overflow")
			}
			total += size
		}
		if total <= 0 {
			return 0, "unavailable", fmt.Errorf("weight metadata: empty resident tensor estimate")
		}
		return total, "validated safetensors headers at immutable SHA; minimum requested-dtype residency; integer buffers preserved", nil
	}
	// Only pure/offline callers may use this fallback. CLI always probes first.
	// It deliberately does not assume FP32 files will shrink when cast to BF16.
	factor := int64(1)
	if dtype == "float32" {
		factor = 2
	}
	for _, target := range overrides {
		if target == "float32" {
			factor = 2
		}
	}
	if download <= 0 || download > math.MaxInt64/factor {
		return 0, "unavailable", fmt.Errorf("weight metadata: invalid fallback weight size")
	}
	total := download * factor
	if m.Hub.Safetensors.Total > 0 {
		if m.Hub.Safetensors.Total > math.MaxInt64/(2*factor) {
			return 0, "unavailable", fmt.Errorf("weight metadata: fallback parameter count overflow")
		}
		total = max64(total, m.Hub.Safetensors.Total*2*factor)
	}
	return total, "unprobed conservative file-size/parameter-count fallback (offline plans only)", nil
}
