package modelship

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"strings"
)

const visionMaxBytes = 4 << 20
const visionMaxPixels = 4 * 1024 * 1024

type ggufContentPart struct {
	Type     string  `json:"type"`
	Text     *string `json:"text,omitempty"`
	ImageURL *struct {
		URL string `json:"url"`
	} `json:"image_url,omitempty"`
}

func strictJSON(data []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if e := d.Decode(out); e != nil {
		return e
	}
	if d.Decode(new(any)) != io.EOF {
		return fmt.Errorf("unexpected trailing JSON")
	}
	return nil
}

type ggufCheckedImage struct {
	url    string
	data   []byte
	pixels int64
}

// Walk the complete bounded PNG container, not compressed pixels. Go's PNG
// decoder ignores animation chunks, whereas the runtime can interpret them.
func checkGGUFStaticPNG(b []byte) error {
	if !bytes.HasPrefix(b, []byte("\x89PNG\r\n\x1a\n")) {
		return fmt.Errorf("invalid PNG signature")
	}
	for pos := 8; pos < len(b); {
		if len(b)-pos < 12 {
			return fmt.Errorf("truncated PNG chunk")
		}
		n := uint64(binary.BigEndian.Uint32(b[pos : pos+4]))
		if n > uint64(len(b)-pos-12) {
			return fmt.Errorf("invalid PNG chunk length")
		}
		end := pos + 8 + int(n)
		kind := string(b[pos+4 : pos+8])
		if kind == "acTL" || kind == "fcTL" || kind == "fdAT" {
			return fmt.Errorf("animated PNG is not supported; static PNG/JPEG only")
		}
		if crc32.ChecksumIEEE(b[pos+4:end]) != binary.BigEndian.Uint32(b[end:end+4]) {
			return fmt.Errorf("invalid PNG chunk checksum")
		}
		pos = end + 4
		if kind == "IEND" {
			if n != 0 || pos != len(b) {
				return fmt.Errorf("invalid PNG end/trailing bytes")
			}
			return nil
		}
	}
	return fmt.Errorf("missing PNG end")
}

// This preflight only decodes bounded base64 and image headers/container. All
// conversation-wide limits are admitted before any full pixel decode.
func inspectGGUFVisionDataURL(url string) (ggufCheckedImage, error) {
	var out ggufCheckedImage
	mime := ""
	for _, v := range []string{"image/png", "image/jpeg"} {
		if strings.HasPrefix(url, "data:"+v+";base64,") {
			mime = v
			break
		}
	}
	if mime == "" || len(url) > ((visionMaxBytes+2)/3)*4+64 {
		return out, fmt.Errorf("vision accepts PNG/JPEG data URLs only, at most 4 MiB; remote/file URLs are not fetched")
	}
	payload := strings.TrimPrefix(url, "data:"+mime+";base64,")
	b, e := base64.StdEncoding.Strict().DecodeString(payload)
	if e != nil || len(b) == 0 || len(b) > visionMaxBytes || base64.StdEncoding.EncodeToString(b) != payload {
		return out, fmt.Errorf("invalid or oversized image encoding")
	}
	cfg, format, e := image.DecodeConfig(bytes.NewReader(b))
	if e != nil || cfg.Width < 1 || cfg.Height < 1 || cfg.Width > 4096 || cfg.Height > 4096 || cfg.Width > visionMaxPixels/cfg.Height {
		return out, fmt.Errorf("invalid image dimensions or 4-megapixel input limit exceeded")
	}
	if mime == "image/png" && format != "png" || mime == "image/jpeg" && format != "jpeg" {
		return out, fmt.Errorf("image MIME type does not match decoded bytes")
	}
	if format == "png" {
		if e := checkGGUFStaticPNG(b); e != nil {
			return out, e
		}
	}
	return ggufCheckedImage{url: "data:" + mime + ";base64," + payload, data: b, pixels: int64(cfg.Width) * int64(cfg.Height)}, nil
}
func checkedVisionDataURL(url string) (string, error) {
	im, e := inspectGGUFVisionDataURL(url)
	if e != nil {
		return "", e
	}
	if _, _, e = image.Decode(bytes.NewReader(im.data)); e != nil {
		return "", fmt.Errorf("image could not be decoded safely")
	}
	return im.url, nil
}

// Preserve the legacy helper/API: no saved image policy means one image.
func validateGGUFMessages(messages []ggufMessage, vision bool) ([]map[string]any, error) {
	return validateGGUFMessagesWithPolicy(messages, vision, nil)
}
func validateGGUFMessagesWithPolicy(messages []ggufMessage, vision bool, policy *ImagePolicy) ([]map[string]any, error) {
	maxImages := 1
	if policy != nil {
		if !vision || !reviewedGGUFImages(policy) {
			return nil, fmt.Errorf("unsupported saved GGUF image policy")
		}
		maxImages = policy.MaxImages
	}
	normalized := make([]map[string]any, 0, len(messages))
	var urls []string
	for _, m := range messages {
		if !contains([]string{"system", "user", "assistant"}, m.Role) {
			return nil, fmt.Errorf("invalid text chat role")
		}
		var text string
		if json.Unmarshal(m.Content, &text) == nil && len(m.Content) > 0 && m.Content[0] == '"' {
			if strings.TrimSpace(text) == "" || len(text) > 32768 {
				return nil, fmt.Errorf("invalid text chat message")
			}
			normalized = append(normalized, map[string]any{"role": m.Role, "content": text})
			continue
		}
		if !vision || m.Role != "user" {
			return nil, fmt.Errorf("this deployment/role does not accept image message content")
		}
		var parts []ggufContentPart
		if strictJSON(m.Content, &parts) != nil || len(parts) < 1 || len(parts) > max(4, maxImages*2) {
			return nil, fmt.Errorf("invalid image chat content parts")
		}
		clean := make([]map[string]any, 0, len(parts))
		textBytes := 0
		for _, p := range parts {
			switch p.Type {
			case "text":
				if p.Text == nil || p.ImageURL != nil || strings.TrimSpace(*p.Text) == "" {
					return nil, fmt.Errorf("invalid image chat text part")
				}
				textBytes += len(*p.Text)
				clean = append(clean, map[string]any{"type": "text", "text": *p.Text})
			case "image_url":
				if p.ImageURL == nil || p.Text != nil {
					return nil, fmt.Errorf("invalid image part")
				}
				urls = append(urls, p.ImageURL.URL)
				if len(urls) > maxImages {
					return nil, fmt.Errorf("this vision profile accepts at most %d image(s) per conversation request", maxImages)
				}
				clean = append(clean, map[string]any{"type": "image_url", "image_url": map[string]string{"url": p.ImageURL.URL}})
			default:
				return nil, fmt.Errorf("unsupported multimodal content type; images only, no tools/audio/video")
			}
		}
		if textBytes < 1 || textBytes > 32768 {
			return nil, fmt.Errorf("image messages require bounded accompanying text")
		}
		normalized = append(normalized, map[string]any{"role": m.Role, "content": clean})
	}
	// Count and shape validation above precede even base64/header inspection.
	var totalBytes, totalPixels int64
	checked := make([]ggufCheckedImage, 0, len(urls))
	for _, url := range urls {
		im, e := inspectGGUFVisionDataURL(url)
		if e != nil {
			return nil, e
		}
		totalBytes += int64(len(im.data)) // encoded PNG/JPEG bytes, before base64
		totalPixels += im.pixels
		if policy != nil && (totalBytes > policy.MaxTotalBytes || totalPixels > policy.MaxTotalPixels) {
			return nil, fmt.Errorf("aggregate image byte/pixel budget exceeded across conversation")
		}
		checked = append(checked, im)
	}
	for _, im := range checked {
		if _, _, e := image.Decode(bytes.NewReader(im.data)); e != nil {
			return nil, fmt.Errorf("image could not be decoded safely")
		}
	}
	return normalized, nil
}
func (g *ggufGateway) visionEnabled() bool { return g.plan.GGUF != nil && g.plan.GGUF.Vision != nil }
func (g *ggufGateway) imagePolicy() *ImagePolicy {
	if g.visionEnabled() {
		return g.plan.GGUF.Vision.Images
	}
	return nil
}
func (g *ggufGateway) inputModalities() []string {
	if g.visionEnabled() {
		return []string{"text", "image"}
	}
	return []string{"text"}
}
func (g *ggufGateway) capabilities() map[string]any {
	chat := map[string]any{"image_input": g.visionEnabled(), "streaming": false, "tools": false, "max_output_tokens": 1024}
	if g.visionEnabled() {
		chat["max_images"] = 1
		if p := g.imagePolicy(); reviewedGGUFImages(p) {
			chat["max_images"] = p.MaxImages
			chat["max_total_image_bytes"] = p.MaxTotalBytes
			chat["max_total_image_pixels"] = p.MaxTotalPixels
		}
		chat["max_image_bytes"] = visionMaxBytes
		chat["max_image_pixels"] = visionMaxPixels
		chat["max_image_dimension"] = 4096
		chat["image_mime_types"] = []string{"image/png", "image/jpeg"}
		chat["max_image_tokens"] = visionImageMaxTokens
		chat["recommended_image_side"] = 768
	}
	return map[string]any{"schema_version": 1, "model": g.plan.Model, "revision": g.plan.Revision, "run_id": g.runID, "runtime": "llama.cpp", "input_modalities": g.inputModalities(), "output_modalities": []string{"text"}, "tasks": []string{"text-generation"}, "context_tokens": g.plan.Context, "chat": chat}
}
