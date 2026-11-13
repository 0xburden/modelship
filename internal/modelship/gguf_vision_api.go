package modelship

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
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
func checkedVisionDataURL(url string) (string, error) {
	mime := ""
	for _, v := range []string{"image/png", "image/jpeg"} {
		if strings.HasPrefix(url, "data:"+v+";base64,") {
			mime = v
			break
		}
	}
	if mime == "" || len(url) > ((visionMaxBytes+2)/3)*4+64 {
		return "", fmt.Errorf("vision accepts PNG/JPEG data URLs only, at most 4 MiB; remote/file URLs are not fetched")
	}
	payload := strings.TrimPrefix(url, "data:"+mime+";base64,")
	b, e := base64.StdEncoding.Strict().DecodeString(payload)
	if e != nil || len(b) == 0 || len(b) > visionMaxBytes || base64.StdEncoding.EncodeToString(b) != payload {
		return "", fmt.Errorf("invalid or oversized image encoding")
	}
	cfg, format, e := image.DecodeConfig(bytes.NewReader(b))
	if e != nil || cfg.Width < 1 || cfg.Height < 1 || cfg.Width > 4096 || cfg.Height > 4096 || cfg.Width > visionMaxPixels/cfg.Height {
		return "", fmt.Errorf("invalid image dimensions or 4-megapixel input limit exceeded")
	}
	if mime == "image/png" && format != "png" || mime == "image/jpeg" && format != "jpeg" {
		return "", fmt.Errorf("image MIME type does not match decoded bytes")
	}
	if _, _, e = image.Decode(bytes.NewReader(b)); e != nil {
		return "", fmt.Errorf("image could not be decoded safely")
	}
	return "data:" + mime + ";base64," + payload, nil
}
func validateGGUFMessages(messages []ggufMessage, vision bool) ([]map[string]any, error) {
	normalized := make([]map[string]any, 0, len(messages))
	images := 0
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
		if strictJSON(m.Content, &parts) != nil || len(parts) < 1 || len(parts) > 4 {
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
				images++
				if images > 1 {
					return nil, fmt.Errorf("this first vision profile accepts one image per conversation request")
				}
				url, e := checkedVisionDataURL(p.ImageURL.URL)
				if e != nil {
					return nil, e
				}
				clean = append(clean, map[string]any{"type": "image_url", "image_url": map[string]string{"url": url}})
			default:
				return nil, fmt.Errorf("unsupported multimodal content type; images only, no tools/audio/video")
			}
		}
		if textBytes < 1 || textBytes > 32768 {
			return nil, fmt.Errorf("image messages require bounded accompanying text")
		}
		normalized = append(normalized, map[string]any{"role": m.Role, "content": clean})
	}
	return normalized, nil
}
func (g *ggufGateway) visionEnabled() bool { return g.plan.GGUF != nil && g.plan.GGUF.Vision != nil }
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
		chat["max_image_bytes"] = visionMaxBytes
		chat["max_image_pixels"] = visionMaxPixels
		chat["max_image_dimension"] = 4096
		chat["image_mime_types"] = []string{"image/png", "image/jpeg"}
		chat["max_image_tokens"] = visionImageMaxTokens
		chat["recommended_image_side"] = 768
	}
	return map[string]any{"schema_version": 1, "model": g.plan.Model, "revision": g.plan.Revision, "runtime": "llama.cpp", "input_modalities": g.inputModalities(), "output_modalities": []string{"text"}, "tasks": []string{"text-generation"}, "context_tokens": g.plan.Context, "chat": chat}
}
