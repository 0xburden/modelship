package modelship

import "fmt"

// ImagePolicy is saved execution behavior, never inferred from a model name.
// A nil policy preserves the original one-image contract.
type ImagePolicy struct {
	MaxImages      int   `json:"max_images"`
	MaxTotalBytes  int64 `json:"max_total_image_bytes"`
	MaxTotalPixels int64 `json:"max_total_image_pixels"`
}

func reviewedImagePolicy() ImagePolicy {
	return ImagePolicy{MaxImages: 4, MaxTotalBytes: 8 << 20, MaxTotalPixels: 8 * 1024 * 1024}
}
func validateImagePolicy(p *ImagePolicy) error {
	if p != nil && *p != reviewedImagePolicy() {
		return fmt.Errorf("image policy differs from the reviewed aggregate limits")
	}
	return nil
}
