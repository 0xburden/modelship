package modelship

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"image"
	"strings"
)

const (
	diffusionEditImageBytes   = 2 << 20 // encoded PNG/JPEG file, before base64
	diffusionEditEncodedBytes = (diffusionEditImageBytes + 2) / 3 * 4
	diffusionEditRGBBytes     = diffusionSide * diffusionSide * 3
)

var errDiffusionReference = errors.New("image_b64 must be a single opaque 512×512 PNG or JPEG (at most 2 MiB), in canonical base64")

// The pinned edit worker accepts only raw 512-square RGB. Do not forward the
// original compressed input or silently resize, crop, or flatten transparency.
func decodeDiffusionEditReference(encoded string) ([]byte, error) {
	if len(encoded) == 0 || len(encoded) > diffusionEditEncodedBytes || strings.ContainsAny(encoded, "\r\n") {
		return nil, errDiffusionReference
	}
	data, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil || len(data) == 0 || len(data) > diffusionEditImageBytes {
		return nil, errDiffusionReference
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || cfg.Width != diffusionSide || cfg.Height != diffusionSide || format != "png" && format != "jpeg" {
		return nil, errDiffusionReference
	}
	if (format == "png" && checkGGUFStaticPNG(data) != nil) || (format == "jpeg" && !singleDiffusionJPEG(data)) {
		return nil, errDiffusionReference
	}
	decoded, actual, err := image.Decode(bytes.NewReader(data))
	if err != nil || actual != format || decoded.Bounds() != image.Rect(0, 0, diffusionSide, diffusionSide) {
		return nil, errDiffusionReference
	}
	pixels := make([]byte, diffusionEditRGBBytes)
	for y := 0; y < diffusionSide; y++ {
		for x := 0; x < diffusionSide; x++ {
			r, g, b, a := decoded.At(x, y).RGBA()
			if a != 0xffff {
				return nil, errDiffusionReference
			}
			i := (y*diffusionSide + x) * 3
			pixels[i], pixels[i+1], pixels[i+2] = byte(r>>8), byte(g>>8), byte(b>>8)
		}
	}
	return pixels, nil
}

// The JPEG decoder stops at EOI and does not check for trailing bytes. Walk
// markers to enforce one complete image; leave image validity to the decoder.
// Segment payloads may themselves contain FF D9 (e.g. an Exif thumbnail).
func singleDiffusionJPEG(data []byte) bool {
	if len(data) < 4 || data[0] != 0xff || data[1] != 0xd8 {
		return false
	}
	inScan := false
	for i := 2; i < len(data); {
		if inScan {
			n := bytes.IndexByte(data[i:], 0xff)
			if n < 0 {
				return false
			}
			i += n
		} else if data[i] != 0xff {
			return false
		}
		for i < len(data) && data[i] == 0xff {
			i++
		}
		if i == len(data) {
			return false
		}
		marker := data[i]
		i++
		if inScan && (marker == 0 || (marker >= 0xd0 && marker <= 0xd7)) {
			continue // stuffed FF or restart marker inside scan data
		}
		switch {
		case marker == 0xd9:
			return i == len(data)
		case marker == 0 || (marker >= 0xd0 && marker <= 0xd8):
			return false
		case marker == 1:
			inScan = false // standalone TEM marker; full decoder still validates
		default:
			if len(data)-i < 2 {
				return false
			}
			n := int(binary.BigEndian.Uint16(data[i:]))
			if n < 2 || n > len(data)-i {
				return false
			}
			i += n
			inScan = marker == 0xda // start of scan
		}
	}
	return false
}
