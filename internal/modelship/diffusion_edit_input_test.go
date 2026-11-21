package modelship

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"strings"
	"testing"
)

func TestDiffusionEditWarmupPin(t *testing.T) {
	const smokeRGB = "49812638e1e498eaff84ba18736947cea6428137a7664149137119ad290949c2"
	if got := fmt.Sprintf("%x", sha256.Sum256(diffusionEditWarmupReference())); got != smokeRGB {
		t.Fatalf("warm-up reference differs from GPU-smoked fixture: %s", got)
	}
}

func TestDiffusionEditRequestFields(t *testing.T) {
	const model = "example/image"
	parse := func(body string) (diffusionRequest, error) {
		return parseDiffusionImageRequest([]byte(body), model, true)
	}
	body := `{"prompt":"make the red mug blue","image_b64":"AA==","seed":0,"width":512,"height":512,"steps":40,"guidance_scale":6,"model":"example/image"}`
	req, err := parse(body)
	if err != nil || req.reference != "AA==" || req.seed == nil || *req.seed != 0 {
		t.Fatalf("edit parse: %#v %v", req, err)
	}
	if _, err := parseDiffusionRequest([]byte(body), model); err == nil {
		t.Fatal("generation accepted an edit image")
	}
	for _, bad := range []string{
		`{"prompt":"hello"}`,
		`{"prompt":"hello","image_b64":null}`,
		`{"prompt":"hello","image_b64":[]}`,
		`{"prompt":"hello","image_b64":""}`,
		`{"prompt":"hello","image_b64":"AA==","image_b64":"AA=="}`,
		`{"prompt":"hello","image_b64":"AA==","width":1024}`,
		`{"prompt":"hello","image_b64":"AA==","strength":0.7}`,
		`{"prompt":"hello","image_b64":"AA==","lora_strength":0}`,
		`{"prompt":"hello","image_b64":"AA==","image_url":"https://example.com"}`,
		`{"prompt":"hello","image_b64":"AA=="} {}`,
	} {
		if _, err := parse(bad); err == nil {
			t.Errorf("accepted %s", bad)
		}
	}
}

func TestDiffusionEditReference(t *testing.T) {
	encode := base64.StdEncoding.EncodeToString
	makePNG := func(im image.Image) []byte {
		t.Helper()
		var buf bytes.Buffer
		if err := png.Encode(&buf, im); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	im := image.NewNRGBA(image.Rect(0, 0, 512, 512))
	for i := 0; i < len(im.Pix); i += 4 {
		im.Pix[i], im.Pix[i+1], im.Pix[i+2], im.Pix[i+3] = 255, 3, 64, 255
	}
	im.SetNRGBA(511, 511, color.NRGBA{R: 4, G: 201, B: 251, A: 255})
	data := makePNG(im)
	pixels, err := decodeDiffusionEditReference(encode(data))
	if err != nil || len(pixels) != 512*512*3 || !bytes.Equal(pixels[:3], []byte{255, 3, 64}) || !bytes.Equal(pixels[len(pixels)-3:], []byte{4, 201, 251}) {
		t.Fatalf("invalid RGB conversion: %v, %d bytes", err, len(pixels))
	}
	var compressed bytes.Buffer
	if err := jpeg.Encode(&compressed, im, nil); err != nil {
		t.Fatal(err)
	}
	jpegData := compressed.Bytes()
	// An EOI byte pair in metadata is not the end of the image.
	withMetadata := append([]byte{0xff, 0xd8, 0xff, 0xfe, 0, 4, 0xff, 0xd9}, jpegData[2:]...)
	for _, valid := range [][]byte{jpegData, withMetadata} {
		if pixels, err := decodeDiffusionEditReference(encode(valid)); err != nil || len(pixels) != 512*512*3 {
			t.Fatalf("JPEG decode: %v", err)
		}
	}
	im.Pix[3] = 254
	for _, bad := range []string{
		"", "data:image/png;base64," + encode(data), "https://example.com/image.png", "AA==\n", "AB==",
		strings.Repeat("A", diffusionEditEncodedBytes+1),
		encode(makePNG(image.NewGray(image.Rect(0, 0, 512, 511)))),
		encode(makePNG(im)), encode(data[:len(data)-1]), encode(append(append([]byte(nil), data...), 0)),
		encode(jpegData[:len(jpegData)-1]), encode(append(append([]byte(nil), jpegData...), 0)),
		encode(append(append([]byte(nil), jpegData...), jpegData...)),
	} {
		if pixels, err := decodeDiffusionEditReference(bad); pixels != nil || !errors.Is(err, errDiffusionReference) {
			t.Errorf("invalid reference not rejected: len=%d, err=%v", len(bad), err)
		}
	}
}
