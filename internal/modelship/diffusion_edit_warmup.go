package modelship

const diffusionEditWarmupPrompt = "Change the red mug to a blue mug. Keep the yellow lemon, tabletop, and composition unchanged."

// Same synthetic reference as the isolated GPU smoke; no user data, decoder,
// remote file or cache needed to probe the supervised edit path.
func diffusionEditWarmupReference() []byte {
	square := func(x float64) float64 { return x * x }
	out := make([]byte, diffusionEditRGBBytes)
	for y := 0; y < diffusionSide; y++ {
		for x := 0; x < diffusionSide; x++ {
			shade := (x + 2*y) / 80
			r, g, b := 235-shade, 224-shade, 207-shade
			if y >= 355 {
				r, g, b = 192-shade, 170-shade, 141-shade
			}
			ellipse := func(cx, cy, rx, ry int) float64 {
				return square(float64(x-cx)/float64(rx)) + square(float64(y-cy)/float64(ry))
			}
			if ellipse(228, 395, 150, 20) < 1 {
				r, g, b = 164-shade, 144-shade, 117-shade
			}
			if handle := ellipse(313, 285, 51, 48); 0.5 < handle && handle < 1 {
				r, g, b = 183, 38, 32
			}
			if x >= 138 && x < 324 && y >= 222 && y < 391 && ellipse(231, 315, 96, 109) < 1.55 {
				r, g, b = 186+(x-138)/4, 43+(x-138)/10, 37+(x-138)/12
			}
			if ellipse(231, 222, 93, 26) < 1 {
				r, g, b = 231, 70, 61
			}
			if ellipse(231, 220, 81, 17) < 1 {
				r, g, b = 103, 28, 25
			}
			if ellipse(406, 365, 49, 35) < 1 {
				r, g, b = 250, 207-max(x-397, 397-x)/2, 56
			}
			i := (y*diffusionSide + x) * 3
			out[i], out[i+1], out[i+2] = byte(r), byte(g), byte(b)
		}
	}
	return out
}
