package modelship

import (
	"fmt"
	"path/filepath"
	"strings"
)

const ggufProjectorFile = "mmproj-F16.gguf"
const visionImageMaxTokens = 1024

func validateGGUFVision(v *GGUFVisionSpec) error {
	if v == nil {
		return fmt.Errorf("missing reviewed vision projector")
	}
	pin, e := ggufArtifact(ggufRepo, ggufRevision, v.File.Name, true)
	if e != nil {
		return e
	}
	m := v.Metadata
	a := m.Attributes
	if v.File != pin || m.Architecture != "clip" || m.TensorBytes != 927587264 || m.TensorCount != 334 || m.HeaderBytes != 20224 || a["general.type"] != "mmproj" || a["clip.projector_type"] != "qwen3vl_merger" || a["clip.has_vision_encoder"] != "1" || a["clip.vision.projection_dim"] != "5120" || a["clip.vision.patch_size"] != "16" || a["clip.vision.spatial_merge_size"] != "2" {
		return fmt.Errorf("vision projector metadata does not match the reviewed Qwen3.8 pairing")
	}
	for typ := range m.Types {
		if typ != 0 && typ != 1 {
			return fmt.Errorf("reviewed F16 vision projector contains an unexpected tensor encoding")
		}
	}
	return nil
}
func ggufSelectedArtifacts(s *GGUFSpec) []ArtifactPin {
	if s == nil {
		return nil
	}
	out := []ArtifactPin{s.File}
	if s.Vision != nil {
		out = append(out, s.Vision.File)
	}
	return out
}
func ggufArtifactPath(p Plan, file string) string {
	return filepath.Join(p.ModelCacheDir, "modelship-gguf", strings.ReplaceAll(p.Model, "/", "--"), p.Revision, file)
}
func ggufBudgets(s *GGUFSpec) (int64, int64) {
	bytes, resident := s.File.Size, s.Metadata.TensorBytes
	if s.Vision != nil {
		bytes += s.Vision.File.Size
		resident += s.Vision.Metadata.TensorBytes
	}
	return bytes, resident
}
