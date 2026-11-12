package modelship

import "fmt"

func applyRecipePolicy(p *Plan, r Recipe, m Model, o Options) {
	p.Research = o.Research
	p.TrustModelCode = o.TrustModelCode && r.RequiresCustomCode
	if r.RequiresResearch && !o.Research {
		p.Blockers = append(p.Blockers, "this recipe requires --research acknowledgment of non-commercial research/evaluation use under the model license; this does not authorize commercial use")
	}
	if r.RequiresCustomCode {
		p.CodeRevision = r.CodeRevision
		p.CodeHashes = make(map[string]string, len(r.CodeHashes))
		for name, digest := range r.CodeHashes {
			p.CodeHashes[name] = digest
		}
		if !o.TrustModelCode {
			p.Blockers = append(p.Blockers, "repository Python requires separate --trust-model-code opt-in after reviewing the pinned source; research use and --experimental do not grant code execution consent")
		}
		if m.Hub.SHA != r.CodeRevision {
			p.Blockers = append(p.Blockers, "custom-code recipe only accepts its reviewed immutable revision; use --revision "+r.CodeRevision)
		}
	} else if o.TrustModelCode {
		p.Warnings = append(p.Warnings, "--trust-model-code is unused for this recipe; standard model loaders still reject repository Python")
	}
}

// Explicit selection is deliberately narrow. Never infer a substitute model or
// silently load a converted/quantized community repository on the user's behalf.
func resolveWeightRepo(requested, selected string) (string, error) {
	if selected == "" {
		return requested, nil
	}
	id, err := ParseModel(selected)
	if err != nil {
		return "", err
	}
	if requested != "Wan-AI/Wan2.2-T2V-A14B" || id != "Wan-AI/Wan2.2-T2V-A14B-Diffusers" {
		return "", fmt.Errorf("--weights-repo only supports explicitly selecting Wan-AI/Wan2.2-T2V-A14B-Diffusers for Wan-AI/Wan2.2-T2V-A14B; no automatic compatibility or substitution is assumed")
	}
	return id, nil
}
