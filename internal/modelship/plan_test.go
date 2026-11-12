package modelship

import (
	"encoding/json"
	"strings"
	"testing"
)

func hostFixture() Host {
	return Host{OS: "linux", Arch: "amd64", Docker: true, GPUs: []GPU{{Index: 0, UUID: "GPU-test", Name: "RTX 4090", TotalMiB: 24576, FreeMiB: 24000, Driver: "575.57.08", Compute: "8.9"}}}
}
func optsFixture() Options {
	return Options{Task: "auto", GPU: 0, Context: 4096, Port: 8000, Name: "modelship", Revision: "main"}
}
func textFixture() Model {
	m := Model{Hub: HubModel{ID: "org/model", SHA: strings.Repeat("a", 40), PipelineTag: "text-generation", Siblings: []ModelFile{{Name: "model.safetensors", Size: 2 * 1024 * 1024 * 1024}}}, Config: ModelConfig{Architectures: []string{"LlamaForCausalLM"}, Layers: 32, Heads: 32, KVHeads: 8, Hidden: 4096, MaxPosition: 8192}}
	m.Hub.Safetensors.Total = 1_000_000_000
	return m
}
func imageFixture() Model {
	return Model{Hub: HubModel{ID: "org/image", SHA: strings.Repeat("b", 40), Siblings: []ModelFile{
		{Name: "unet/diffusion_pytorch_model.safetensors", Size: 5 * 1024 * 1024 * 1024},
		{Name: "unet/diffusion_pytorch_model.fp16.safetensors", Size: 2 * 1024 * 1024 * 1024},
		{Name: "text_encoder/model.safetensors", Size: 1024 * 1024 * 1024},
		{Name: "vae/diffusion_pytorch_model.safetensors", Size: 300 * 1024 * 1024},
	}}, Index: ModelIndex{Class: "StableDiffusionXLPipeline"}}
}
func TestTextPlan(t *testing.T) {
	p, err := BuildPlan(textFixture(), hostFixture(), optsFixture())
	if err != nil {
		t.Fatal(err)
	}
	if p.Runtime != "vllm" || p.Task != "text-generation" || p.Context != 4096 || len(p.Blockers) != 0 || len(p.Files) != 1 {
		t.Fatalf("%+v", p)
	}
	if p.EstimatedVRAMMiB < 4000 || p.EstimatedVRAMMiB > 6000 {
		t.Fatalf("unexpected estimate %d", p.EstimatedVRAMMiB)
	}
}
func TestImageTasks(t *testing.T) {
	for _, task := range []string{"auto", "text-to-image", "image-to-image"} {
		o := optsFixture()
		o.Task = task
		p, err := BuildPlan(imageFixture(), hostFixture(), o)
		if err != nil {
			t.Fatal(err)
		}
		if p.Runtime != "diffusers" || p.Context != 0 || len(p.Files) != 3 || len(p.Blockers) > 0 {
			t.Fatalf("%+v", p)
		}
	}
}
func TestModelRejections(t *testing.T) {
	tests := []struct {
		name   string
		change func(*Model, *Options)
	}{
		{"architecture", func(m *Model, o *Options) { m.Config.Architectures = []string{"UntrustedModel"} }},
		{"quantized", func(m *Model, o *Options) { m.Config.Quantization = json.RawMessage(`{"quant_method":"awq"}`) }},
		{"context", func(m *Model, o *Options) { o.Context = 16000 }},
		{"weights", func(m *Model, o *Options) { m.Hub.Siblings = nil }},
		{"unknown-shard-size", func(m *Model, o *Options) {
			m.Hub.Siblings = append(m.Hub.Siblings, ModelFile{Name: "model-00002-of-00002.safetensors"})
		}},
		{"task", func(m *Model, o *Options) { o.Task = "image-to-image" }},
		{"name", func(m *Model, o *Options) { o.Name = "--privileged" }},
		{"port", func(m *Model, o *Options) { o.Port = 80 }},
		{"gpu", func(m *Model, o *Options) { o.GPU = 9 }},
		{"non-text", func(m *Model, o *Options) { m.Hub.PipelineTag = "feature-extraction" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := textFixture()
			o := optsFixture()
			tc.change(&m, &o)
			if _, err := BuildPlan(m, hostFixture(), o); err == nil {
				t.Fatal("wanted error")
			}
		})
	}
	m := imageFixture()
	m.Index.Class = "CustomPipeline"
	if _, err := BuildPlan(m, hostFixture(), optsFixture()); err == nil {
		t.Fatal("custom pipeline accepted")
	}
}
func TestMemoryOverrideCannotOverrideDriver(t *testing.T) {
	h := hostFixture()
	h.GPUs[0].FreeMiB = 1000
	h.GPUs[0].Driver = "550.54.15"
	p, err := BuildPlan(textFixture(), h, optsFixture())
	if err != nil || len(p.Blockers) != 3 {
		t.Fatalf("%+v %v", p, err)
	}
	o := optsFixture()
	o.AllowRisk = true
	p, err = BuildPlan(textFixture(), h, o)
	if err != nil || len(p.Blockers) != 2 || !strings.Contains(p.Blockers[0], "driver") || !strings.Contains(p.Blockers[1], "currently free") {
		t.Fatalf("%+v %v", p, err)
	}
}
func TestFluxCapabilityAndMemory(t *testing.T) {
	m := imageFixture()
	m.Index.Class = "FluxPipeline"
	h := hostFixture()
	h.GPUs[0].Compute = "7.5"
	o := optsFixture()
	o.AllowRisk = true
	p, err := BuildPlan(m, h, o)
	if err != nil || len(p.Blockers) != 1 {
		t.Fatalf("%+v %v", p, err)
	}
}
func TestGPUParsing(t *testing.T) {
	g, err := ParseGPUs("0, GPU-one, NVIDIA RTX 4090, 24564, 24000, 575.57.08, 8.9\n1, GPU-two, NVIDIA A100, 81920, 81000, 575.57.08, 8.0")
	if err != nil || len(g) != 2 || g[1].FreeMiB != 81000 {
		t.Fatalf("%+v %v", g, err)
	}
	for _, s := range []string{"", "garbage", "0, GPU-x, X, 10, N/A, 570.26, 8.0", "0, GPU-x, X, 10, 20, 570.26, 8.0"} {
		if _, err := ParseGPUs(s); err == nil {
			t.Fatalf("accepted %q", s)
		}
	}
}
func TestBuildArgs(t *testing.T) {
	t.Setenv("HF_TOKEN", "super-secret")
	p, err := BuildPlan(textFixture(), hostFixture(), optsFixture())
	if err != nil {
		t.Fatal(err)
	}
	args := BuildArgs(p)
	s := strings.Join(args, " ")
	for _, want := range []string{"127.0.0.1:8000:8000", "device=GPU-test", "--revision " + p.Revision, "--env HF_TOKEN", "--load-format safetensors"} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
	for _, bad := range []string{"super-secret", "--trust-remote-code", "--privileged", "--network host", "sh -c"} {
		if strings.Contains(s, bad) {
			t.Fatalf("unsafe args: %s", s)
		}
	}
	p, err = BuildPlan(imageFixture(), hostFixture(), optsFixture())
	if err != nil {
		t.Fatal(err)
	}
	s = strings.Join(BuildArgs(p), " ")
	if !strings.Contains(s, "MODEL_REVISION="+p.Revision) || !strings.Contains(s, "MODEL_PIPELINE=StableDiffusionXLPipeline") {
		t.Fatal(s)
	}
}
