package modelship

import (
	"context"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
)

type WeightGroup struct {
	Folder       string `json:"folder"`
	Format       string `json:"format"`
	Files        int    `json:"files"`
	Bytes        int64  `json:"bytes"`
	UnknownSizes int    `json:"unknown_sizes,omitempty"`
}
type Inspection struct {
	SelectedWeightBytes int64 `json:"selected_weight_bytes,omitempty"`

	GGUF *GGUFSpec `json:"gguf,omitempty"`

	Model                 string        `json:"model"`
	Revision              string        `json:"revision"`
	Task                  string        `json:"repository_task"`
	Library               string        `json:"library"`
	Gated                 any           `json:"gated"`
	License               any           `json:"license"`
	SourceConfig          string        `json:"source_config,omitempty"`
	ConfigurationReadable bool          `json:"configuration_readable"`
	Architectures         []string      `json:"architectures,omitempty"`
	Pipeline              string        `json:"pipeline,omitempty"`
	TextConfig            *ModelConfig  `json:"text_config,omitempty"`
	WeightGroups          []WeightGroup `json:"weight_groups"`
	WeightArtifactBytes   int64         `json:"weight_artifact_bytes"`
	Recipe                *Recipe       `json:"recipe,omitempty"`
	RecipeStatus          string        `json:"recipe_status"`
	Warnings              []string      `json:"warnings"`
}

func inspectModel(m Model) (Inspection, error) {
	report := Inspection{GGUF: m.GGUF, Model: m.Hub.ID, Revision: m.Hub.SHA, Task: m.Hub.PipelineTag, Library: m.Hub.Library, Gated: m.Hub.Gated, License: m.Hub.CardData.License, SourceConfig: m.SourceConfig, ConfigurationReadable: m.SourceConfig != "" && len(m.InspectionWarnings) == 0, Architectures: m.Config.Architectures, Pipeline: m.Index.Class, TextConfig: m.Config.TextConfig, RecipeStatus: "unsupported"}
	if m.GGUF != nil {
		report.SelectedWeightBytes = m.GGUF.File.Size
		report.Warnings = append(report.Warnings, "Repository artifact totals include other quantizations and auxiliaries; deployment selects only the explicitly chosen GGUF file, not every listed artifact.")
	}
	report.Warnings = append(report.Warnings, m.InspectionWarnings...)
	groups := map[string]*WeightGroup{}
	for _, file := range m.Hub.Siblings {
		split := strings.LastIndex(file.Name, ".")
		if split < 0 {
			continue
		}
		format := file.Name[split+1:]
		if !contains([]string{"safetensors", "gguf", "bin", "pth", "pt"}, format) {
			continue
		}
		folder := "(root)"
		if at := strings.LastIndex(file.Name, "/"); at >= 0 {
			folder = file.Name[:at]
		}
		key := folder + ":" + format
		group := groups[key]
		if group == nil {
			group = &WeightGroup{Folder: folder, Format: format}
			groups[key] = group
		}
		group.Files++
		size := file.Bytes()
		if size <= 0 {
			group.UnknownSizes++
		} else {
			group.Bytes += size
			report.WeightArtifactBytes += size
		}
	}
	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		report.WeightGroups = append(report.WeightGroups, *groups[key])
	}
	recipe, found, err := findRecipe(m)
	if err != nil {
		return report, err
	}
	if found {
		report.Recipe = &recipe
		report.RecipeStatus = recipe.Status
		if err = validateRecipeShape(recipe, m); err != nil {
			report.Warnings = append(report.Warnings, err.Error())
		}
	}
	report.Warnings = append(report.Warnings, "Artifact totals include all listed weight variants/support files, not a selected download set or VRAM requirement. Metadata parameter counts can omit components.", "Inspection does not install anything, execute repository code, download weights, or establish hardware compatibility. Use plan on the target host before deploy.")
	return report, nil
}
func inspectCommand(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	ggufFile := fs.String("gguf-file", "", "inspect a selected approved GGUF header without tensor payloads")
	revision := fs.String("revision", "main", "Hugging Face revision")
	asJSON := fs.Bool("json", false, "structured inspection")
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		args = append(append([]string{}, args[1:]...), args[0])
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("inspect requires one model URL or repo ID")
	}
	id, err := ParseModel(fs.Arg(0))
	if err != nil {
		return err
	}
	hub := NewHub()
	hub.Token = os.Getenv("HF_TOKEN")
	m, err := hub.InspectPartial(ctx, id, *revision)
	if err != nil {
		return err
	}
	if *ggufFile != "" {
		if err = hub.inspectGGUF(ctx, &m, *ggufFile); err != nil {
			return err
		}
	}
	report, err := inspectModel(m)
	if err != nil {
		return err
	}
	if *asJSON {
		return printJSON(report)
	}
	fmt.Printf("Model:       %s\nRevision:    %s\nRepo task:   %s\nArchitecture:%v\nPipeline:    %s\nRecipe:      %s\nWeight files: %.2f GiB across listed artifacts\n", report.Model, report.Revision, report.Task, report.Architectures, report.Pipeline, report.RecipeStatus, float64(report.WeightArtifactBytes)/(1024*1024*1024))
	if report.GGUF != nil {
		fmt.Printf("Selected GGUF: %s · %.2f GiB; %d tensors, %d encoded tensor bytes\n", report.GGUF.File.Name, float64(report.SelectedWeightBytes)/(1<<30), report.GGUF.Metadata.TensorCount, report.GGUF.Metadata.TensorBytes)
	}
	if report.Recipe != nil {
		r := report.Recipe
		fmt.Printf("Recipe ID:   %s\nBackends:    %s\nAPI tasks:   %s\n", r.ID, strings.Join(r.Backends, ", "), strings.Join(r.Tasks, ", "))
		for _, b := range r.Blockers {
			fmt.Println("Not implemented: " + b)
		}
		for _, n := range r.Notes {
			fmt.Println("Recipe note: " + n)
		}
	}
	for _, g := range report.WeightGroups {
		fmt.Printf("  %s / %s: %d files, %.2f GiB, %d unknown sizes\n", g.Folder, g.Format, g.Files, float64(g.Bytes)/(1024*1024*1024), g.UnknownSizes)
	}
	for _, w := range report.Warnings {
		fmt.Println("Warning: " + w)
	}
	return nil
}
func catalogCommand(args []string) error {
	fs := flag.NewFlagSet("catalog", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "structured recipe registry")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("catalog takes no model argument")
	}
	catalog, err := LoadRecipes()
	if err != nil {
		return err
	}
	if *asJSON {
		return printJSON(catalog)
	}
	out := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(out, "RECIPE\tSTATUS\tTASKS\tBACKENDS")
	for _, r := range catalog.Recipes {
		fmt.Fprintf(out, "%s\t%s\t%s\t%s\n", r.ID, r.Status, strings.Join(r.Tasks, ","), strings.Join(r.Backends, ","))
	}
	return out.Flush()
}
