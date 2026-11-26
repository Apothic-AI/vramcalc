// Command vramcalc estimates how much GPU memory a model needs for inference
// and which GPUs can hold it. It is a command-line port of the aquanode GPU
// VRAM Calculator Space: https://huggingface.co/spaces/aquanode/gpu-vram-calculator
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/Apothic-AI/vramcalc/internal/calc"
	"github.com/Apothic-AI/vramcalc/internal/catalog"
)

const usageHeader = `vramcalc estimates how much GPU memory a model needs for inference
and which GPUs can hold it.

Usage:
  vramcalc [flags] MODEL        estimate for a model in the catalog
  vramcalc [flags] --params N   estimate for N billion parameters
  vramcalc models [QUERY]       list catalog models whose ID contains QUERY
  vramcalc gpus                 list the GPUs in the catalog

Estimate flags:
`

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		switch args[0] {
		case "models":
			return runModels(args[1:], stdout, stderr)
		case "gpus":
			return runGPUs(args[1:], stdout, stderr)
		case "help":
			args = []string{"-h"}
		}
	}
	return runEstimate(args, stdout, stderr)
}

func runEstimate(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("vramcalc", flag.ContinueOnError)
	fs.SetOutput(stderr)
	paramsFlag := fs.String("params", "", "parameter count in billions, e.g. 70 or 8.2B")
	dtypeFlag := fs.String("dtype", "BF16", "published precision for --params: BF16, FP16, FP32 or FP8")
	precisionFlag := fs.String("precision", "native", "precision for the GPU table: native, fp8 or int4")
	jsonFlag := fs.Bool("json", false, "print JSON instead of tables")
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), usageHeader)
		fs.PrintDefaults()
	}

	positional, err := parseInterleaved(fs, args)
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		return 2
	}

	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	switch {
	case len(positional) > 0 && (positional[0] == "models" || positional[0] == "gpus"):
		return usageError(stderr, "%s is a subcommand; put it first, as in: vramcalc %s --json", positional[0], positional[0])
	case len(positional) > 1:
		return usageError(stderr, "expected one model ID, got %d", len(positional))
	case len(positional) == 1 && strings.TrimSpace(positional[0]) == "":
		return usageError(stderr, "model ID is empty")
	case len(positional) == 1 && set["params"]:
		return usageError(stderr, "give a model ID or --params, not both")
	case len(positional) == 1 && set["dtype"]:
		return usageError(stderr, "--dtype applies only with --params; a model uses its published precision")
	case len(positional) == 0 && !set["params"]:
		return usageError(stderr, "give a model ID or --params")
	}

	cat, err := catalog.Load()
	if err != nil {
		return fail(stderr, err)
	}

	var (
		modelID string
		params  float64
		dtype   calc.Dtype
	)
	if len(positional) == 1 {
		query := strings.TrimSpace(positional[0])
		m, ok := cat.Find(query)
		if !ok {
			return fail(stderr, notFoundError(cat, query))
		}
		d, ok := calc.ParseDtype(m.Dtype)
		if !ok {
			return fail(stderr, fmt.Errorf("catalog entry %s has unknown dtype %q", m.ID, m.Dtype))
		}
		modelID, params, dtype = m.ID, float64(m.Params), d
	} else {
		billions, err := parseBillions(*paramsFlag)
		if err != nil {
			return usageError(stderr, "%v", err)
		}
		d, ok := calc.ParseDtype(*dtypeFlag)
		if !ok {
			return usageError(stderr, "unknown --dtype %q (use BF16, FP16, FP32 or FP8)", *dtypeFlag)
		}
		params, dtype = billions*1e9, d
	}

	rows := calc.Precisions(params, dtype)
	selected, err := calc.Select(rows, *precisionFlag, dtype)
	if err != nil {
		return usageError(stderr, "%v", err)
	}
	fits := calc.Fits(selected.NeededGiB, cat.GPUs)

	if *jsonFlag {
		est := newEstimateJSON(modelID, params, dtype, rows, *precisionFlag, selected, fits)
		return writeJSON(stdout, stderr, est)
	}
	writeEstimateText(stdout, modelID, params, dtype, rows, selected, fits)
	return 0
}

func runModels(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("vramcalc models", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonFlag := fs.Bool("json", false, "print JSON instead of a table")
	positional, err := parseInterleaved(fs, args)
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		return 2
	}
	if len(positional) > 1 {
		return usageError(stderr, "expected at most one QUERY, got %d", len(positional))
	}
	query := ""
	if len(positional) == 1 {
		query = positional[0]
	}

	cat, err := catalog.Load()
	if err != nil {
		return fail(stderr, err)
	}
	models := cat.Search(query)
	if len(models) == 0 {
		return fail(stderr, fmt.Errorf("no models match %q", query))
	}
	if *jsonFlag {
		out := make([]modelJSON, 0, len(models))
		for _, m := range models {
			out = append(out, newModelJSON(m))
		}
		return writeJSON(stdout, stderr, out)
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "MODEL\tPARAMS\tPUBLISHED")
	for _, m := range models {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", m.ID, calc.FormatParams(float64(m.Params)), dtypeLabel(m.Dtype))
	}
	tw.Flush()
	return 0
}

func runGPUs(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("vramcalc gpus", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonFlag := fs.Bool("json", false, "print JSON instead of a table")
	positional, err := parseInterleaved(fs, args)
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		return 2
	}
	if len(positional) > 0 {
		return usageError(stderr, "gpus takes no arguments")
	}

	cat, err := catalog.Load()
	if err != nil {
		return fail(stderr, err)
	}
	gpus := calc.SortGPUs(cat.GPUs)
	if *jsonFlag {
		return writeJSON(stdout, stderr, gpus)
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "GPU\tID\tVRAM")
	for _, g := range gpus {
		fmt.Fprintf(tw, "%s\t%s\t%d GiB\n", g.Name, g.ID, g.MemoryGiB)
	}
	tw.Flush()
	return 0
}

func writeEstimateText(w io.Writer, modelID string, params float64, dtype calc.Dtype, rows []calc.Precision, selected calc.Precision, fits []calc.Fit) {
	if modelID != "" {
		fmt.Fprintf(w, "%s: %s parameters, published in %s.\n\n", modelID, calc.FormatParams(params), dtype.Label)
	} else {
		fmt.Fprintf(w, "%s parameters at %s.\n\n", calc.FormatParams(params), dtype.Label)
	}

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "Precision\tWeights\tNeeded to run")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", r.Label, calc.FormatGiB(r.WeightsGiB), calc.FormatGiB(r.NeededGiB))
	}
	tw.Flush()

	fmt.Fprintf(w, "\nGPUs for %s, needing %s:\n", selected.Label, calc.FormatGiB(selected.NeededGiB))
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "GPU\tVRAM\tFit")
	for _, f := range fits {
		fmt.Fprintf(tw, "%s\t%d GiB\t%s\n", f.GPU.Name, f.GPU.MemoryGiB, fitText(f.GPUsNeeded))
	}
	tw.Flush()

	fmt.Fprintf(w, "\nNeeded to run = weights x %.1f: a rule of thumb for KV cache, activations and allocator overhead at single-request inference around an 8K context. Larger batches or longer contexts need more.\n", calc.Overhead)
}

// fitText words a GPU count the way the Space does.
func fitText(n int) string {
	switch {
	case n == 1:
		return "Fits on 1"
	case n <= 8:
		return fmt.Sprintf("Needs %d", n)
	default:
		return fmt.Sprintf("Needs %d (more than one 8-GPU node)", n)
	}
}

type estimateJSON struct {
	Model              *string         `json:"model"`
	Parameters         float64         `json:"parameters"`
	ParametersLabel    string          `json:"parameters_label"`
	PublishedPrecision string          `json:"published_precision"`
	OverheadFactor     float64         `json:"overhead_factor"`
	Unit               string          `json:"unit"`
	Precisions         []precisionJSON `json:"precisions"`
	SelectedPrecision  string          `json:"selected_precision"`
	SelectedLabel      string          `json:"selected_label"`
	GPUs               []fitJSON       `json:"gpus"`
}

type precisionJSON struct {
	Key           string  `json:"key"`
	Label         string  `json:"label"`
	BytesPerParam float64 `json:"bytes_per_param"`
	WeightsGiB    float64 `json:"weights_gib"`
	NeededGiB     float64 `json:"needed_gib"`
}

type fitJSON struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	MemoryGiB  int    `json:"memory_gib"`
	GPUsNeeded int    `json:"gpus_needed"`
	FitsOnOne  bool   `json:"fits_on_one"`
}

type modelJSON struct {
	ID                 string `json:"id"`
	Parameters         int64  `json:"parameters"`
	ParametersLabel    string `json:"parameters_label"`
	PublishedPrecision string `json:"published_precision"`
}

func newEstimateJSON(modelID string, params float64, dtype calc.Dtype, rows []calc.Precision, requested string, selected calc.Precision, fits []calc.Fit) estimateJSON {
	out := estimateJSON{
		Parameters:         params,
		ParametersLabel:    calc.FormatParams(params),
		PublishedPrecision: dtype.Label,
		Unit:               "GiB",
		OverheadFactor:     calc.Overhead,
		SelectedPrecision:  requested,
		SelectedLabel:      selected.Label,
	}
	if modelID != "" {
		out.Model = &modelID
	}
	for _, r := range rows {
		out.Precisions = append(out.Precisions, precisionJSON{
			Key:           r.Key,
			Label:         r.Label,
			BytesPerParam: r.Bytes,
			WeightsGiB:    r.WeightsGiB,
			NeededGiB:     r.NeededGiB,
		})
	}
	for _, f := range fits {
		out.GPUs = append(out.GPUs, fitJSON{
			ID:         f.GPU.ID,
			Name:       f.GPU.Name,
			MemoryGiB:  f.GPU.MemoryGiB,
			GPUsNeeded: f.GPUsNeeded,
			FitsOnOne:  f.GPUsNeeded == 1,
		})
	}
	return out
}

func newModelJSON(m catalog.Model) modelJSON {
	return modelJSON{
		ID:                 m.ID,
		Parameters:         m.Params,
		ParametersLabel:    calc.FormatParams(float64(m.Params)),
		PublishedPrecision: dtypeLabel(m.Dtype),
	}
}

// parseInterleaved parses flags that may appear before or after positional arguments.
func parseInterleaved(fs *flag.FlagSet, args []string) ([]string, error) {
	var positional []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		rest := fs.Args()
		if len(rest) == 0 {
			return positional, nil
		}
		positional = append(positional, rest[0])
		args = rest[1:]
	}
}

// parseBillions reads a parameter count in billions, such as 70, 8.2 or 8.2B.
func parseBillions(s string) (float64, error) {
	t := strings.TrimSpace(s)
	if n := len(t); n > 0 && (t[n-1] == 'B' || t[n-1] == 'b') {
		t = t[:n-1]
	}
	v, err := strconv.ParseFloat(t, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
		return 0, fmt.Errorf("--params must be a positive number of billions, got %q", s)
	}
	return v, nil
}

func dtypeLabel(key string) string {
	if d, ok := calc.ParseDtype(key); ok {
		return d.Label
	}
	return key
}

// notFoundError names the miss and suggests up to five close matches.
func notFoundError(cat *catalog.Catalog, query string) error {
	matches := cat.Search(query)
	if len(matches) == 0 {
		return fmt.Errorf("model %q is not in the catalog; find IDs with: vramcalc models %s", query, query)
	}
	if len(matches) > 5 {
		matches = matches[:5]
	}
	ids := make([]string, len(matches))
	for i, m := range matches {
		ids[i] = m.ID
	}
	return fmt.Errorf("model %q is not in the catalog; close matches: %s", query, strings.Join(ids, ", "))
}

// usageError reports bad arguments and returns the usage exit code.
func usageError(stderr io.Writer, format string, args ...any) int {
	fmt.Fprintf(stderr, "vramcalc: "+format+"\n", args...)
	fmt.Fprintln(stderr, "Run 'vramcalc -h' for usage.")
	return 2
}

// fail reports a runtime error and returns exit code 1.
func fail(stderr io.Writer, err error) int {
	fmt.Fprintf(stderr, "vramcalc: %v\n", err)
	return 1
}

func writeJSON(stdout, stderr io.Writer, v any) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return fail(stderr, err)
	}
	return 0
}
