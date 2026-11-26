// Package calc ports the memory math of the aquanode GPU VRAM Calculator Space.
//
// Weights are parameters times bytes per parameter. The memory needed to run
// is the weights times Overhead. Every figure is in GiB (1024^3 bytes).
package calc

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/Apothic-AI/vramcalc/internal/catalog"
)

// GiB is 1024^3 bytes.
const GiB = 1 << 30

// Overhead is the Space's rule of thumb for KV cache, activations and
// allocator overhead at single-request inference around an 8K context.
const Overhead = 1.2

// Dtype is a published weight format.
type Dtype struct {
	Key   string  // catalog key: F32, BF16, F16 or F8_E4M3
	Label string  // display name: FP32, BF16, FP16 or FP8
	Bytes float64 // bytes per parameter
}

// Dtypes lists the published formats the Space offers.
var Dtypes = []Dtype{
	{Key: "F32", Label: "FP32", Bytes: 4},
	{Key: "BF16", Label: "BF16", Bytes: 2},
	{Key: "F16", Label: "FP16", Bytes: 2},
	{Key: "F8_E4M3", Label: "FP8", Bytes: 1},
}

// ParseDtype matches a catalog key or display label, ignoring case.
func ParseDtype(s string) (Dtype, bool) {
	name := strings.ToUpper(strings.TrimSpace(s))
	for _, d := range Dtypes {
		if name == d.Key || name == d.Label {
			return d, true
		}
	}
	return Dtype{}, false
}

// Precision is one row of the memory table.
type Precision struct {
	Key        string  // native, fp8 or int4
	Label      string  // for example "BF16 (native)"
	Bytes      float64 // bytes per parameter
	WeightsGiB float64
	NeededGiB  float64
}

// Precisions returns the rows the Space shows for a model with the given
// parameter count and published dtype: native, then FP8 and INT4. A candidate
// is dropped when an earlier row already has the same bytes per parameter, so
// an FP8 model gets no second FP8 row.
func Precisions(params float64, d Dtype) []Precision {
	candidates := []Precision{
		{Key: "native", Label: d.Label + " (native)", Bytes: d.Bytes},
		{Key: "fp8", Label: "FP8 (quantized)", Bytes: 1},
		{Key: "int4", Label: "INT4 (quantized)", Bytes: 0.5},
	}
	seen := map[float64]bool{}
	var rows []Precision
	for _, c := range candidates {
		if seen[c.Bytes] {
			continue
		}
		seen[c.Bytes] = true
		c.WeightsGiB = params * c.Bytes / GiB
		c.NeededGiB = c.WeightsGiB * Overhead
		rows = append(rows, c)
	}
	return rows
}

// Select returns the row for a precision name: native, fp8 or int4. Rows are
// unique by bytes per parameter, so fp8 on an FP8 model returns its native row.
func Select(rows []Precision, name string, d Dtype) (Precision, error) {
	var want float64
	switch name {
	case "native":
		want = d.Bytes
	case "fp8":
		want = 1
	case "int4":
		want = 0.5
	default:
		return Precision{}, fmt.Errorf("unknown --precision %q (use native, fp8 or int4)", name)
	}
	for _, r := range rows {
		if r.Bytes == want {
			return r, nil
		}
	}
	return Precision{}, fmt.Errorf("no %s row for %s", name, d.Label)
}

// SortGPUs returns a copy of gpus ordered by memory, smallest first. Equal
// capacities keep catalog order, as the Space's stable sort does.
func SortGPUs(gpus []catalog.GPU) []catalog.GPU {
	sorted := slices.Clone(gpus)
	slices.SortStableFunc(sorted, func(a, b catalog.GPU) int {
		return a.MemoryGiB - b.MemoryGiB
	})
	return sorted
}

// Fit is one GPU and the number of them a model needs.
type Fit struct {
	GPU        catalog.GPU
	GPUsNeeded int
}

// Fits lists every GPU with the number needed to hold neededGiB, smallest
// first. The count is at least 1, as in the Space.
func Fits(neededGiB float64, gpus []catalog.GPU) []Fit {
	sorted := SortGPUs(gpus)
	fits := make([]Fit, 0, len(sorted))
	for _, g := range sorted {
		n := int(math.Ceil(neededGiB / float64(g.MemoryGiB)))
		fits = append(fits, Fit{GPU: g, GPUsNeeded: max(n, 1)})
	}
	return fits
}
