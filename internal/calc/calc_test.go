package calc

import (
	"testing"

	"github.com/Apothic-AI/vramcalc/internal/catalog"
)

type wantRow struct {
	key     string
	label   string
	weights float64
	needed  float64
}

func mustDtype(t *testing.T, key string) Dtype {
	t.Helper()
	d, ok := ParseDtype(key)
	if !ok {
		t.Fatalf("unknown dtype %q", key)
	}
	return d
}

func checkRows(t *testing.T, rows []Precision, want []wantRow) {
	t.Helper()
	if len(rows) < len(want) {
		t.Fatalf("got %d rows, want at least %d", len(rows), len(want))
	}
	for i, w := range want {
		r := rows[i]
		if r.Key != w.key || r.Label != w.label || r.WeightsGiB != w.weights || r.NeededGiB != w.needed {
			t.Errorf("row %d = %+v, want %+v", i, r, w)
		}
	}
}

// Values are the Space's own output for Qwen/Qwen3-8B (8190735360 parameters,
// BF16), computed under node with the Space's rows() function.
func TestPrecisionsMatchSpace(t *testing.T) {
	params := 8190735360.0
	checkRows(t, Precisions(params, mustDtype(t, "BF16")), []wantRow{
		{"native", "BF16 (native)", 15.256433486938477, 18.307720184326172},
		{"fp8", "FP8 (quantized)", 7.628216743469238, 9.153860092163086},
		{"int4", "INT4 (quantized)", 3.814108371734619, 4.576930046081543},
	})
}

// A custom 8.2 billion parameters. The product is taken at run time, as the
// Space does; a constant expression would be exact and differ in the last bits.
func TestPrecisionsCustomParamsMatchSpace(t *testing.T) {
	billions := 8.2
	params := billions * 1e9
	checkRows(t, Precisions(params, mustDtype(t, "BF16")), []wantRow{
		{"native", "BF16 (native)", 15.273690223693846, 18.328428268432614},
		{"fp8", "FP8 (quantized)", 7.636845111846923, 9.164214134216307},
	})
}

func TestNativeFP8HasNoSecondFP8Row(t *testing.T) {
	// deepseek-ai/DeepSeek-V3 is published in F8_E4M3, so its FP8 row is the native one.
	rows := Precisions(684531386000, mustDtype(t, "F8_E4M3"))
	if len(rows) != 2 || rows[0].Label != "FP8 (native)" || rows[1].Key != "int4" {
		t.Fatalf("rows = %+v, want FP8 (native) and INT4 only", rows)
	}
}

func TestFP32HasThreeRows(t *testing.T) {
	rows := Precisions(1e9, mustDtype(t, "F32"))
	labels := []string{"FP32 (native)", "FP8 (quantized)", "INT4 (quantized)"}
	if len(rows) != len(labels) {
		t.Fatalf("got %d rows, want %d", len(rows), len(labels))
	}
	for i, l := range labels {
		if rows[i].Label != l {
			t.Errorf("row %d = %q, want %q", i, rows[i].Label, l)
		}
	}
}

func TestSelect(t *testing.T) {
	bf16 := mustDtype(t, "BF16")
	rows := Precisions(1e9, bf16)
	if r, err := Select(rows, "fp8", bf16); err != nil || r.Label != "FP8 (quantized)" {
		t.Errorf("Select fp8 = %+v, %v", r, err)
	}
	if r, err := Select(rows, "native", bf16); err != nil || r.Label != "BF16 (native)" {
		t.Errorf("Select native = %+v, %v", r, err)
	}
	fp8 := mustDtype(t, "F8_E4M3")
	rows = Precisions(1e9, fp8)
	if r, err := Select(rows, "fp8", fp8); err != nil || r.Label != "FP8 (native)" {
		t.Errorf("Select fp8 on an FP8 model = %+v, %v; want its native row", r, err)
	}
	if _, err := Select(rows, "bogus", fp8); err == nil {
		t.Error("Select accepted an unknown precision")
	}
}

func TestFitsOrderAndCounts(t *testing.T) {
	cat, err := catalog.Load()
	if err != nil {
		t.Fatal(err)
	}
	fits := Fits(20, cat.GPUs)
	if len(fits) != len(cat.GPUs) {
		t.Fatalf("got %d fits, want %d", len(fits), len(cat.GPUs))
	}
	// Smallest first. The three 12 GiB cards keep their catalog order.
	wantOrder := []string{"rtx-3070", "rtx-3080", "rtx-5070", "rtx-4070-ti", "rtx-4070"}
	for i, id := range wantOrder {
		if fits[i].GPU.ID != id {
			t.Errorf("fits[%d] = %s, want %s", i, fits[i].GPU.ID, id)
		}
	}
	if last := fits[len(fits)-1].GPU.ID; last != "amd-mi300x" {
		t.Errorf("last fit = %s, want amd-mi300x", last)
	}
	// 20 GiB: 8 GiB cards need 3, 10 and 12 GiB cards need 2, 24 GiB cards need 1.
	counts := map[string]int{"rtx-3070": 3, "rtx-3080": 2, "rtx-4070": 2, "rtx-4090": 1}
	for _, f := range fits {
		if want, ok := counts[f.GPU.ID]; ok && f.GPUsNeeded != want {
			t.Errorf("%s needs %d, want %d", f.GPU.ID, f.GPUsNeeded, want)
		}
	}
	// Exactly one card's capacity still needs one card.
	for _, f := range Fits(24, cat.GPUs) {
		if f.GPU.ID == "rtx-4090" && f.GPUsNeeded != 1 {
			t.Errorf("rtx-4090 for 24 GiB needs %d, want 1", f.GPUsNeeded)
		}
	}
}

func TestFitsNeedsAtLeastOne(t *testing.T) {
	cat, err := catalog.Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range Fits(0.5, cat.GPUs) {
		if f.GPUsNeeded != 1 {
			t.Errorf("%s needs %d for 0.5 GiB, want 1", f.GPU.ID, f.GPUsNeeded)
		}
	}
}

func TestCatalogDtypesAreKnown(t *testing.T) {
	cat, err := catalog.Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range cat.Models {
		if _, ok := ParseDtype(m.Dtype); !ok {
			t.Fatalf("%s has unknown dtype %q", m.ID, m.Dtype)
		}
	}
}
