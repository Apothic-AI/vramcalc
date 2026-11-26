package main

import (
	"bytes"
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	"github.com/Apothic-AI/vramcalc/internal/catalog"
)

func runCLI(args ...string) (int, string, string) {
	var stdout, stderr bytes.Buffer
	code := run(args, &stdout, &stderr)
	return code, stdout.String(), stderr.String()
}

func TestEstimateCatalogModelJSON(t *testing.T) {
	code, out, errOut := runCLI("--json", "Qwen/Qwen3-8B")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut)
	}
	var est estimateJSON
	if err := json.Unmarshal([]byte(out), &est); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if est.Model == nil || *est.Model != "Qwen/Qwen3-8B" {
		t.Errorf("model = %v, want Qwen/Qwen3-8B", est.Model)
	}
	if est.Parameters != 8190735360 || est.PublishedPrecision != "BF16" || est.Unit != "GiB" {
		t.Errorf("parameters %v, precision %q, unit %q", est.Parameters, est.PublishedPrecision, est.Unit)
	}
	if len(est.Precisions) != 3 {
		t.Fatalf("got %d precision rows, want 3", len(est.Precisions))
	}
	native := est.Precisions[0]
	if native.Label != "BF16 (native)" || native.WeightsGiB != 15.256433486938477 || native.NeededGiB != 18.307720184326172 {
		t.Errorf("native row = %+v", native)
	}
	for _, g := range est.GPUs {
		switch g.ID {
		case "rtx-3070": // 18.3 GiB over 8 GiB cards
			if g.GPUsNeeded != 3 || g.FitsOnOne {
				t.Errorf("rtx-3070 = %+v, want 3 GPUs", g)
			}
		case "rtx-4090": // 18.3 GiB fits one 24 GiB card
			if g.GPUsNeeded != 1 || !g.FitsOnOne {
				t.Errorf("rtx-4090 = %+v, want 1 GPU", g)
			}
		}
	}
}

func TestEstimateCustomParamsJSON(t *testing.T) {
	code, out, errOut := runCLI("--params", "8.2", "--dtype", "fp8", "--json")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut)
	}
	var est estimateJSON
	if err := json.Unmarshal([]byte(out), &est); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if est.Model != nil || est.PublishedPrecision != "FP8" {
		t.Errorf("model %v, precision %q; want null and FP8", est.Model, est.PublishedPrecision)
	}
	if len(est.Precisions) != 2 || est.Precisions[0].Label != "FP8 (native)" || est.Precisions[1].Key != "int4" {
		t.Fatalf("precisions = %+v", est.Precisions)
	}
	// Space values for 8.2 billion parameters at 1 byte each.
	if est.Precisions[0].WeightsGiB != 7.636845111846923 || est.Precisions[0].NeededGiB != 9.164214134216307 {
		t.Errorf("FP8 row = %+v", est.Precisions[0])
	}
}

func TestEstimateTextOutput(t *testing.T) {
	code, out, errOut := runCLI("Qwen/Qwen3-8B")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut)
	}
	if first := strings.SplitN(out, "\n", 2)[0]; first != "Qwen/Qwen3-8B: 8.2B parameters, published in BF16." {
		t.Errorf("header = %q", first)
	}
	checks := []string{
		`BF16 \(native\)\s+15 GiB\s+18 GiB`,
		`FP8 \(quantized\)\s+7\.6 GiB\s+9\.2 GiB`,
		`INT4 \(quantized\)\s+3\.8 GiB\s+4\.6 GiB`,
		`GPUs for BF16 \(native\), needing 18 GiB:`,
		`RTX 3070\s+8 GiB\s+Needs 3`,
		`RTX 4090\s+24 GiB\s+Fits on 1`,
	}
	for _, pat := range checks {
		if !regexp.MustCompile(pat).MatchString(out) {
			t.Errorf("output lacks %q:\n%s", pat, out)
		}
	}
}

func TestPrecisionFlagSelectsTable(t *testing.T) {
	code, out, errOut := runCLI("--precision", "int4", "--json", "Qwen/Qwen3-8B")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut)
	}
	var est estimateJSON
	if err := json.Unmarshal([]byte(out), &est); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if est.SelectedPrecision != "int4" || est.SelectedLabel != "INT4 (quantized)" {
		t.Errorf("selected %q / %q", est.SelectedPrecision, est.SelectedLabel)
	}
	for _, g := range est.GPUs {
		if g.ID == "rtx-3070" && (g.GPUsNeeded != 1 || !g.FitsOnOne) {
			t.Errorf("INT4 needs 4.6 GiB, so rtx-3070 should fit on one: %+v", g)
		}
	}
}

func TestUsageErrorsExitTwo(t *testing.T) {
	cases := [][]string{
		{},
		{""},
		{"--params", ""},
		{"--params", "0"},
		{"--params", "-3"},
		{"--params", "abc"},
		{"--params", "NaN"},
		{"--params", "Inf"},
		{"--params", "70", "--precision", "bogus"},
		{"--params", "70", "--dtype", "bogus"},
		{"Qwen/Qwen3-8B", "--params", "70"},
		{"Qwen/Qwen3-8B", "--dtype", "FP16"},
		{"Qwen/Qwen3-8B", "Qwen/Qwen3-8B"},
		{"--json", "models", "qwen"},
		{"--no-such-flag"},
	}
	for _, args := range cases {
		if code, _, _ := runCLI(args...); code != 2 {
			t.Errorf("args %q: exit %d, want 2", args, code)
		}
	}
}

func TestLookupErrorsExitOne(t *testing.T) {
	code, _, errOut := runCLI("Qwen3-8B")
	if code != 1 || !strings.Contains(errOut, "Qwen/Qwen3-8B") {
		t.Errorf("partial ID: exit %d, stderr %q; want 1 with a close match", code, errOut)
	}
	code, _, errOut = runCLI("no-such/model")
	if code != 1 || !strings.Contains(errOut, "vramcalc models") {
		t.Errorf("unknown ID: exit %d, stderr %q; want 1 with a hint", code, errOut)
	}
	if code, _, _ := runCLI("models", "no-such-model-xyz"); code != 1 {
		t.Errorf("models with no match: exit %d, want 1", code)
	}
}

func TestModelsSearch(t *testing.T) {
	code, out, errOut := runCLI("models", "qwen3-8b")
	if code != 0 || !strings.Contains(out, "Qwen/Qwen3-8B") {
		t.Errorf("exit %d, stdout %q, stderr %q", code, out, errOut)
	}
}

func TestGPUsJSONOrder(t *testing.T) {
	code, out, errOut := runCLI("gpus", "--json")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut)
	}
	var gpus []catalog.GPU
	if err := json.Unmarshal([]byte(out), &gpus); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(gpus) != 29 || gpus[0].ID != "rtx-3070" || gpus[len(gpus)-1].ID != "amd-mi300x" {
		t.Errorf("got %d GPUs, first %s, last %s", len(gpus), gpus[0].ID, gpus[len(gpus)-1].ID)
	}
}

func TestHelpExitsZero(t *testing.T) {
	for _, args := range [][]string{{"-h"}, {"help"}, {"models", "-h"}} {
		if code, _, _ := runCLI(args...); code != 0 {
			t.Errorf("args %q: exit %d, want 0", args, code)
		}
	}
}

func TestParseBillions(t *testing.T) {
	good := map[string]float64{"70": 70, "8.2": 8.2, "8.2B": 8.2, " 8.2b ": 8.2, "1e3": 1000}
	for in, want := range good {
		if got, err := parseBillions(in); err != nil || got != want {
			t.Errorf("parseBillions(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	for _, in := range []string{"", "0", "-1", "abc", "NaN", "Inf", "B"} {
		if _, err := parseBillions(in); err == nil {
			t.Errorf("parseBillions(%q) accepted bad input", in)
		}
	}
}
