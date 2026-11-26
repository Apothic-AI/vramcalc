package catalog

import (
	"strings"
	"testing"
)

func TestSnapshotSize(t *testing.T) {
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Models) != 1333 || len(c.GPUs) != 29 {
		t.Fatalf("got %d models and %d GPUs, want 1333 and 29", len(c.Models), len(c.GPUs))
	}
	seen := map[string]bool{}
	for _, m := range c.Models {
		key := strings.ToLower(m.ID)
		if seen[key] {
			t.Errorf("duplicate model ID %q", m.ID)
		}
		seen[key] = true
	}
}

func TestFindIsCaseInsensitiveAndExact(t *testing.T) {
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	m, ok := c.Find("qwen/qwen3-8b")
	if !ok || m.ID != "Qwen/Qwen3-8B" || m.Params != 8190735360 || m.Dtype != "BF16" {
		t.Fatalf("Find = %+v, %v", m, ok)
	}
	if _, ok := c.Find("Qwen3-8B"); ok {
		t.Error("Find matched a partial ID")
	}
}

func TestSearchMatchesSubstrings(t *testing.T) {
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, m := range c.Search("QWEN3-8B") {
		if m.ID == "Qwen/Qwen3-8B" {
			found = true
		}
	}
	if !found {
		t.Error("Search did not find Qwen/Qwen3-8B")
	}
	if got := len(c.Search("")); got != 1333 {
		t.Errorf("empty query matched %d models, want 1333", got)
	}
}
