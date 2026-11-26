// Package catalog holds the GPU and model data from the aquanode GPU VRAM
// Calculator Space. The data is embedded in the binary at build time.
package catalog

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

//go:embed catalog.json
var raw []byte

// Model is one Hugging Face model. Params is the total parameter count, and
// Dtype is the published precision key: F32, BF16, F16 or F8_E4M3.
type Model struct {
	ID     string `json:"id"`
	Params int64  `json:"params"`
	Dtype  string `json:"dtype"`
}

// GPU is one accelerator. MemoryGiB is the capacity the Space compares against.
type GPU struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	MemoryGiB int    `json:"memory_gib"`
}

// Catalog is the parsed data set.
type Catalog struct {
	Source string  `json:"source"`
	Commit string  `json:"commit"`
	GPUs   []GPU   `json:"gpus"`
	Models []Model `json:"models"`
}

var load = sync.OnceValues(func() (*Catalog, error) {
	var c Catalog
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("parse embedded catalog: %w", err)
	}
	return &c, nil
})

// Load returns the embedded catalog. It parses the data once.
func Load() (*Catalog, error) {
	return load()
}

// Find returns the model whose ID equals id, ignoring case.
func (c *Catalog) Find(id string) (Model, bool) {
	for _, m := range c.Models {
		if strings.EqualFold(m.ID, id) {
			return m, true
		}
	}
	return Model{}, false
}

// Search returns the models whose ID contains query, ignoring case.
// An empty query matches every model.
func (c *Catalog) Search(query string) []Model {
	q := strings.ToLower(query)
	var out []Model
	for _, m := range c.Models {
		if strings.Contains(strings.ToLower(m.ID), q) {
			out = append(out, m)
		}
	}
	return out
}
