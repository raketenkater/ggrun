package main

import (
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/models"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

func TestMatchLocalModelNeedsExactBytesAndTheQuantName(t *testing.T) {
	root := "/models"
	list := []models.Model{
		{Name: "Qwen-UD-Q4_K_XL.gguf", Files: []string{"Qwen-UD-Q4_K_XL.gguf"}, Bytes: 999},
		{Name: filepath.Join("UD-IQ2_XXS", "Big.gguf"), Files: []string{
			filepath.Join("UD-IQ2_XXS", "Big-00001-of-00002.gguf"), filepath.Join("UD-IQ2_XXS", "Big-00002-of-00002.gguf")}, Bytes: 5000},
	}
	if path, ok := matchLocalModel(list, root, recommend.QuantOption{Name: "UD-IQ2_XXS", SizeBytes: 5000}); !ok ||
		path != filepath.Join(root, "UD-IQ2_XXS", "Big-00001-of-00002.gguf") {
		t.Fatalf("split model: %q %v", path, ok)
	}
	for _, q := range []recommend.QuantOption{
		{Name: "UD-Q4_K_XL", SizeBytes: 998}, // another file of the same name
		{Name: "Q8_0", SizeBytes: 999},       // same size, another quant
		{Name: "UD-Q4_K_XL"},                 // no catalog size
	} {
		if path, ok := matchLocalModel(list, root, q); ok {
			t.Fatalf("%+v matched %s", q, path)
		}
	}
}
