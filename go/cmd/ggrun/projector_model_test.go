package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/gguf"
)

func TestProjectorIsNotLaunchedAsAModel(t *testing.T) {
	err := rejectProjectorModel(&gguf.Info{Architecture: "clip"}, "/m/mmproj-gemma-4-12b-it-qat-q4_0.gguf")
	if err == nil || !strings.Contains(err.Error(), "mmproj-gemma-4-12b-it-qat-q4_0.gguf is a multimodal projector") {
		t.Fatalf("projector accepted as a model: %v", err)
	}
	if err := rejectProjectorModel(&gguf.Info{Architecture: "gemma4", BlockCount: 48}, "/m/gemma.gguf"); err != nil {
		t.Fatalf("language model refused: %v", err)
	}
}
