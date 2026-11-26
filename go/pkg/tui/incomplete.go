package tui

import (
	"fmt"
	"os"

	"github.com/raketenkater/ggrun/pkg/gguf"
	modelstore "github.com/raketenkater/ggrun/pkg/models"
)

// incompleteDownloadNote returns "incomplete download: X of Y GB" when any file
// of the model is shorter than its GGUF header declares (an interrupted
// download whose header arrived but whose tensor data did not). For a split
// model every shard is checked. It returns "" when the model is complete or
// when no verdict is possible (unreadable header, unknown tensor type).
func incompleteDownloadNote(path string) string {
	files, _, err := modelstore.ResolveGGUFShardFiles(path)
	if err != nil || len(files) == 0 {
		return ""
	}
	short := false
	var haveBytes, wantBytes int64
	for _, file := range files {
		st, err := os.Stat(file)
		if err != nil {
			return ""
		}
		actual := st.Size()
		want := actual
		if expected, ok := gguf.ExpectedSize(file); ok {
			want = expected
			if actual < expected {
				short = true
			}
		}
		haveBytes += actual
		wantBytes += want
	}
	if !short {
		return ""
	}
	const gb = 1024 * 1024 * 1024
	return fmt.Sprintf("incomplete download: %.1f of %.1f GB", float64(haveBytes)/gb, float64(wantBytes)/gb)
}
