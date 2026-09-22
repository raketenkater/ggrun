package update

import (
	"os"
	"testing"
)

// A scripted run redirects stdin from /dev/null; that must not enable the
// interactive startup update prompt.
func TestDevNullIsNotATerminal(t *testing.T) {
	f, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if isTerminal(f) {
		t.Fatal("/dev/null was treated as an interactive terminal")
	}
}
