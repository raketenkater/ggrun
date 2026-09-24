package main

import (
	"fmt"
	"reflect"
	"testing"
)

// The optimized path is the default one: `ggrun <model>` and a fresh TUI launch
// of the same model must resolve to the same request. The TUI types a few
// flags whose values equal the CLI defaults; they may differ only in the
// recorded argv (used for resume); every resolved field, including the backend
// passthrough args the optimizer may rewrite, must be equal.
func TestFreshTUIDefaultLaunchMatchesPlainCLI(t *testing.T) {
	tuiTail := []string{"--port", "8081", "--ctx-size", "fit", "--kv-placement", "auto", "--support-expert", "auto"}
	plain, err := parseLaunchArgs([]string{"m.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	fromTUI, err := parseLaunchArgs(append([]string{"m.gguf"}, tuiTail...))
	if err != nil {
		t.Fatal(err)
	}
	a, b := reflect.ValueOf(*plain), reflect.ValueOf(*fromTUI)
	for i := 0; i < a.NumField(); i++ {
		name := a.Type().Field(i).Name
		// OriginalArgs records the argv (used for resume); unexported fields are
		// runtime state filled in after parsing, not part of the request.
		if name == "OriginalArgs" || !a.Type().Field(i).IsExported() {
			continue
		}
		if x, y := fmt.Sprintf("%#v", a.Field(i).Interface()), fmt.Sprintf("%#v", b.Field(i).Interface()); x != y {
			t.Errorf("%s differs: plain CLI %s, fresh TUI %s", name, x, y)
		}
	}
}
