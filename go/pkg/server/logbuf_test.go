package server

import (
	"fmt"
	"strings"
	"testing"
)

// C-07: a long verbose session must not grow the captured log without bound,
// and the load-phase lines the parsers read must survive.
func TestBackendLogKeepsHeadAndTailWithinBounds(t *testing.T) {
	b := &threadSafeBuffer{headCap: 1024, tailCap: 2048}
	load := "llama_model_load: CUDA0 model buffer size = 1234.00 MiB\n"
	b.Write([]byte(load))
	for i := 0; i < 5000; i++ {
		fmt.Fprintf(b, "slot update_slots: id 0 | task %d | n_past = %d\n", i, i)
	}
	b.Write([]byte("last line\n"))
	if kept := len(b.head) + len(b.tail); kept > 1024+2048+2048/4 {
		t.Fatalf("kept %d bytes, over the bound", kept)
	}
	s := b.String()
	if !strings.HasPrefix(s, load) {
		t.Fatal("load-phase head lost")
	}
	if !strings.HasSuffix(s, "last line\n") {
		t.Fatal("recent tail lost")
	}
	if !strings.Contains(s, "bytes of backend output not kept]") {
		t.Fatal("dropped output not marked")
	}
	if !strings.HasSuffix(b.Tail(10), "last line\n") {
		t.Fatalf("Tail = %q", b.Tail(10))
	}
}

// The watchdog reads with Since; offsets stay absolute and monotonic across
// dropping, and no byte is returned twice.
func TestBackendLogSinceIsMonotonicAcrossDrops(t *testing.T) {
	b := &threadSafeBuffer{headCap: 64, tailCap: 128}
	offset := 0
	var seen strings.Builder
	for i := 0; i < 200; i++ {
		fmt.Fprintf(b, "line %04d\n", i)
		chunk, next := b.Since(offset)
		if next < offset {
			t.Fatalf("offset went back: %d -> %d", offset, next)
		}
		seen.WriteString(chunk)
		offset = next
	}
	if offset != b.written {
		t.Fatalf("final offset %d, written %d", offset, b.written)
	}
	if strings.Count(seen.String(), "line 0199\n") != 1 || strings.Count(seen.String(), "line 0000\n") != 1 {
		t.Fatal("a line was missed or repeated")
	}
	// A stale offset inside the dropped gap gets the marker and the tail.
	chunk, _ := b.Since(200)
	if !strings.HasPrefix(chunk, "\n[ggrun:") || !strings.HasSuffix(chunk, "line 0199\n") {
		t.Fatalf("gap read = %q", chunk)
	}
}

// Small logs are unchanged byte for byte.
func TestBackendLogUnderTheCapIsVerbatim(t *testing.T) {
	b := &threadSafeBuffer{}
	in := "a\nb\nc\n"
	b.Write([]byte(in))
	if b.String() != in {
		t.Fatalf("String = %q", b.String())
	}
	if got, next := b.Since(2); got != "b\nc\n" || next != len(in) {
		t.Fatalf("Since = %q %d", got, next)
	}
}
