package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/raketenkater/ggrun/pkg/backendverify"
	"github.com/raketenkater/ggrun/pkg/detect"
)

// cmdBackendVerify is `ggrun backend verify`: load a small model with a fixed
// CPU argv and check that the planner's log parsers still read this backend
// (review I-05). Exits non-zero when they do not.
func cmdBackendVerify(args []string) {
	fs := flag.NewFlagSet("backend verify", flag.ExitOnError)
	model := fs.String("model", os.Getenv("GGRUN_BACKEND_VERIFY_MODEL"), "small GGUF to load (default $GGRUN_BACKEND_VERIFY_MODEL)")
	serverBin := fs.String("server-bin", "", "llama-server to check (default: the backend ggrun would use)")
	timeout := fs.Duration("timeout", 3*time.Minute, "load deadline")
	_ = fs.Parse(args)
	if *model == "" {
		fmt.Fprintln(os.Stderr, "usage: ggrun backend verify --model SMALL.gguf [--server-bin PATH]")
		os.Exit(2)
	}
	binary := *serverBin
	if binary == "" {
		caps, _ := detect.Detect()
		be := findBackend(caps)
		if be == nil || be.Path == "" {
			fmt.Fprintln(os.Stderr, "no backend found; pass --server-bin")
			os.Exit(1)
		}
		binary = be.Path
	}
	fmt.Printf("[verify] %s\n[verify] version: %s\n", binary, firstLine(backendVersionOutput(binary)))
	r, err := backendverify.Run(binary, *model, *timeout)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[verify] FAILED: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("[verify] ok: model buffer %d MiB, host buffers %d MiB, KV %.1f MiB, %d/%d layers on GPU (expected 0)\n",
		r.HostModelMB, r.HostBuffersMB, r.KVTotalMB, r.OffloadedLayers, r.TotalLayers)
}

func firstLine(s string) string {
	for i, c := range s {
		if c == '\n' {
			return s[:i]
		}
	}
	return s
}

// printBackendBuild names the backend build a plan targets, so a plan can be
// matched to the build whose log the parsers were checked against.
func printBackendBuild(label string, be *backendInfo) {
	if be == nil || be.Path == "" {
		return
	}
	version := firstLine(be.Version)
	if version == "" {
		version = "version unknown"
	}
	fmt.Printf("[%s] backend %s (%s)\n", label, be.Path, version)
}
