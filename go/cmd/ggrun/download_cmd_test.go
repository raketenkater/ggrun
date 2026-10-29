package main

import (
	"io"
	"os"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
)

// "ggrun <repo> --download --dir X --quant Q" used to forward only the repo,
// so a 300 GB download landed on the default disk with an interactive picker.
func TestCompatDownloadKeepsDirQuantAndYes(t *testing.T) {
	yes, args := downloadAssumeYes(downloadCompatArgs([]string{
		"unsloth/M-GGUF", "--download", "--dir", "/mnt/models", "--quant", "UD-Q2_K_XL", "--yes"}))
	repo, dir, quant, err := downloadOptionsFromArgs(args, "/default")
	if err != nil || !yes || repo != "unsloth/M-GGUF" || dir != "/mnt/models" || quant != "UD-Q2_K_XL" {
		t.Fatalf("yes=%v repo=%q dir=%q quant=%q err=%v", yes, repo, dir, quant, err)
	}
	// Launch flags beside --download stay ignored; a --dir before the repo
	// is not taken for the repo.
	_, args = downloadAssumeYes(downloadCompatArgs([]string{
		"--parallel", "2", "--dir=/mnt/models", "org/model-GGUF", "--download", "--ctx", "8192"}))
	if repo, dir, _, err := downloadOptionsFromArgs(args, "/default"); repo != "org/model-GGUF" || dir != "/mnt/models" || err != nil {
		t.Fatalf("launch flags: repo=%q dir=%q err=%v", repo, dir, err)
	}
	// -y anywhere, including before the repo, is not taken for the repo.
	yes, args = downloadAssumeYes([]string{"-y", "org/repo"})
	if repo, _, _, err := downloadOptionsFromArgs(args, "/default"); !yes || repo != "org/repo" || err != nil {
		t.Fatalf("-y before repo: yes=%v repo=%q err=%v", yes, repo, err)
	}
}

// An unknown flag became the repository ("--repo --help" in the script) and a
// second positional was silently dropped.
func TestDownloadRejectsUnknownOptionsAndExtraArguments(t *testing.T) {
	for _, args := range [][]string{
		{"org/repo", "--ctx", "4096"},
		{"--frobnicate", "org/repo"},
		{"org/repo", "other/repo"},
	} {
		if _, _, _, err := downloadOptionsFromArgs(args, "/default"); err == nil {
			t.Errorf("%v: accepted", args)
		}
	}
}

func TestDownloadHelpPrintsUsage(t *testing.T) {
	for _, flag := range []string{"--help", "-h"} {
		r, w, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		saved := os.Stdout
		os.Stdout = w
		cmdDownload([]string{flag})
		os.Stdout = saved
		w.Close()
		out, _ := io.ReadAll(r)
		r.Close()
		if !strings.Contains(string(out), "Usage: ggrun download") || !strings.Contains(string(out), "--yes") {
			t.Fatalf("%s printed %q", flag, out)
		}
	}
}

// The default quant must come from the same RAM the recommender plans with,
// not from what a running benchmark happens to leave free.
func TestDownloadPlansWithConfiguredRAMCeiling(t *testing.T) {
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 2048}}
	cfg := config.Defaults()
	cfg.RAMLimitPercent = 0
	if got := downloadPlanningCaps(caps, cfg).RAM.FreeMB; got != 131072 {
		t.Fatalf("planning RAM = %d, want installed 131072", got)
	}
	cfg.RamBudget = "64G"
	if got := downloadPlanningCaps(caps, cfg).RAM.FreeMB; got != 65536 {
		t.Fatalf("planning RAM = %d, want the 64G budget", got)
	}
	if caps.RAM.FreeMB != 2048 {
		t.Fatal("detected capabilities were modified")
	}
}
