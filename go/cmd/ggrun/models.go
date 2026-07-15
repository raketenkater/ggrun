package main

import (
	"errors"
	"fmt"
	"os"
	"strings"

	modelstore "github.com/raketenkater/ggrun/pkg/models"
)

func cmdModels(args []string) {
	subcommand := "list"
	if len(args) > 0 {
		subcommand = args[0]
		args = args[1:]
	}
	cfg := loadConfigOrExit()

	switch subcommand {
	case "list", "ls":
		if len(args) != 0 {
			modelsUsage()
		}
		listModels(cfg.ModelDir)
	case "path":
		if len(args) != 0 {
			modelsUsage()
		}
		fmt.Println(cfg.ModelDir)
	case "rm", "remove":
		removeModel(cfg.ModelDir, cfg.AssumeYes, args)
	default:
		modelsUsage()
	}
}

func listModels(root string) {
	list, err := modelstore.List(root)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Printf("Model directory does not exist yet: %s\n", root)
			fmt.Println("Download one with: ggrun download <repo/name>")
			return
		}
		fmt.Fprintf(os.Stderr, "Error listing models: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Models in %s\n", root)
	if len(list) == 0 {
		fmt.Println("  No GGUF models found.")
		fmt.Println("  Try: ggrun recommend")
		return
	}
	for _, model := range list {
		files := "file"
		if len(model.Files) != 1 {
			files = "files"
		}
		fmt.Printf("  %-9s %2d %s  %s\n", formatModelBytes(model.Bytes), len(model.Files), files, model.Name)
	}
	fmt.Println("\nRemove one with: ggrun models rm <model.gguf>")
}

func removeModel(root string, assumeYes bool, args []string) {
	var name string
	for _, arg := range args {
		switch arg {
		case "--yes", "-y":
			assumeYes = true
		default:
			if name != "" {
				modelsUsage()
			}
			name = arg
		}
	}
	if name == "" {
		modelsUsage()
	}

	list, err := modelstore.List(root)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "Model directory does not exist: %s\n", root)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Error listing models: %v\n", err)
		os.Exit(1)
	}
	var target *modelstore.Model
	for i := range list {
		if list[i].Name == name {
			target = &list[i]
			break
		}
	}
	if target == nil {
		fmt.Fprintf(os.Stderr, "Model not found: %s\nRun: ggrun models list\n", name)
		os.Exit(1)
	}
	if !assumeYes {
		fmt.Fprintf(os.Stderr, "Remove %s (%s, %d file(s))? [y/N] ", target.Name, formatModelBytes(target.Bytes), len(target.Files))
		var answer string
		if _, err := fmt.Fscan(os.Stdin, &answer); err != nil || !strings.EqualFold(answer, "y") && !strings.EqualFold(answer, "yes") {
			fmt.Println("Cancelled.")
			return
		}
	}
	removed, err := modelstore.Remove(root, name)
	if err != nil {
		if errors.Is(err, modelstore.ErrNotFound) {
			fmt.Fprintf(os.Stderr, "Model not found: %s\n", name)
		} else {
			fmt.Fprintf(os.Stderr, "Error removing model: %v\n", err)
		}
		os.Exit(1)
	}
	fmt.Printf("Removed %s (%s, %d file(s)).\n", removed.Name, formatModelBytes(removed.Bytes), len(removed.Files))
}

func formatModelBytes(bytes int64) string {
	const gib = 1024 * 1024 * 1024
	const mib = 1024 * 1024
	switch {
	case bytes >= gib:
		return fmt.Sprintf("%.1f GiB", float64(bytes)/gib)
	case bytes >= mib:
		return fmt.Sprintf("%.1f MiB", float64(bytes)/mib)
	case bytes >= 1024:
		return fmt.Sprintf("%.1f KiB", float64(bytes)/1024)
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

func modelsUsage() {
	fmt.Fprintln(os.Stderr, "Usage: ggrun models [list|path|rm <model.gguf> [--yes]]")
	os.Exit(2)
}
