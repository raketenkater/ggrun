package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

type recommendOptions struct {
	limit           int
	first           bool
	json            bool
	gpus            string
	ramBudgetMB     int
	ramLimitPercent int
	vramHeadroomMB  int
	ramHeadroomMB   int
}

func parseRecommendArgs(args []string, cfg *config.Config) (recommendOptions, error) {
	opts := recommendOptions{limit: 5, ramLimitPercent: cfg.RAMLimitPercent}
	fs := flag.NewFlagSet("recommend", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.IntVar(&opts.limit, "n", 5, "maximum models per category")
	fs.IntVar(&opts.limit, "limit", 5, "maximum models per category")
	fs.BoolVar(&opts.first, "first", false, "print only the first repository")
	fs.BoolVar(&opts.json, "json", false, "print the planning hardware and recommendations as JSON")
	fs.StringVar(&opts.gpus, "gpus", "", "physical GPU indices available for recommendations")
	budget := fs.String("ram-budget", cfg.RamBudget, "RAM ceiling, e.g. 32G")
	fs.IntVar(&opts.ramLimitPercent, "ram-limit-percent", cfg.RAMLimitPercent, "installed RAM percentage available for planning")
	vram := fs.String("vram-headroom", cfg.VRAMHeadroom, "VRAM held back")
	ram := fs.String("ram-headroom", cfg.RAMHeadroom, "RAM held back")
	// Preserve the historical -n5 spelling.
	normalized := make([]string, 0, len(args))
	for _, arg := range args {
		if strings.HasPrefix(arg, "-n") && len(arg) > 2 && arg[2] >= '0' && arg[2] <= '9' {
			arg = "-n=" + arg[2:]
		}
		normalized = append(normalized, arg)
	}
	if err := fs.Parse(normalized); err != nil {
		return opts, err
	}
	if len(fs.Args()) > 0 {
		if len(fs.Args()) != 1 {
			return opts, fmt.Errorf("unexpected arguments: %s", strings.Join(fs.Args(), " "))
		}
		n, err := strconv.Atoi(fs.Args()[0])
		if err != nil {
			return opts, fmt.Errorf("unexpected argument: %s", fs.Args()[0])
		}
		opts.limit = n
	}
	if opts.limit < 1 {
		return opts, fmt.Errorf("recommendation limit must be positive")
	}
	if opts.ramLimitPercent < 0 || opts.ramLimitPercent > 100 {
		return opts, fmt.Errorf("--ram-limit-percent must be between 0 and 100")
	}
	if _, err := parseGPUIndices(opts.gpus); err != nil {
		return opts, fmt.Errorf("--gpus: %w", err)
	}
	for _, v := range []struct {
		name, value string
		dst         *int
	}{
		{"--ram-budget", *budget, &opts.ramBudgetMB},
		{"--vram-headroom", *vram, &opts.vramHeadroomMB},
		{"--ram-headroom", *ram, &opts.ramHeadroomMB},
	} {
		n, err := config.ParseBudgetMBStrict(v.value)
		if err != nil {
			return opts, fmt.Errorf("%s: %w", v.name, err)
		}
		*v.dst = n
	}
	if opts.first && opts.json {
		return opts, fmt.Errorf("--first and --json cannot be combined")
	}
	return opts, nil
}

func recommendationCapabilities(caps *detect.Capabilities, opts recommendOptions) (*detect.Capabilities, error) {
	if caps == nil {
		return nil, fmt.Errorf("hardware detection returned no inventory")
	}
	if err := validateRequestedGPUs(caps, &launchRequest{GPUsFlag: opts.gpus}); err != nil {
		return nil, err
	}
	selected := *caps
	if strings.TrimSpace(opts.gpus) != "" {
		indices, err := parseGPUIndices(opts.gpus)
		if err != nil {
			return nil, err
		}
		wanted := map[int]bool{}
		for _, i := range indices {
			wanted[i] = true
		}
		selected.GPUs = nil
		for _, gpu := range caps.GPUs {
			if wanted[gpu.Index] {
				selected.GPUs = append(selected.GPUs, gpu)
			}
		}
	}
	return recommend.PlanningCapabilities(&selected, opts.ramBudgetMB, opts.ramLimitPercent, opts.vramHeadroomMB, opts.ramHeadroomMB), nil
}

func cmdRecommend(args []string) {
	cfg := loadConfigOrExit()
	opts, err := parseRecommendArgs(args, cfg)
	if errors.Is(err, flag.ErrHelp) {
		fmt.Println("Usage: ggrun recommend [-n N] [--first | --json] [--gpus 0,1] [--ram-budget 32G] [--ram-limit-percent N] [--vram-headroom 1G] [--ram-headroom 4G]")
		fmt.Println("Limits affect recommendations only; pass the same restrictions when launching. Speeds are estimates.")
		return
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	caps, err := detect.Detect()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error detecting hardware: %v\n", err)
		os.Exit(1)
	}
	caps, err = recommendationCapabilities(caps, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}
	recommend.MaybeRefresh()
	cats := recommend.TopCategories(caps, opts.limit)
	if opts.json {
		data, err := json.MarshalIndent(struct {
			Hardware   *detect.Capabilities `json:"planning_hardware"`
			Categories recommend.Categories `json:"categories"`
			Note       string               `json:"note"`
		}{caps, cats, "Advisory capacity and estimated speeds; launch rechecks actual hardware. Repeat restrictions on launch."}, "", "  ")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Println(string(data))
		return
	}
	if opts.first {
		if len(cats.Balanced) > 0 {
			fmt.Println(cats.Balanced[0].Repo)
		}
		return
	}
	gpu := "CPU only"
	if len(caps.GPUs) > 0 {
		names := make([]string, 0, len(caps.GPUs))
		for _, g := range caps.GPUs {
			names = append(names, fmt.Sprintf("GPU %d: %s %.1f GiB", g.Index, g.Name, float64(g.VRAMTotalMB)/1024))
		}
		gpu = strings.Join(names, " + ")
	}
	fmt.Printf("Recommendation budget: %s | RAM %.1f GiB\n", gpu, float64(caps.RAM.TotalMB)/1024)
	if opts.gpus != "" || opts.ramBudgetMB > 0 {
		fmt.Println("Restrictions apply to recommendations; repeat them when launching.")
	}
	if len(cats.Balanced) == 0 {
		fmt.Println("No models in the catalog fit this budget.")
		return
	}
	printRecGroup := func(title string, rows []recommend.Recommendation) {
		if len(rows) == 0 {
			return
		}
		fmt.Printf("\n%s\n", title)
		fmt.Printf("  %-36s %-10s %-8s %6s %5s %8s\n", "Model", "Fit", "Quant", "Size", "Qual", "Est.speed")
		for _, r := range rows {
			name := r.Name
			if len(name) > 36 {
				name = name[:35] + "…"
			}
			tps := "—"
			if r.PredictedTPS > 0 {
				tps = fmt.Sprintf("%.0f t/s", r.PredictedTPS)
			}
			fmt.Printf("  %-36s %-10s %-8s %5.1fG %4.0f%% %8s\n",
				name, recommend.DisplayFit(r.Fit), r.QuantName, r.QuantSizeGB, r.QualityRetained*100, tps)
		}
	}
	printRecGroup("Best overall — balanced quality, speed and fit", cats.Balanced)
	printRecGroup("Smartest — highest intelligence that fits", cats.Smartest)
	printRecGroup("Fastest — quickest while still capable", cats.Fastest)
	fmt.Println("\nSpeed is an estimate for ranking; run --benchmark on the downloaded model for a measured result.")
	fmt.Println("Fit uses installed capacity; every launch rechecks currently free RAM and VRAM.")
	fmt.Printf("\n%s\n", recommend.CatalogAttribution())
}
