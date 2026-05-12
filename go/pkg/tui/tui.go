package tui

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/raketenkater/llm-server/pkg/detect"
	"github.com/raketenkater/llm-server/pkg/probe"
	"github.com/raketenkater/llm-server/pkg/tune"
)

var (
	titleStyle        = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#7D56F4"))
	subtitleStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("#A0A0A0"))
	selectedStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("#7D56F4")).Bold(true)
	highlightStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#00FF00"))
	warningStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFAA00"))
	errorStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF0000"))
	recommendStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#00AAFF")).Bold(true)
	mutedStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("#666666"))
)

// Screen represents the current TUI screen.
type Screen int

const (
	ScreenMain Screen = iota
	ScreenLaunchPrompt
	ScreenModelConfig
	ScreenPrelaunch
	ScreenTunedPicker
	ScreenSettings
	ScreenDownload
	ScreenBackend
	ScreenFirstRun
)

// Model is the Bubble Tea model.
type Model struct {
	screen      Screen
	width       int
	height      int

	// Data
	caps        *detect.Capabilities
	models      []ModelItem
	backend     string
	modelDir    string
	settingsPath string
	cacheDir    string

	// Main menu list
	mainList    list.Model

	// Quick launch / smart predictions
	selectedModel int
	recommendation *LaunchRecommendation

	// Advanced config
	ctxSize       string
	ctxMode       string
	kvPlacement   string
	parallel      string
	aitune        bool
	aituneRounds  int
	benchmark     bool
	vision        bool
	keepalive     bool

	// Tuned config
	tunedConfigs  []tune.ConfigEntry
	tunedIndex    int  // -1 = auto, 0+ = selected
	tunePath      string

	// Inputs
	input         textinput.Model
	inputMode     string

	// Launch request (set when user chooses to launch)
	launchRequest *LaunchRequest

	// Messages
	message       string
	messageType   string // info, warning, error
}

// ModelItem represents a discovered GGUF model.
type ModelItem struct {
	Name    string
	Path    string
	Tuned   int
	SizeGB  float64
	Arch    string
	MaxCtx  int // trained max context from GGUF
	FitCtx  int // empirically proven fit context from probes
}

// LaunchRecommendation holds smart-predicted settings.
type LaunchRecommendation struct {
	ContextSize    int
	GPULayers      int
	UseGPU         bool
	TensorSplit    bool
	KVPlacement    string
	FlashAttention bool
	ParallelSlots  int
	Benchmark      bool
	Reason         string
	Warning        string
}

func InitialModel() Model {
	m := Model{
		screen:       ScreenMain,
		backend:      "ik_llama",
		modelDir:     os.Getenv("HOME") + "/ai_models",
		settingsPath: os.Getenv("HOME") + "/.config/llm-server/config",
		cacheDir:     os.Getenv("HOME") + "/.cache/llm-server",
		ctxSize:      "fit",
		ctxMode:      "fit",
		kvPlacement:  "auto",
		aituneRounds: 8,
	}

	m.input = textinput.New()
	m.input.Placeholder = ""
	m.input.Focus()

	// Discover models
	m.models = discoverModels(m.modelDir)

	// Populate context estimates and tuned counts
	for i := range m.models {
		m.models[i].MaxCtx = probe.DetectMaxCtx(m.models[i].Path)
		m.models[i].FitCtx = probe.EstimateFitCtx(m.models[i].Path, m.cacheDir)
		backendTag := "llama"
		if m.backend == "ik_llama" {
			backendTag = "ik"
		}
		m.models[i].Tuned = tune.CountTunedConfigs(m.cacheDir, m.models[i].Name, backendTag)
	}

	if len(m.models) == 0 {
		m.screen = ScreenFirstRun
	}

	// Detect hardware
	caps, _ := detect.Detect()
	m.caps = caps

	m.mainList = newMainList(m.models)
	return m
}

func newMainList(models []ModelItem) list.Model {
	items := []list.Item{}
	for i, m := range models {
		desc := fmt.Sprintf("%.1fGB, %s", m.SizeGB, m.Arch)
		if m.Tuned > 0 {
			desc += fmt.Sprintf("  [tuned: %d]", m.Tuned)
		}
		items = append(items, mainItem{
			title:    fmt.Sprintf("%d. %s", i+1, m.Name),
			desc:     desc,
			index:    i,
			isModel:  true,
		})
	}
	// Minimal action items
	items = append(items, mainItem{title: "d. Download model", desc: "Get from Hugging Face", isAction: true, action: "download"})
	items = append(items, mainItem{title: "m. Model directory", desc: "Change search path", isAction: true, action: "modeldir"})
	items = append(items, mainItem{title: "q. Quit", desc: "Exit", isAction: true, action: "quit"})

	l := list.New(items, mainItemDelegate{}, 40, 20)
	l.Title = ""
	l.SetShowStatusBar(false)
	l.SetFilteringEnabled(false)
	l.SetShowHelp(false)
	return l
}

type mainItem struct {
	title    string
	desc     string
	index    int
	isModel  bool
	isAction bool
	action   string
}

func (i mainItem) Title() string       { return i.title }
func (i mainItem) Description() string { return i.desc }
func (i mainItem) FilterValue() string { return i.title }

type mainItemDelegate struct{}

func (d mainItemDelegate) Height() int                             { return 2 }
func (d mainItemDelegate) Spacing() int                            { return 1 }
func (d mainItemDelegate) Update(_ tea.Msg, _ *list.Model) tea.Cmd { return nil }
func (d mainItemDelegate) Render(w io.Writer, m list.Model, index int, listItem list.Item) {
	i, ok := listItem.(mainItem)
	if !ok {
		return
	}
	if index == m.Index() {
		fmt.Fprint(w, selectedStyle.Render("▸ "+i.title)+"\n  "+subtitleStyle.Render(i.desc))
	} else {
		fmt.Fprint(w, "  "+i.title+"\n  "+subtitleStyle.Render(i.desc))
	}
}

func (m Model) Init() tea.Cmd {
	return nil
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.mainList.SetWidth(msg.Width - 4)
		m.mainList.SetHeight(msg.Height - 12)
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			if m.screen == ScreenMain {
				return m, tea.Quit
			}
		case "esc":
			if m.screen != ScreenMain {
				m.screen = ScreenMain
				m.message = ""
				return m, nil
			}
		}
	}

	switch m.screen {
	case ScreenMain:
		return m.updateMain(msg)
	case ScreenLaunchPrompt:
		return m.updateLaunchPrompt(msg)
	case ScreenModelConfig:
		return m.updateModelConfig(msg)
	case ScreenPrelaunch:
		return m.updatePrelaunch(msg)
	case ScreenTunedPicker:
		return m.updateTunedPicker(msg)
	case ScreenFirstRun:
		return m.updateFirstRun(msg)
	case ScreenSettings, ScreenDownload, ScreenBackend:
		return m.updateInputScreen(msg)
	}

	return m, nil
}

func (m Model) updateMain(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "enter":
			if item, ok := m.mainList.SelectedItem().(mainItem); ok {
				if item.isModel {
					m.selectedModel = item.index
					m.recommendation = computeRecommendation(m.caps, m.models[m.selectedModel])
					m.screen = ScreenLaunchPrompt
					return m, nil
				}
				switch item.action {
				case "download":
					m.screen = ScreenDownload
					m.inputMode = "download"
					m.input.SetValue("")
					m.input.Placeholder = "Hugging Face repo (e.g. unsloth/Llama-3.2-1B-Instruct)"
					m.input.Focus()
				case "modeldir":
					m.screen = ScreenBackend
					m.inputMode = "modeldir"
					m.input.SetValue(m.modelDir)
					m.input.Placeholder = "Path to model directory"
					m.input.Focus()
				case "quit":
					return m, tea.Quit
				}
			}
		case "s", "S":
			// Hidden settings shortcut
			m.screen = ScreenSettings
			m.inputMode = "settings"
			m.input.SetValue("")
			m.input.Placeholder = "Press Enter to open in $EDITOR"
			m.input.Focus()
		case "b", "B":
			// Hidden backend shortcut
			m.screen = ScreenBackend
			m.inputMode = "backend"
			m.input.SetValue(m.backend)
			m.input.Placeholder = "llama or ik_llama"
			m.input.Focus()
		}
	}

	var cmd tea.Cmd
	m.mainList, cmd = m.mainList.Update(msg)
	return m, cmd
}

func (m Model) updateLaunchPrompt(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "enter":
			m.screen = ScreenPrelaunch
			return m, nil
		case "d", "D":
			m.message = fmt.Sprintf("Dry run: %s", strings.Join(m.buildArgs(), " "))
			m.messageType = "info"
		case "c", "C":
			m.screen = ScreenModelConfig
			return m, nil
		}
	}
	return m, nil
}

func (m Model) updateModelConfig(msg tea.Msg) (tea.Model, tea.Cmd) {
	if len(m.models) == 0 {
		m.screen = ScreenMain
		return m, nil
	}

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "c", "C":
			m.inputMode = "ctx"
			m.input.SetValue(m.ctxSize)
			m.input.Placeholder = "fit, max, or token count"
			m.input.Focus()
		case "p", "P":
			m.inputMode = "parallel"
			m.input.SetValue(m.parallel)
			m.input.Placeholder = "Parallel slots (default 4)"
			m.input.Focus()
		case "k", "K":
			if m.kvPlacement == "auto" {
				m.kvPlacement = "gpu"
			} else if m.kvPlacement == "gpu" {
				m.kvPlacement = "cpu"
			} else {
				m.kvPlacement = "auto"
			}
		case "a", "A":
			m.aitune = !m.aitune
		case "r", "R":
			if m.aitune {
				m.inputMode = "aitune"
				m.input.SetValue(strconv.Itoa(m.aituneRounds))
				m.input.Placeholder = "AI tune rounds (1-30, default 8)"
				m.input.Focus()
			}
		case "b", "B":
			m.benchmark = !m.benchmark
		case "v", "V":
			m.vision = !m.vision
		case "l", "L":
			m.screen = ScreenPrelaunch
			return m, nil
		case "d", "D":
			m.message = fmt.Sprintf("Dry run: %s", strings.Join(m.buildArgs(), " "))
			m.messageType = "info"
		case "t", "T":
			m.tunedConfigs = tune.ListTunedConfigs(m.cacheDir, m.models[m.selectedModel].Name, m.backendTag(), false)
			m.tunedIndex = -1
			m.screen = ScreenTunedPicker
			return m, nil
		}
	}

	if m.inputMode != "" {
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		if keyMsg, ok := msg.(tea.KeyMsg); ok && keyMsg.String() == "enter" {
			val := m.input.Value()
			switch m.inputMode {
			case "ctx":
				m.ctxSize = val
				m.ctxMode = "manual"
				if val == "fit" || val == "" {
					m.ctxMode = "fit"
					m.ctxSize = "fit"
				} else if val == "max" {
					m.ctxMode = "max"
				}
			case "parallel":
				m.parallel = val
			case "aitune":
				if n, err := strconv.Atoi(val); err == nil && n >= 1 && n <= 30 {
					m.aituneRounds = n
				}
			}
			m.inputMode = ""
		}
		return m, cmd
	}

	return m, nil
}

func (m Model) updateFirstRun(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "d", "D", "enter":
			m.screen = ScreenDownload
			m.inputMode = "download"
			m.input.SetValue("")
			m.input.Placeholder = "Hugging Face repo"
			m.input.Focus()
		case "m", "M":
			m.screen = ScreenBackend
			m.inputMode = "modeldir"
			m.input.SetValue(m.modelDir)
			m.input.Placeholder = "Path to model directory"
			m.input.Focus()
		case "q", "Q":
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m Model) updateInputScreen(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	if keyMsg, ok := msg.(tea.KeyMsg); ok && keyMsg.String() == "enter" {
		val := m.input.Value()
		switch m.inputMode {
		case "settings":
			editor := os.Getenv("EDITOR")
			if editor == "" {
				editor = "nano"
			}
			cmd := exec.Command(editor, m.settingsPath)
			cmd.Stdin = os.Stdin
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Run()
		case "download":
			m.message = fmt.Sprintf("Download %s (not yet implemented)", val)
			m.messageType = "info"
		case "backend":
			if val == "llama" || val == "ik_llama" {
				m.backend = val
			}
		case "modeldir":
			if val != "" {
				m.modelDir = val
				m.models = discoverModels(m.modelDir)
				m.mainList = newMainList(m.models)
			}
		}
		m.inputMode = ""
		m.screen = ScreenMain
	}
	return m, cmd
}

func (m Model) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	switch m.screen {
	case ScreenFirstRun:
		return m.viewFirstRun()
	case ScreenLaunchPrompt:
		return m.viewQuickLaunch()
	case ScreenModelConfig:
		return m.viewModelConfig()
	case ScreenPrelaunch:
		return m.viewPrelaunch()
	case ScreenTunedPicker:
		return m.viewTunedPicker()
	case ScreenSettings, ScreenDownload, ScreenBackend:
		return m.viewInputScreen()
	default:
		return m.viewMain()
	}
}

func (m Model) viewMain() string {
	var b strings.Builder

	b.WriteString(titleStyle.Render("═══ llm-server ═══") + "\n")
	b.WriteString(fmt.Sprintf("  Backend:  %s\n", m.backend))
	b.WriteString(fmt.Sprintf("  Hardware: %s\n", hwSummary(m.caps)))
	b.WriteString(fmt.Sprintf("  Models:   %s (%d)\n", m.modelDir, len(m.models)))
	b.WriteString(fmt.Sprintf("  Settings: %s\n", m.settingsPath))
	b.WriteString("\n")

	if len(m.models) == 0 {
		b.WriteString("  (no GGUF models found)\n")
	}

	b.WriteString(m.mainList.View())

	b.WriteString("\n")
	b.WriteString(mutedStyle.Render("  [s] Settings  [b] Backend"))

	if m.message != "" {
		b.WriteString("\n")
		switch m.messageType {
		case "error":
			b.WriteString(errorStyle.Render(m.message))
		case "warning":
			b.WriteString(warningStyle.Render(m.message))
		default:
			b.WriteString(highlightStyle.Render(m.message))
		}
	}

	return b.String()
}

func (m Model) viewFirstRun() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ llm-server First Run ═══") + "\n")
	b.WriteString(fmt.Sprintf("  No runnable GGUF models found in: %s\n", m.modelDir))
	b.WriteString("\n")
	b.WriteString("  [d] Download a model from Hugging Face\n")
	b.WriteString("  [m] Point at an existing model directory\n")
	b.WriteString("  [q] Quit\n")
	return b.String()
}

func (m Model) viewQuickLaunch() string {
	if len(m.models) == 0 || m.recommendation == nil {
		return "No model selected"
	}
	model := m.models[m.selectedModel]
	rec := m.recommendation
	var b strings.Builder

	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ %s ═══", model.Name)) + "\n")
	b.WriteString(fmt.Sprintf("  Size: %.1fGB  Arch: %s\n", model.SizeGB, model.Arch))
	b.WriteString(fmt.Sprintf("  Hardware: %s\n", hwSummary(m.caps)))
	b.WriteString("\n")

	b.WriteString(recommendStyle.Render("  Recommended settings:\n"))
	b.WriteString(fmt.Sprintf("    Context:  %d tokens\n", rec.ContextSize))
	b.WriteString(fmt.Sprintf("    GPU layers: %d/%s\n", rec.GPULayers, func() string {
		if rec.UseGPU {
			return "all on GPU"
		}
		return "CPU only"
	}()))
	if rec.TensorSplit {
		b.WriteString("    Multi-GPU: tensor split enabled\n")
	}
	b.WriteString(fmt.Sprintf("    KV placement: %s\n", rec.KVPlacement))
	b.WriteString(fmt.Sprintf("    Flash attention: %s\n", boolLabel(rec.FlashAttention)))
	b.WriteString(fmt.Sprintf("    Parallel slots: %d\n", rec.ParallelSlots))
	if rec.Benchmark {
		b.WriteString("    Benchmark: enabled (quick model test)\n")
	}
	b.WriteString("\n")

	if rec.Warning != "" {
		b.WriteString(warningStyle.Render("  ⚠ "+rec.Warning) + "\n\n")
	}

	b.WriteString(highlightStyle.Render("  [Enter] Launch with recommendations"))
	b.WriteString("\n")
	b.WriteString("  [d] Dry run    [c] Advanced config    [Esc] Back\n")

	if m.message != "" {
		b.WriteString("\n")
		b.WriteString(highlightStyle.Render(m.message))
	}

	return b.String()
}

func (m Model) viewModelConfig() string {
	if len(m.models) == 0 {
		return "No models"
	}
	model := m.models[m.selectedModel]
	var b strings.Builder

	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ Advanced: %s ═══", model.Name)) + "\n")

	ctxLabel := m.ctxSize
	if m.ctxMode == "fit" {
		ctxLabel = "fit"
	}
	var ctxHint string
	if model.FitCtx > 0 || model.MaxCtx > 0 {
		ctxHint = " ("
		if model.FitCtx > 0 {
			ctxHint += fmt.Sprintf("fits ~%d", model.FitCtx)
		}
		if model.FitCtx > 0 && model.MaxCtx > 0 {
			ctxHint += ", "
		}
		if model.MaxCtx > 0 {
			ctxHint += fmt.Sprintf("train max %d", model.MaxCtx)
		}
		ctxHint += ")"
	}
	b.WriteString(fmt.Sprintf("  [c] Context size       %s%s\n", ctxLabel, ctxHint))
	b.WriteString(fmt.Sprintf("  [p] Parallel slots     %s\n", func() string {
		if m.parallel == "" {
			return "default (4)"
		}
		return m.parallel
	}()))

	kvLabel := "auto (GPU KV first)"
	if m.kvPlacement == "gpu" {
		kvLabel = "gpu (best long-context decode)"
	} else if m.kvPlacement == "cpu" {
		kvLabel = "cpu (more GPU experts for short chat)"
	}
	b.WriteString(fmt.Sprintf("  [K] KV placement       %s\n", kvLabel))
	tuneLabel := "auto"
	if m.tunePath != "" {
		tuneLabel = filepath.Base(m.tunePath)
	}
	b.WriteString(fmt.Sprintf("  [t] Tuned config       %s\n", tuneLabel))
	b.WriteString(fmt.Sprintf("  [a] AI tune            %s\n", boolLabel(m.aitune)))
	if m.aitune {
		b.WriteString(fmt.Sprintf("  [r] AI tune rounds     %d\n", m.aituneRounds))
	}
	b.WriteString(fmt.Sprintf("  [v] Vision (mmproj)    %s\n", boolLabel(m.vision)))
	b.WriteString(fmt.Sprintf("  [b] Benchmark mode     %s\n", boolLabel(m.benchmark)))
	b.WriteString(fmt.Sprintf("  [k] Keep-alive restart %s\n", boolLabel(m.keepalive)))
	b.WriteString("\n")
	b.WriteString("  [L] Launch    [D] Dry run    [Esc] Back\n")

	if m.inputMode != "" {
		b.WriteString("\n")
		b.WriteString(m.input.View())
	}

	if m.message != "" {
		b.WriteString("\n")
		b.WriteString(highlightStyle.Render(m.message))
	}

	return b.String()
}

func (m Model) updatePrelaunch(msg tea.Msg) (tea.Model, tea.Cmd) {
	if len(m.models) == 0 {
		m.screen = ScreenMain
		return m, nil
	}
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "enter":
			m.launchRequest = m.buildLaunchRequest()
			return m, tea.Quit
		case "esc":
			m.screen = ScreenModelConfig
			return m, nil
		}
	}
	return m, nil
}

func (m Model) viewPrelaunch() string {
	if len(m.models) == 0 {
		return "No model selected"
	}
	model := m.models[m.selectedModel]
	var b strings.Builder
	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ Pre-launch: %s ═══", model.Name)) + "\n\n")

	ctx := m.ctxSize
	if m.ctxMode == "fit" {
		ctx = "fit"
	}
	b.WriteString(fmt.Sprintf("  Context:        %s\n", ctx))
	if model.FitCtx > 0 {
		b.WriteString(fmt.Sprintf("  Fit estimate:   ~%d tokens\n", model.FitCtx))
	}
	if model.MaxCtx > 0 {
		b.WriteString(fmt.Sprintf("  Train max:      %d tokens\n", model.MaxCtx))
	}
	b.WriteString(fmt.Sprintf("  Parallel:       %s\n", func() string {
		if m.parallel == "" {
			return "4 (default)"
		}
		return m.parallel
	}()))
	b.WriteString(fmt.Sprintf("  KV placement:   %s\n", m.kvPlacement))
	b.WriteString(fmt.Sprintf("  AI tune:        %s\n", boolLabel(m.aitune)))
	if m.aitune {
		b.WriteString(fmt.Sprintf("  AI tune rounds: %d\n", m.aituneRounds))
	}
	b.WriteString(fmt.Sprintf("  Vision:         %s\n", boolLabel(m.vision)))
	b.WriteString(fmt.Sprintf("  Benchmark:      %s\n", boolLabel(m.benchmark)))
	b.WriteString(fmt.Sprintf("  Keep-alive:     %s\n", boolLabel(m.keepalive)))
	if m.tunePath != "" {
		b.WriteString(fmt.Sprintf("  Tuned config:   %s\n", filepath.Base(m.tunePath)))
	}
	b.WriteString("\n")
	b.WriteString(highlightStyle.Render("  [Enter] Confirm and launch"))
	b.WriteString("\n")
	b.WriteString("  [Esc] Back to config\n")
	return b.String()
}

func (m Model) updateTunedPicker(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			m.screen = ScreenModelConfig
			return m, nil
		case "enter":
			if m.tunedIndex >= 0 && m.tunedIndex < len(m.tunedConfigs) {
				m.tunePath = m.tunedConfigs[m.tunedIndex].Path
			} else {
				m.tunePath = ""
			}
			m.screen = ScreenModelConfig
			return m, nil
		case "up", "k":
			m.tunedIndex--
			if m.tunedIndex < -1 {
				m.tunedIndex = len(m.tunedConfigs) - 1
			}
		case "down", "j":
			m.tunedIndex++
			if m.tunedIndex >= len(m.tunedConfigs) {
				m.tunedIndex = -1
			}
		}
	}
	return m, nil
}

func (m Model) viewTunedPicker() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ Tuned Configs ═══") + "\n\n")
	if len(m.tunedConfigs) == 0 {
		b.WriteString("  No tuned configs match this model/backend.\n")
		b.WriteString("  Run AI tune to create one.\n")
	} else {
		if m.tunedIndex == -1 {
			b.WriteString(selectedStyle.Render("▸ [0] Auto / heuristic cache selection") + "\n")
		} else {
			b.WriteString("  [0] Auto / heuristic cache selection\n")
		}
		for i, entry := range m.tunedConfigs {
			if i == m.tunedIndex {
				b.WriteString(selectedStyle.Render(fmt.Sprintf("▸ [%d] %s", i+1, entry.Label)) + "\n")
				b.WriteString(subtitleStyle.Render(fmt.Sprintf("     %s", filepath.Base(entry.Path))) + "\n")
			} else {
				b.WriteString(fmt.Sprintf("  [%d] %s\n", i+1, entry.Label))
				b.WriteString(subtitleStyle.Render(fmt.Sprintf("     %s", filepath.Base(entry.Path))) + "\n")
			}
		}
	}
	b.WriteString("\n")
	b.WriteString("  [Enter] Select  [Esc] Cancel  [↑/↓] Navigate\n")
	return b.String()
}

func (m Model) viewInputScreen() string {
	var b strings.Builder
	var title string
	switch m.inputMode {
	case "settings":
		title = "Settings"
	case "download":
		title = "Download Model"
	case "backend":
		title = "Backend Selection"
	case "modeldir":
		title = "Model Directory"
	default:
		title = "Input"
	}
	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ %s ═══", title)) + "\n\n")
	b.WriteString(m.input.View())
	b.WriteString("\n\n  Press Enter to confirm, Esc to cancel")
	return b.String()
}

func hwSummary(caps *detect.Capabilities) string {
	if caps == nil {
		return "detecting..."
	}
	ramGB := caps.RAM.TotalMB / 1024
	if len(caps.GPUs) == 0 {
		return fmt.Sprintf("%dGB RAM, %d cores (no GPU)", ramGB, caps.CPU.Cores)
	}
	var vramTotal int
	for _, g := range caps.GPUs {
		vramTotal += g.VRAMTotalMB
	}
	vramGB := float64(vramTotal) / 1024
	return fmt.Sprintf("%d GPU(s) %.1fGB VRAM, %dGB RAM, %d cores",
		len(caps.GPUs), vramGB, ramGB, caps.CPU.Cores)
}

func boolLabel(v bool) string {
	if v {
		return "on"
	}
	return "off"
}

// computeRecommendation generates smart launch settings based on model + hardware.
func computeRecommendation(caps *detect.Capabilities, model ModelItem) *LaunchRecommendation {
	rec := &LaunchRecommendation{
		ContextSize:    4096,
		GPULayers:      0,
		UseGPU:         false,
		KVPlacement:    "auto",
		FlashAttention: true,
		ParallelSlots:  1,
		Benchmark:      false,
	}

	if caps == nil {
		rec.Reason = "No hardware detected, using safe defaults"
		rec.ContextSize = 4096
		return rec
	}

	totalVRAM := caps.TotalVRAM()
	modelSizeMB := int(model.SizeGB * 1024)

	// Determine GPU availability
	numGPUs := len(caps.GPUs)
	hasGPU := numGPUs > 0

	if !hasGPU {
		rec.UseGPU = false
		rec.GPULayers = 0
		rec.ContextSize = min(4096, int(float64(caps.RAM.TotalMB)*0.25/2)) // rough: 0.5MB per 1K ctx
		if rec.ContextSize < 2048 {
			rec.ContextSize = 2048
		}
		rec.Reason = "No GPU detected — CPU-only mode"
		return rec
	}

	// Calculate if model fits on single GPU
	largestGPUVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMTotalMB > largestGPUVRAM {
			largestGPUVRAM = g.VRAMTotalMB
		}
	}

	// Heuristic overhead: KV cache + CUDA context (~1-2GB)
	overheadMB := 2048
	if model.Arch == "MoE" {
		overheadMB = 4096 // MoE needs more overhead
	}

	fitsSingle := modelSizeMB+overheadMB <= largestGPUVRAM
	fitsAll := modelSizeMB+overheadMB <= totalVRAM

	if fitsSingle {
		// Model fits on one GPU — optimal case
		rec.UseGPU = true
		rec.GPULayers = -1 // all layers
		rec.ContextSize = min(32768, modelNumLayers(model)*1024/4) // rough heuristic
		if rec.ContextSize < 4096 {
			rec.ContextSize = 4096
		}
		rec.KVPlacement = "gpu"
		rec.ParallelSlots = 4
		rec.Benchmark = model.SizeGB < 10 // small models get benchmark by default
		if model.Arch == "MoE" {
			rec.Reason = fmt.Sprintf("Fits on %s — full GPU offload with active-experts scheduling", caps.GPUs[0].Name)
		} else {
			rec.Reason = fmt.Sprintf("Fits on %s — full GPU offload, max performance", caps.GPUs[0].Name)
		}
	} else if fitsAll && numGPUs > 1 {
		// Fits across multiple GPUs
		rec.UseGPU = true
		rec.GPULayers = -1
		rec.TensorSplit = true
		rec.ContextSize = 32768
		rec.KVPlacement = "gpu"
		rec.ParallelSlots = 4
		rec.Reason = fmt.Sprintf("Spans %d GPUs with tensor split — all layers on GPU", numGPUs)
	} else if model.Arch == "MoE" || (numGPUs > 0 && modelSizeMB > totalVRAM) {
		// Large model that doesn't fit in VRAM — GPU attention, CPU spill
		rec.UseGPU = true
		rec.GPULayers = -1
		if model.Arch == "MoE" {
			rec.KVPlacement = "cpu"
			rec.ContextSize = 16384
			rec.ParallelSlots = 2
			rec.Warning = "MoE model requires CPU expert offloading — expect lower short-chat tok/s"
			rec.Reason = "MoE model — GPU attention layers, CPU routing experts"
		} else {
			rec.KVPlacement = "auto"
			rec.ContextSize = 8192
			rec.ParallelSlots = 2
			rec.Warning = "Model exceeds total VRAM — partial GPU offload recommended"
			rec.Reason = "Partial GPU offload — attention on GPU, rest on CPU"
		}
	} else {
		// Fallback: small model that might fit partially
		vramAvailable := totalVRAM - overheadMB
		if vramAvailable > modelSizeMB/2 {
			rec.UseGPU = true
			rec.GPULayers = -1
			rec.ContextSize = 8192
			rec.KVPlacement = "auto"
			rec.Warning = "Model exceeds total VRAM — partial GPU offload recommended"
			rec.Reason = "Partial GPU offload — attention on GPU, rest on CPU"
		} else {
			rec.UseGPU = false
			rec.GPULayers = 0
			rec.ContextSize = 4096
			rec.KVPlacement = "cpu"
			rec.Warning = "Model too large for available VRAM — CPU-only mode"
			rec.Reason = "CPU-only — safe but slower"
		}
	}

	return rec
}

// min returns the smaller of a and b.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// modelNumLayers and hidden size heuristics from filename/size.
func modelNumLayers(model ModelItem) int {
	name := strings.ToLower(model.Name)
	switch {
	case strings.Contains(name, "0.6b"):
		return 28
	case strings.Contains(name, "1b") || strings.Contains(name, "1.5b"):
		return 24
	case strings.Contains(name, "3b") || strings.Contains(name, "4b"):
		return 36
	case strings.Contains(name, "7b") || strings.Contains(name, "8b"):
		return 32
	case strings.Contains(name, "14b") || strings.Contains(name, "15b"):
		return 48
	case strings.Contains(name, "27b") || strings.Contains(name, "30b"):
		return 64
	case strings.Contains(name, "32b"):
		return 64
	case strings.Contains(name, "70b") || strings.Contains(name, "72b"):
		return 80
	case strings.Contains(name, "122b"):
		return 80
	default:
		// Estimate from size: ~100MB per layer for Q4
		return int(model.SizeGB * 1024 / 100)
	}
}

func discoverModels(dir string) []ModelItem {
	var items []ModelItem
	seen := make(map[string]bool)

	_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		name := info.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
			return nil
		}
		// Filter out mmproj and draft files
		lower := strings.ToLower(name)
		if strings.Contains(lower, "mmproj") || strings.Contains(lower, "dflash-draft") {
			return nil
		}

		// Handle multi-part models: only list -00001-of-NNNNN.gguf
		isMultiPart := false
		baseName := name
		if re := strings.Index(name, "-00001-of-"); re > 0 {
			baseName = name[:re] + ".gguf"
			isMultiPart = true
		} else if strings.Contains(name, "-of-") {
			// Skip non-first parts of multi-part models
			return nil
		}

		if seen[baseName] {
			return nil
		}
		seen[baseName] = true

		// Sum sizes for multi-part models
		dirPath := filepath.Dir(path)
		var totalBytes int64
		if isMultiPart {
			pattern := baseName[:len(baseName)-5] + "*" // remove .gguf
			matches, _ := filepath.Glob(filepath.Join(dirPath, pattern+"*.gguf"))
			for _, match := range matches {
				st, err := os.Stat(match)
				if err == nil {
					totalBytes += st.Size()
				}
			}
		} else {
			totalBytes = info.Size()
		}

		sizeGB := float64(totalBytes) / (1024 * 1024 * 1024)
		arch := "dense"
		if match := strings.Contains(name, "A") && strings.Contains(name, "B"); match {
			// Check A[0-9]+B pattern for MoE detection
			for i := 0; i < len(name)-1; i++ {
				if name[i] == 'A' || name[i] == 'a' {
					j := i + 1
					for j < len(name) && name[j] >= '0' && name[j] <= '9' {
						j++
					}
					if j < len(name) && (name[j] == 'B' || name[j] == 'b') {
						arch = "MoE"
						break
					}
				}
			}
		}

		items = append(items, ModelItem{
			Name:   baseName,
			Path:   path,
			SizeGB: sizeGB,
			Arch:   arch,
		})
		return nil
	})

	return items
}

func (m Model) backendTag() string {
	if m.backend == "ik_llama" {
		return "ik"
	}
	return "llama"
}

func (m Model) buildLaunchRequest() *LaunchRequest {
	if len(m.models) == 0 || m.selectedModel < 0 || m.selectedModel >= len(m.models) {
		return nil
	}
	model := m.models[m.selectedModel]
	ctx := 32768
	if m.recommendation != nil && m.recommendation.ContextSize > 0 {
		ctx = m.recommendation.ContextSize
	}
	if m.ctxMode == "max" {
		ctx = 131072
	} else if m.ctxMode == "manual" && m.ctxSize != "" {
		if n, err := strconv.Atoi(m.ctxSize); err == nil {
			ctx = n
		}
	}
	gpuLayers := 999
	if m.recommendation != nil {
		gpuLayers = m.recommendation.GPULayers
	}
	parallel := 4
	if m.parallel != "" {
		if n, err := strconv.Atoi(m.parallel); err == nil {
			parallel = n
		}
	}
	return &LaunchRequest{
		ModelPath:    model.Path,
		Port:         8081,
		CtxSize:      ctx,
		KVPlacement:  m.kvPlacement,
		KVQuality:    "mid",
		GPULayers:    gpuLayers,
		FlashAttn:    true,
		Parallel:     parallel,
		Vision:       m.vision,
		Backend:      m.backend,
		TuneCache:    m.tunePath,
		AITune:       m.aitune,
		AITuneRounds: m.aituneRounds,
		Benchmark:    m.benchmark,
		KeepAlive:    m.keepalive,
	}
}

func (m Model) buildArgs() []string {
	req := m.buildLaunchRequest()
	if req == nil {
		return nil
	}
	args := []string{
		"-m", req.ModelPath,
		"--port", strconv.Itoa(req.Port),
		"-c", strconv.Itoa(req.CtxSize),
		"-ngl", strconv.Itoa(req.GPULayers),
	}
	if req.KVPlacement != "" && req.KVPlacement != "auto" {
		args = append(args, "--kv-placement", req.KVPlacement)
	}
	if req.KVQuality == "high" {
		args = append(args, "--cache-type-k", "f16", "--cache-type-v", "f16")
	} else if req.KVQuality == "mid" {
		args = append(args, "--cache-type-k", "q8_0", "--cache-type-v", "q8_0")
	}
	if req.FlashAttn {
		args = append(args, "--flash-attn", "on")
	}
	if req.Parallel > 1 {
		args = append(args, "-np", strconv.Itoa(req.Parallel))
	}
	return args
}

// LaunchRequest is returned when the user chooses to launch a model.
type LaunchRequest struct {
	ModelPath   string
	Port        int
	CtxSize     int
	KVPlacement string
	KVQuality   string
	GPULayers   int
	FlashAttn   bool
	Parallel    int
	Vision      bool
	Backend     string
	TuneCache   string
	AITune      bool
	AITuneRounds int
	Benchmark   bool
	KeepAlive   bool
}

// Run starts the TUI and returns a launch request if the user chose to launch.
func Run() (*LaunchRequest, error) {
	p := tea.NewProgram(InitialModel(), tea.WithAltScreen())
	m, err := p.Run()
	if err != nil {
		return nil, err
	}
	if model, ok := m.(Model); ok && model.launchRequest != nil {
		return model.launchRequest, nil
	}
	return nil, nil
}
