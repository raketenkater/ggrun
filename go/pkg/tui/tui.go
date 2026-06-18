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
	"github.com/raketenkater/llm-server/pkg/config"
	"github.com/raketenkater/llm-server/pkg/detect"
	"github.com/raketenkater/llm-server/pkg/probe"
	"github.com/raketenkater/llm-server/pkg/recommend"
	"github.com/raketenkater/llm-server/pkg/tune"
)

var (
	titleStyle     = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#7D56F4"))
	subtitleStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("#A0A0A0"))
	selectedStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("#7D56F4")).Bold(true)
	highlightStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00FF00"))
	warningStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFAA00"))
	errorStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF0000"))
	recommendStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00AAFF")).Bold(true)
	mutedStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("#666666"))
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
	ScreenRecommended
	ScreenChoice
)

// Model is the Bubble Tea model.
type Model struct {
	screen Screen
	width  int
	height int

	// Data
	caps                   *detect.Capabilities
	models                 []ModelItem
	backend                string
	modelDir               string
	settingsPath           string
	cacheDir               string
	recommendationGroups   recommend.Categories
	recommendations        []recommend.Recommendation
	selectedRecommendation int

	// Main menu list
	mainList list.Model

	// Quick launch / smart predictions
	selectedModel  int
	recommendation *LaunchRecommendation

	// Advanced config
	ctxSize      string
	ctxMode      string
	kvPlacement  string
	parallel     string
	aitune       bool
	aituneRounds int
	benchmark    bool
	vision       bool
	keepalive    bool

	// Tuned config
	tunedConfigs []tune.ConfigEntry
	tunedIndex   int // -1 = auto, 0+ = selected
	tunePath     string

	// Inputs
	input     textinput.Model
	inputMode string

	// Settings screen (arrow-navigable list of all config options)
	settingsCfg    *config.Config
	settingsCursor int

	// Advanced (per-launch) config screen cursor
	cfgCursor int

	// First-run / quick-launch action-menu cursor
	menuCursor int

	// Generic arrow-select screen
	choiceTitle   string
	choiceOptions []string
	choiceCursor  int
	choiceApply   func(*Model, string)
	choiceReturn  Screen

	// Launch request (set when user chooses to launch)
	launchRequest *LaunchRequest

	// Messages
	message     string
	messageType string // info, warning, error
}

// ModelItem represents a discovered GGUF model.
type ModelItem struct {
	Name   string
	Path   string
	Tuned  int
	SizeGB float64
	Arch   string
	MaxCtx int // trained max context from GGUF
	FitCtx int // empirically proven fit context from probes
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
	cfg, _ := config.Load()
	settingsPath := config.Path()
	backend := cfg.Backend
	if backend == "" {
		backend = "ik_llama"
	}
	rounds := cfg.TuneRounds
	if rounds <= 0 {
		rounds = 8
	}
	m := Model{
		screen:       ScreenMain,
		backend:      backend,
		modelDir:     cfg.ModelDir,
		settingsPath: settingsPath,
		cacheDir:     cfg.CacheDir,
		ctxSize:      "fit",
		ctxMode:      "fit",
		kvPlacement:  cfg.KVPlacement,
		aituneRounds: rounds,
	}
	if m.kvPlacement == "" {
		m.kvPlacement = "auto"
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

	// Detect hardware
	caps, _ := detect.Detect()
	m.caps = caps
	m.recommendationGroups = recommend.TopCategories(caps, 4)
	m.recommendations = flattenRecommendationCategories(m.recommendationGroups)

	if len(m.models) == 0 {
		m.screen = ScreenFirstRun
	}

	m.mainList = newMainList(m.models)
	return m
}

func flattenRecommendationCategories(cats recommend.Categories) []recommend.Recommendation {
	total := len(cats.Balanced) + len(cats.Smartest) + len(cats.Fastest)
	rows := make([]recommend.Recommendation, 0, total)
	rows = append(rows, cats.Balanced...)
	rows = append(rows, cats.Smartest...)
	rows = append(rows, cats.Fastest...)
	return rows
}

func newMainList(models []ModelItem) list.Model {
	items := []list.Item{
		mainItem{title: "r. Recommended downloads", desc: "Top models that fit this machine; Vulkan fallback aware", isAction: true, action: "recommend"},
	}
	for i, m := range models {
		desc := fmt.Sprintf("%.1fGB, %s", m.SizeGB, m.Arch)
		if m.Tuned > 0 {
			desc += fmt.Sprintf("  [tuned: %d]", m.Tuned)
		}
		items = append(items, mainItem{
			title:   fmt.Sprintf("%d. %s", i+1, m.Name),
			desc:    desc,
			index:   i,
			isModel: true,
		})
	}
	// Minimal action items
	items = append(items, mainItem{title: "d. Download model", desc: "Get from Hugging Face", isAction: true, action: "download"})
	items = append(items, mainItem{title: "m. Model directory", desc: "Change search path", isAction: true, action: "modeldir"})
	items = append(items, mainItem{title: "b. Backend", desc: "Switch llama / ik_llama", isAction: true, action: "backend"})
	items = append(items, mainItem{title: "s. Settings", desc: "All options (arrow keys)", isAction: true, action: "settings"})
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
			if m.screen == ScreenChoice {
				m.screen = m.choiceReturn
				return m, nil
			}
			if m.inputMode == "setting" {
				m.inputMode = ""
				return m, nil
			}
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
	case ScreenRecommended:
		return m.updateRecommended(msg)
	case ScreenSettings:
		return m.updateSettings(msg)
	case ScreenChoice:
		return m.updateChoice(msg)
	case ScreenDownload, ScreenBackend:
		return m.updateInputScreen(msg)
	}

	return m, nil
}

func (m Model) updateMain(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "r", "R":
			m.selectedRecommendation = 0
			m.screen = ScreenRecommended
			return m, nil
		case "enter":
			if item, ok := m.mainList.SelectedItem().(mainItem); ok {
				if item.isModel {
					m.selectedModel = item.index
					m.recommendation = computeRecommendation(m.caps, m.models[m.selectedModel])
					m.menuCursor = 0
					m.screen = ScreenLaunchPrompt
					return m, nil
				}
				switch item.action {
				case "recommend":
					m.selectedRecommendation = 0
					m.screen = ScreenRecommended
					return m, nil
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
				case "backend":
					m.openBackendChoice(ScreenMain)
				case "settings":
					m.openSettings()
				case "quit":
					return m, tea.Quit
				}
			}
		case "s", "S":
			m.openSettings()
		case "b", "B":
			m.openBackendChoice(ScreenMain)
		}
	}

	var cmd tea.Cmd
	m.mainList, cmd = m.mainList.Update(msg)
	return m, cmd
}

func launchPromptActions() []string { return []string{"launch", "advanced", "dryrun"} }

func (m Model) doLaunchPromptAction(action string) (tea.Model, tea.Cmd) {
	switch action {
	case "launch":
		m.screen = ScreenPrelaunch
	case "advanced":
		m.cfgCursor = 0
		m.screen = ScreenModelConfig
	case "dryrun":
		m.message = fmt.Sprintf("Dry run: %s", strings.Join(m.buildArgs(), " "))
		m.messageType = "info"
	}
	return m, nil
}

func (m Model) updateLaunchPrompt(msg tea.Msg) (tea.Model, tea.Cmd) {
	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	actions := launchPromptActions()
	if m.menuCursor >= len(actions) {
		m.menuCursor = len(actions) - 1
	}
	switch keyMsg.String() {
	case "up":
		if m.menuCursor > 0 {
			m.menuCursor--
		}
	case "down":
		if m.menuCursor < len(actions)-1 {
			m.menuCursor++
		}
	case "enter":
		return m.doLaunchPromptAction(actions[m.menuCursor])
	case "c", "C":
		return m.doLaunchPromptAction("advanced")
	case "d", "D":
		return m.doLaunchPromptAction("dryrun")
	}
	return m, nil
}

// cfgRows returns the ordered focusable rows of the Advanced config screen.
func (m Model) cfgRows() []string {
	rows := []string{"context", "parallel", "kv", "tuned", "aitune"}
	if m.aitune {
		rows = append(rows, "rounds")
	}
	return append(rows, "vision", "benchmark", "keepalive", "launch", "dryrun")
}

func (m *Model) openCfgInput(mode, val, placeholder string) {
	m.inputMode = mode
	m.input.SetValue(val)
	m.input.Placeholder = placeholder
	m.input.Focus()
}

func (m *Model) setCtx(val string) {
	val = strings.TrimSpace(val)
	switch val {
	case "", "fit":
		m.ctxMode = "fit"
		m.ctxSize = "fit"
	case "max":
		m.ctxMode = "max"
		m.ctxSize = "max"
	default:
		m.ctxMode = "manual"
		m.ctxSize = val
	}
}

// cycleCfgRow changes the focused Advanced-config row with ←/→ (dir -1/+1).
func (m *Model) cycleCfgRow(row string, dir int) {
	switch row {
	case "kv":
		order := []string{"auto", "gpu", "cpu"}
		if dir < 0 {
			m.kvPlacement = prevOption(order, m.kvPlacement)
		} else {
			m.kvPlacement = nextOption(order, m.kvPlacement)
		}
	case "context":
		order := []string{"fit", "max"}
		cur := "fit"
		if m.ctxMode == "max" {
			cur = "max"
		}
		if dir < 0 {
			m.setCtx(prevOption(order, cur))
		} else {
			m.setCtx(nextOption(order, cur))
		}
	case "aitune":
		m.aitune = !m.aitune
	case "vision":
		m.vision = !m.vision
	case "benchmark":
		m.benchmark = !m.benchmark
	case "keepalive":
		m.keepalive = !m.keepalive
	}
}

// activateCfgRow handles Enter on the focused Advanced-config row.
func (m Model) activateCfgRow(row string) (tea.Model, tea.Cmd) {
	switch row {
	case "context":
		m.openCfgInput("ctx", m.ctxSize, "fit, max, or token count")
	case "parallel":
		m.openCfgInput("parallel", m.parallel, "Parallel slots (blank = let placement decide)")
	case "kv":
		m.cycleCfgRow("kv", 1)
	case "tuned":
		m.tunedConfigs = tune.ListTunedConfigs(m.cacheDir, m.models[m.selectedModel].Name, m.backendTag(), false)
		m.tunedIndex = -1
		m.screen = ScreenTunedPicker
	case "rounds":
		m.openCfgInput("aitune", strconv.Itoa(m.aituneRounds), "AI tune rounds (1-30, default 8)")
	case "aitune":
		m.aitune = !m.aitune
	case "vision":
		m.vision = !m.vision
	case "benchmark":
		m.benchmark = !m.benchmark
	case "keepalive":
		m.keepalive = !m.keepalive
	case "launch":
		m.screen = ScreenPrelaunch
	case "dryrun":
		m.message = fmt.Sprintf("Dry run: %s", strings.Join(m.buildArgs(), " "))
		m.messageType = "info"
	}
	return m, nil
}

func (m Model) updateModelConfig(msg tea.Msg) (tea.Model, tea.Cmd) {
	if len(m.models) == 0 {
		m.screen = ScreenMain
		return m, nil
	}

	// Free-text edit mode (context / parallel / AI-tune rounds).
	if m.inputMode != "" {
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		if keyMsg, ok := msg.(tea.KeyMsg); ok && keyMsg.String() == "enter" {
			val := m.input.Value()
			switch m.inputMode {
			case "ctx":
				m.setCtx(val)
			case "parallel":
				m.parallel = strings.TrimSpace(val)
			case "aitune":
				if n, err := strconv.Atoi(strings.TrimSpace(val)); err == nil && n >= 1 && n <= 30 {
					m.aituneRounds = n
				}
			}
			m.inputMode = ""
		}
		return m, cmd
	}

	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	rows := m.cfgRows()
	if m.cfgCursor >= len(rows) {
		m.cfgCursor = len(rows) - 1
	}
	switch keyMsg.String() {
	// Arrow-key navigation (works alongside the letter hotkeys below).
	case "up":
		if m.cfgCursor > 0 {
			m.cfgCursor--
		}
	case "down":
		if m.cfgCursor < len(rows)-1 {
			m.cfgCursor++
		}
	case "left":
		m.cycleCfgRow(rows[m.cfgCursor], -1)
	case "right":
		m.cycleCfgRow(rows[m.cfgCursor], 1)
	case "enter":
		return m.activateCfgRow(rows[m.cfgCursor])
	// Letter hotkeys.
	case "c", "C":
		m.openCfgInput("ctx", m.ctxSize, "fit, max, or token count")
	case "p", "P":
		m.openCfgInput("parallel", m.parallel, "Parallel slots (blank = let placement decide)")
	case "K":
		m.cycleCfgRow("kv", 1)
	case "a", "A":
		m.aitune = !m.aitune
	case "r", "R":
		if m.aitune {
			m.openCfgInput("aitune", strconv.Itoa(m.aituneRounds), "AI tune rounds (1-30, default 8)")
		}
	case "b", "B":
		m.benchmark = !m.benchmark
	case "v", "V":
		m.vision = !m.vision
	case "k":
		m.keepalive = !m.keepalive
	case "l", "L":
		m.screen = ScreenPrelaunch
	case "d", "D":
		m.message = fmt.Sprintf("Dry run: %s", strings.Join(m.buildArgs(), " "))
		m.messageType = "info"
	case "t", "T":
		m.tunedConfigs = tune.ListTunedConfigs(m.cacheDir, m.models[m.selectedModel].Name, m.backendTag(), false)
		m.tunedIndex = -1
		m.screen = ScreenTunedPicker
	}
	return m, nil
}

func firstRunActions() []string { return []string{"recommend", "download", "modeldir", "quit"} }

func (m Model) doFirstRunAction(action string) (tea.Model, tea.Cmd) {
	switch action {
	case "recommend":
		m.selectedRecommendation = 0
		m.screen = ScreenRecommended
	case "download":
		m.screen = ScreenDownload
		m.inputMode = "download"
		m.input.SetValue("")
		m.input.Placeholder = "Hugging Face repo"
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
	return m, nil
}

func (m Model) updateFirstRun(msg tea.Msg) (tea.Model, tea.Cmd) {
	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	actions := firstRunActions()
	if m.menuCursor >= len(actions) {
		m.menuCursor = len(actions) - 1
	}
	switch keyMsg.String() {
	case "up":
		if m.menuCursor > 0 {
			m.menuCursor--
		}
	case "down":
		if m.menuCursor < len(actions)-1 {
			m.menuCursor++
		}
	case "enter":
		return m.doFirstRunAction(actions[m.menuCursor])
	case "r", "R":
		return m.doFirstRunAction("recommend")
	case "d", "D":
		return m.doFirstRunAction("download")
	case "m", "M":
		return m.doFirstRunAction("modeldir")
	case "q", "Q":
		return m, tea.Quit
	}
	return m, nil
}

func (m Model) updateInputScreen(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	if keyMsg, ok := msg.(tea.KeyMsg); ok && keyMsg.String() == "enter" {
		val := m.input.Value()
		switch m.inputMode {
		case "download":
			val = strings.TrimSpace(val)
			if val != "" {
				m.launchRequest = &LaunchRequest{DownloadRepo: val}
				return m, tea.Quit
			}
			m.message = "Enter a Hugging Face GGUF repository"
			m.messageType = "warning"
		case "modeldir":
			val = strings.TrimSpace(val)
			if val != "" {
				m.modelDir = val
				m.models = discoverModels(m.modelDir)
				m.refreshTunedCounts()
				if err := persistConfig(func(c *config.Config) { c.ModelDir = val }); err != nil {
					m.message = fmt.Sprintf("Using %s for this session — could not save config: %v", val, err)
					m.messageType = "warning"
				} else {
					m.message = fmt.Sprintf("Model directory saved: %s (%d models)", val, len(m.models))
					m.messageType = "info"
				}
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
	case ScreenRecommended:
		return m.viewRecommended()
	case ScreenSettings:
		return m.viewSettings()
	case ScreenChoice:
		return m.viewChoice()
	case ScreenDownload, ScreenBackend:
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
	b.WriteString(mutedStyle.Render("  [r] Recommended downloads  [s] Settings  [b] Backend"))

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

	actions := firstRunActions()
	labels := map[string]string{
		"recommend": "[r] Recommended downloads for this machine",
		"download":  "[d] Manual Hugging Face repository",
		"modeldir":  "[m] Point at an existing model directory",
		"quit":      "[q] Quit",
	}
	for i, a := range actions {
		if i == m.menuCursor {
			b.WriteString(selectedStyle.Render("▸ "+labels[a]) + "\n")
		} else {
			b.WriteString("  " + labels[a] + "\n")
		}
	}
	b.WriteString("\n" + mutedStyle.Render("  ↑/↓ move · Enter select · q quit"))
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

	actions := launchPromptActions()
	labels := map[string]string{
		"launch":   "Launch with recommendations",
		"advanced": "[c] Advanced config",
		"dryrun":   "[d] Dry run",
	}
	for i, a := range actions {
		if i == m.menuCursor {
			b.WriteString(selectedStyle.Render("▸ "+labels[a]) + "\n")
		} else {
			b.WriteString("  " + labels[a] + "\n")
		}
	}
	b.WriteString("\n" + mutedStyle.Render("  ↑/↓ move · Enter select · Esc back"))

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

	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ Advanced: %s ═══", model.Name)) + "\n\n")

	rows := m.cfgRows()
	focused := ""
	if m.cfgCursor < len(rows) {
		focused = rows[m.cfgCursor]
	}
	line := func(key, label, value string) {
		text := fmt.Sprintf("%-24s %s", label, value)
		if key == focused {
			b.WriteString(selectedStyle.Render("▸ "+text) + "\n")
		} else {
			b.WriteString("  " + text + "\n")
		}
	}

	ctxLabel := m.ctxSize
	if m.ctxMode == "fit" {
		ctxLabel = "fit"
	}
	if model.FitCtx > 0 || model.MaxCtx > 0 {
		ctxHint := " ("
		if model.FitCtx > 0 {
			ctxHint += fmt.Sprintf("fits ~%d", model.FitCtx)
		}
		if model.FitCtx > 0 && model.MaxCtx > 0 {
			ctxHint += ", "
		}
		if model.MaxCtx > 0 {
			ctxHint += fmt.Sprintf("train max %d", model.MaxCtx)
		}
		ctxLabel += ctxHint + ")"
	}
	line("context", "[c] Context size", ctxLabel)

	parallelLabel := m.parallel
	if parallelLabel == "" {
		parallelLabel = "default (4)"
	}
	line("parallel", "[p] Parallel slots", parallelLabel)

	kvLabel := "auto (GPU KV first)"
	if m.kvPlacement == "gpu" {
		kvLabel = "gpu (best long-context decode)"
	} else if m.kvPlacement == "cpu" {
		kvLabel = "cpu (more GPU experts for short chat)"
	}
	line("kv", "[K] KV placement", kvLabel)

	tuneLabel := "auto"
	if m.tunePath != "" {
		tuneLabel = filepath.Base(m.tunePath)
	}
	line("tuned", "[t] Tuned config", tuneLabel)
	line("aitune", "[a] AI tune", boolLabel(m.aitune))
	if m.aitune {
		line("rounds", "[r] AI tune rounds", strconv.Itoa(m.aituneRounds))
	}
	line("vision", "[v] Vision (mmproj)", boolLabel(m.vision))
	line("benchmark", "[b] Benchmark mode", boolLabel(m.benchmark))
	line("keepalive", "[k] Keep-alive restart", boolLabel(m.keepalive))
	b.WriteString("\n")
	line("launch", "[L] Launch", "")
	line("dryrun", "[D] Dry run", "")
	b.WriteString("\n" + mutedStyle.Render("  ↑/↓ navigate · ←/→ change · Enter select/launch · Esc back"))

	if m.inputMode != "" {
		b.WriteString("\n\n  " + m.input.View())
	}
	if m.message != "" {
		b.WriteString("\n  ")
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
			return "1 (default)"
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

func (m Model) updateRecommended(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			if len(m.models) == 0 {
				m.screen = ScreenFirstRun
			} else {
				m.screen = ScreenMain
			}
			return m, nil
		case "up", "k":
			if len(m.recommendations) == 0 {
				return m, nil
			}
			m.selectedRecommendation--
			if m.selectedRecommendation < 0 {
				m.selectedRecommendation = len(m.recommendations) - 1
			}
		case "down", "j":
			if len(m.recommendations) == 0 {
				return m, nil
			}
			m.selectedRecommendation++
			if m.selectedRecommendation >= len(m.recommendations) {
				m.selectedRecommendation = 0
			}
		case "enter":
			if len(m.recommendations) > 0 && m.selectedRecommendation >= 0 && m.selectedRecommendation < len(m.recommendations) {
				rec := m.recommendations[m.selectedRecommendation]
				m.launchRequest = &LaunchRequest{DownloadRepo: rec.Repo, DownloadQuant: rec.QuantName}
				return m, tea.Quit
			}
		case "d", "D":
			m.screen = ScreenDownload
			m.inputMode = "download"
			m.input.SetValue("")
			m.input.Placeholder = "Hugging Face repo"
			m.input.Focus()
			return m, nil
		}
	}
	return m, nil
}

func (m Model) viewRecommended() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ Recommended Downloads ═══") + "\n")
	b.WriteString(fmt.Sprintf("  Hardware: %s\n", hwSummary(m.caps)))
	b.WriteString("  " + recommend.CatalogAttribution() + "\n\n")

	if len(m.recommendations) == 0 {
		b.WriteString(warningStyle.Render("  No safe recommendation fits the detected RAM/VRAM."))
		b.WriteString("\n  [d] Manual Hugging Face repository  [Esc] Back\n")
		return b.String()
	}

	idx := 0
	writeGroup := func(title string, rows []recommend.Recommendation) {
		if len(rows) == 0 {
			return
		}
		b.WriteString(recommendStyle.Render("  "+title) + "\n")
		for _, rec := range rows {
			prefix := "  "
			if idx == m.selectedRecommendation {
				prefix = selectedStyle.Render("▸ ")
			}
			quant := rec.QuantName
			if quant == "" {
				quant = "auto"
			}
			tps := "—"
			if rec.PredictedTPS > 0 {
				tps = fmt.Sprintf("%.0f t/s", rec.PredictedTPS)
			}
			name := rec.Name
			if len(name) > 34 {
				name = name[:33] + "…"
			}
			line := fmt.Sprintf("%-34s %-9s %-8s %5.1fG %3.0f%% %7s",
				name, recommend.DisplayFit(rec.Fit), quant, rec.QuantSizeGB, rec.QualityRetained*100, tps)
			if idx == m.selectedRecommendation {
				b.WriteString(prefix + selectedStyle.Render(line) + "\n")
			} else {
				b.WriteString(prefix + line + "\n")
			}
			idx++
		}
		b.WriteString("\n")
	}
	writeGroup("Best overall — balanced quality, speed and fit", m.recommendationGroups.Balanced)
	writeGroup("Smartest — highest intelligence that fits", m.recommendationGroups.Smartest)
	writeGroup("Fastest — quickest while still capable", m.recommendationGroups.Fastest)

	b.WriteString(highlightStyle.Render("  [Enter] Download selected"))
	b.WriteString("\n  [d] Manual repo  [Esc] Back  [↑/↓] Navigate\n")
	return b.String()
}

func (m Model) viewInputScreen() string {
	var b strings.Builder
	var title string
	switch m.inputMode {
	case "download":
		title = "Download Model"
	case "modeldir":
		title = "Model Directory"
	default:
		title = "Input"
	}
	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ %s ═══", title)) + "\n\n")
	b.WriteString(m.input.View())
	b.WriteString("\n\n  Press Enter to confirm, Esc to cancel")
	if m.message != "" {
		b.WriteString("\n\n  ")
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
		rec.GPULayers = -1                                         // all layers
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

// persistConfig loads the current config, applies mutate, and writes it back to
// the canonical config file, preserving all other settings so GUI changes
// survive across sessions.
func persistConfig(mutate func(*config.Config)) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	mutate(cfg)
	return cfg.Save()
}

// refreshTunedCounts recomputes per-model tuned config counts for the current
// backend and rebuilds the main list so the counts reflect the active backend.
func (m *Model) refreshTunedCounts() {
	tag := m.backendTag()
	for i := range m.models {
		m.models[i].Tuned = tune.CountTunedConfigs(m.cacheDir, m.models[i].Name, tag)
	}
	m.mainList = newMainList(m.models)
}

// settingRow describes one editable config setting on the Settings screen.
// kind is "enum" (pick from options), "bool" (toggle), or "text" (free input).
type settingRow struct {
	label   string
	kind    string
	options []string
	get     func(*config.Config) string
	set     func(*config.Config, string)
}

// settingRows returns every setting shown on the Settings screen, in order.
func settingRows() []settingRow {
	atoiSet := func(assign func(*config.Config, int)) func(*config.Config, string) {
		return func(c *config.Config, v string) {
			if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
				assign(c, n)
			}
		}
	}
	return []settingRow{
		{label: "Backend", kind: "enum", options: []string{"ik_llama", "llama"},
			get: func(c *config.Config) string { return c.Backend },
			set: func(c *config.Config, v string) { c.Backend = v }},
		{label: "Model directory", kind: "text",
			get: func(c *config.Config) string { return c.ModelDir },
			set: func(c *config.Config, v string) { c.ModelDir = v }},
		{label: "Context", kind: "enum", options: []string{"fit", "max"},
			get: func(c *config.Config) string { return c.CtxValue() },
			set: func(c *config.Config, v string) {
				if v == "max" {
					c.CtxMode = "max"
				} else {
					c.CtxMode = "fit"
				}
				c.CtxSize = 0
			}},
		{label: "KV placement", kind: "enum", options: []string{"auto", "gpu", "cpu"},
			get: func(c *config.Config) string { return c.KVPlacement },
			set: func(c *config.Config, v string) { c.KVPlacement = v }},
		{label: "KV quality", kind: "enum", options: []string{"low", "mid", "high"},
			get: func(c *config.Config) string { return c.KVQuality },
			set: func(c *config.Config, v string) { c.KVQuality = v }},
		{label: "Speculative", kind: "enum",
			options: []string{"off", "auto", "draft", "eagle3", "ngram", "ngram-mod", "ngram-k4v", "mtp"},
			get:     func(c *config.Config) string { return c.Spec },
			set:     func(c *config.Config, v string) { c.Spec = v }},
		{label: "Vision", kind: "bool",
			get: func(c *config.Config) string { return boolLabel(c.Vision) },
			set: func(c *config.Config, v string) { c.Vision = v == "on" }},
		{label: "Port", kind: "text",
			get: func(c *config.Config) string { return strconv.Itoa(c.Port) },
			set: atoiSet(func(c *config.Config, n int) { c.Port = n })},
		{label: "Host", kind: "text",
			get: func(c *config.Config) string { return c.Host },
			set: func(c *config.Config, v string) { c.Host = strings.TrimSpace(v) }},
		{label: "Parallel", kind: "text",
			get: func(c *config.Config) string { return strconv.Itoa(c.Parallel) },
			set: atoiSet(func(c *config.Config, n int) { c.Parallel = n })},
		{label: "AI-tune rounds", kind: "text",
			get: func(c *config.Config) string { return strconv.Itoa(c.TuneRounds) },
			set: atoiSet(func(c *config.Config, n int) { c.TuneRounds = n })},
	}
}

// applySetting mutates the in-memory config, persists it to disk, and applies
// any side effects (re-scan models, refresh tuned counts) for the given row.
func (m *Model) applySetting(row settingRow, val string) {
	row.set(m.settingsCfg, val)
	if err := m.settingsCfg.Save(); err != nil {
		m.message = fmt.Sprintf("%s set to %s for this session — save failed: %v", row.label, val, err)
		m.messageType = "warning"
	} else {
		m.message = fmt.Sprintf("Saved: %s = %s", row.label, val)
		m.messageType = "info"
	}
	switch row.label {
	case "Backend":
		m.backend = val
		m.refreshTunedCounts()
	case "Model directory":
		m.modelDir = val
		m.models = discoverModels(val)
		m.refreshTunedCounts()
		if m.messageType != "warning" {
			m.message = fmt.Sprintf("Saved: Model directory = %s (%d models)", val, len(m.models))
		}
	}
}

// openChoice configures and shows the generic arrow-select screen.
func (m *Model) openChoice(title string, options []string, current string, ret Screen, apply func(*Model, string)) {
	m.choiceTitle = title
	m.choiceOptions = options
	m.choiceCursor = indexOf(options, current)
	if m.choiceCursor < 0 {
		m.choiceCursor = 0
	}
	m.choiceApply = apply
	m.choiceReturn = ret
	m.screen = ScreenChoice
}

// openSettings loads the current config and shows the Settings screen.
func (m *Model) openSettings() {
	cfg, err := config.Load()
	if err != nil || cfg == nil {
		cfg = config.Defaults()
	}
	m.settingsCfg = cfg
	m.settingsCursor = 0
	m.inputMode = ""
	m.screen = ScreenSettings
}

// openBackendChoice shows the arrow-select backend chooser, persisting the
// choice and returning to ret afterwards.
func (m *Model) openBackendChoice(ret Screen) {
	m.openChoice("Backend", []string{"ik_llama", "llama"}, m.backend, ret, func(mm *Model, v string) {
		mm.backend = v
		if err := persistConfig(func(c *config.Config) { c.Backend = v }); err != nil {
			mm.message = fmt.Sprintf("Backend set to %s for this session — save failed: %v", v, err)
			mm.messageType = "warning"
		} else {
			mm.message = "Backend saved: " + v
			mm.messageType = "info"
		}
		mm.refreshTunedCounts()
	})
}

func indexOf(opts []string, v string) int {
	for i, o := range opts {
		if o == v {
			return i
		}
	}
	return -1
}

func prevOption(opts []string, v string) string {
	i := indexOf(opts, v)
	if i <= 0 {
		return opts[len(opts)-1]
	}
	return opts[i-1]
}

func nextOption(opts []string, v string) string {
	i := indexOf(opts, v)
	if i < 0 || i >= len(opts)-1 {
		return opts[0]
	}
	return opts[i+1]
}

func (m Model) updateChoice(msg tea.Msg) (tea.Model, tea.Cmd) {
	if keyMsg, ok := msg.(tea.KeyMsg); ok {
		switch keyMsg.String() {
		case "up", "k":
			if m.choiceCursor > 0 {
				m.choiceCursor--
			}
		case "down", "j":
			if m.choiceCursor < len(m.choiceOptions)-1 {
				m.choiceCursor++
			}
		case "enter", " ":
			if m.choiceApply != nil && m.choiceCursor < len(m.choiceOptions) {
				m.choiceApply(&m, m.choiceOptions[m.choiceCursor])
			}
			m.screen = m.choiceReturn
		}
	}
	return m, nil
}

func (m Model) updateSettings(msg tea.Msg) (tea.Model, tea.Cmd) {
	rows := settingRows()

	// Free-text edit mode for "text" settings.
	if m.inputMode == "setting" {
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		if keyMsg, ok := msg.(tea.KeyMsg); ok && keyMsg.String() == "enter" {
			val := strings.TrimSpace(m.input.Value())
			if val != "" && m.settingsCursor < len(rows) {
				m.applySetting(rows[m.settingsCursor], val)
			}
			m.inputMode = ""
		}
		return m, cmd
	}

	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	row := rows[m.settingsCursor]
	switch keyMsg.String() {
	case "up", "k":
		if m.settingsCursor > 0 {
			m.settingsCursor--
		}
	case "down", "j":
		if m.settingsCursor < len(rows)-1 {
			m.settingsCursor++
		}
	case "enter":
		switch row.kind {
		case "enum":
			m.openChoice(row.label, row.options, row.get(m.settingsCfg), ScreenSettings,
				func(mm *Model, v string) { mm.applySetting(row, v) })
		case "bool":
			m.applySetting(row, toggleBool(row.get(m.settingsCfg)))
		case "text":
			m.inputMode = "setting"
			m.input.SetValue(row.get(m.settingsCfg))
			m.input.Placeholder = row.label
			m.input.Focus()
		}
	case "right", "l":
		if row.kind == "enum" {
			m.applySetting(row, nextOption(row.options, row.get(m.settingsCfg)))
		} else if row.kind == "bool" {
			m.applySetting(row, toggleBool(row.get(m.settingsCfg)))
		}
	case "left", "h":
		if row.kind == "enum" {
			m.applySetting(row, prevOption(row.options, row.get(m.settingsCfg)))
		} else if row.kind == "bool" {
			m.applySetting(row, toggleBool(row.get(m.settingsCfg)))
		}
	case "e", "E":
		editor := os.Getenv("EDITOR")
		if editor == "" {
			editor = "nano"
		}
		c := exec.Command(editor, m.settingsPath)
		c.Stdin, c.Stdout, c.Stderr = os.Stdin, os.Stdout, os.Stderr
		c.Run()
		if cfg, err := config.Load(); err == nil {
			m.settingsCfg = cfg
		}
	}
	return m, nil
}

func toggleBool(cur string) string {
	if cur == "on" {
		return "off"
	}
	return "on"
}

func (m Model) viewChoice() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ %s ═══", m.choiceTitle)) + "\n\n")
	for i, opt := range m.choiceOptions {
		if i == m.choiceCursor {
			b.WriteString("  " + selectedStyle.Render("> "+opt) + "\n")
		} else {
			b.WriteString("    " + opt + "\n")
		}
	}
	b.WriteString("\n" + mutedStyle.Render("  ↑/↓ select · Enter confirm · Esc cancel"))
	return b.String()
}

func (m Model) viewSettings() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ Settings ═══") + "\n\n")
	rows := settingRows()
	for i, row := range rows {
		line := fmt.Sprintf("%-17s %s", row.label+":", row.get(m.settingsCfg))
		if i == m.settingsCursor {
			b.WriteString("  " + selectedStyle.Render("> "+line) + "\n")
		} else {
			b.WriteString("    " + line + "\n")
		}
	}
	if m.inputMode == "setting" {
		b.WriteString("\n  " + m.input.View() + "\n")
	}
	b.WriteString("\n" + mutedStyle.Render("  ↑/↓ navigate · Enter/→ change · ←/→ cycle enums · [e] edit file · Esc back"))
	b.WriteString("\n" + mutedStyle.Render("  config: "+m.settingsPath))
	if m.message != "" {
		b.WriteString("\n  ")
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

func (m Model) buildLaunchRequest() *LaunchRequest {
	if len(m.models) == 0 || m.selectedModel < 0 || m.selectedModel >= len(m.models) {
		return nil
	}
	model := m.models[m.selectedModel]
	// Default: 0 = auto-fit (Compute() finds max context that fits hardware)
	ctx := 0
	if m.recommendation != nil && m.recommendation.ContextSize > 0 {
		ctx = m.recommendation.ContextSize
	}
	if m.ctxMode == "max" {
		ctx = 131072
	} else if m.ctxMode == "manual" && m.ctxSize != "" {
		if n, err := strconv.Atoi(m.ctxSize); err == nil {
			ctx = n
		}
	} else if m.ctxMode == "fit" {
		ctx = 0 // auto-fit
	}
	gpuLayers := 999
	if m.recommendation != nil {
		gpuLayers = m.recommendation.GPULayers
	}
	parallel := 1
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
	DownloadRepo  string
	DownloadQuant string
	ModelPath     string
	Port          int
	CtxSize       int
	KVPlacement   string
	KVQuality     string
	GPULayers     int
	FlashAttn     bool
	Parallel      int
	Vision        bool
	Backend       string
	TuneCache     string
	AITune        bool
	AITuneRounds  int
	Benchmark     bool
	KeepAlive     bool
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
