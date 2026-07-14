package tui

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/gguf"
	"github.com/raketenkater/ggrun/pkg/probe"
	"github.com/raketenkater/ggrun/pkg/recommend"
	"github.com/raketenkater/ggrun/pkg/tune"
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
	port                   int
	recommendationGroups   recommend.Categories
	recommendations        []recommend.Recommendation
	selectedRecommendation int

	// Main menu list
	mainList list.Model

	// Quick launch / smart predictions
	selectedModel int

	// Advanced config
	ctxSize        string
	ctxMode        string
	kvPlacement    string
	kvQuality      string
	vramHeadroomMB int
	ramHeadroomMB  int
	parallel       string
	parallelSet    bool
	aitune         bool
	aituneRounds   int
	benchmark      bool
	vision         bool
	claudeCode     bool

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

	// Recommended-downloads memory reserve control focus: "", "vram", or "ram".
	recHeadroomFocus string

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
	Name        string
	Path        string
	Tuned       int
	SizeGB      float64
	Arch        string
	IsMoE       bool
	AutoBackend string // registered architecture route used when backend=auto
	MaxCtx      int    // trained max context from GGUF
	FitCtx      int    // empirically proven fit context from probes
}

func InitialModel() Model {
	cfg, err := config.Load()
	if err != nil || cfg == nil {
		cfg = config.Defaults()
	}
	settingsPath := config.Path()
	backend := cfg.Backend
	if backend == "" {
		backend = "auto"
	}
	rounds := cfg.TuneRounds
	if rounds <= 0 {
		rounds = 8
	}
	ctxValue := cfg.CtxValue()
	ctxMode := "fit"
	switch cfg.CtxMode {
	case "max", "native":
		ctxMode = "max"
	case "manual":
		ctxMode = "manual"
	}
	parallel := ""
	if cfg.Parallel > 0 {
		parallel = strconv.Itoa(cfg.Parallel)
	}
	m := Model{
		screen:       ScreenMain,
		backend:      backend,
		modelDir:     cfg.ModelDir,
		settingsPath: settingsPath,
		cacheDir:     cfg.CacheDir,
		port:         cfg.Port,
		ctxSize:      ctxValue,
		ctxMode:      ctxMode,
		kvPlacement:  cfg.KVPlacement,
		kvQuality:    cfg.KVQuality,
		parallel:     parallel,
		vision:       cfg.Vision,
		aituneRounds: rounds,
	}
	if m.port <= 0 {
		m.port = 8081
	}
	if m.ctxSize == "" {
		m.ctxSize = "fit"
	}
	if m.kvPlacement == "" {
		m.kvPlacement = "auto"
	}
	if m.kvQuality == "" {
		m.kvQuality = "mid"
	}

	m.input = textinput.New()
	m.input.Placeholder = ""
	m.input.Focus()

	// Detect once, then reuse that result while enriching every discovered GGUF.
	caps, _ := detect.Detect()
	m.caps = caps
	m.models = loadModels(m.modelDir, m.cacheDir, m.backend, m.caps)

	m.vramHeadroomMB = config.ParseBudgetMB(cfg.VRAMHeadroom)
	m.ramHeadroomMB = config.ParseBudgetMB(cfg.RAMHeadroom)
	m.refreshRecommendations()

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
		mainItem{title: "r. Recommended downloads", desc: "Best models and quants that fit this computer", isAction: true, action: "recommend"},
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
	items = append(items, mainItem{title: "b. Backend", desc: "Auto-select or choose an installed backend", isAction: true, action: "backend"})
	items = append(items, mainItem{title: "s. Settings", desc: "All options (arrow keys)", isAction: true, action: "settings"})
	items = append(items, mainItem{title: "u. Update", desc: "Update ggrun and installed backends", isAction: true, action: "update"})
	items = append(items, mainItem{title: "q. Quit", desc: "Exit", isAction: true, action: "quit"})

	l := list.New(items, mainItemDelegate{}, 40, 20)
	l.Title = ""
	l.SetShowStatusBar(false)
	l.SetFilteringEnabled(true)
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
func (i mainItem) FilterValue() string { return i.title + " " + i.desc }

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
		case "ctrl+c":
			return m, tea.Quit
		case "q":
			if m.screen == ScreenMain {
				if m.mainList.FilterState() != list.Filtering {
					return m, tea.Quit
				}
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
	// While the search box is active, every printable key belongs to the filter;
	// do not let shortcuts such as r/s/b steal letters from the query.
	if m.mainList.FilterState() == list.Filtering {
		var cmd tea.Cmd
		m.mainList, cmd = m.mainList.Update(msg)
		return m, cmd
	}
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
					// Open the config screen first so settings (context, KV,
					// Claude Code, …) are discoverable. The recommended defaults
					// are pre-filled, so launching is one more keypress: press L
					// (or Enter on the Launch row) to start.
					m.selectedModel = item.index
					m.cfgCursor = 0
					m.screen = ScreenModelConfig
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
				case "update":
					m.launchRequest = &LaunchRequest{Update: true}
					return m, tea.Quit
				case "quit":
					return m, tea.Quit
				}
			}
		case "s", "S":
			m.openSettings()
		case "u", "U":
			m.launchRequest = &LaunchRequest{Update: true}
			return m, tea.Quit
		case "b", "B":
			m.openBackendChoice(ScreenMain)
		case "c", "C":
			if item, ok := m.mainList.SelectedItem().(mainItem); ok && item.isModel {
				m.selectedModel = item.index
				m.cfgCursor = 0
				m.screen = ScreenModelConfig
				return m, nil
			}
		}
	}

	var cmd tea.Cmd
	m.mainList, cmd = m.mainList.Update(msg)
	return m, cmd
}

// cfgRows returns the ordered focusable rows of the Advanced config screen.
func (m Model) cfgRows() []string {
	rows := []string{"context", "parallel", "kv", "tuned", "aitune"}
	if m.aitune {
		rows = append(rows, "rounds")
	}
	return append(rows, "vision", "claudecode", "benchmark", "launch", "dryrun")
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
		if m.aitune {
			m.benchmark = false
		}
	case "vision":
		m.vision = !m.vision
	case "claudecode":
		m.claudeCode = !m.claudeCode
	case "benchmark":
		m.benchmark = !m.benchmark
		if m.benchmark {
			m.aitune = false
		}
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
		if m.aitune {
			m.benchmark = false
		}
	case "vision":
		m.vision = !m.vision
	case "claudecode":
		m.claudeCode = !m.claudeCode
	case "benchmark":
		m.benchmark = !m.benchmark
		if m.benchmark {
			m.aitune = false
		}
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
				m.parallelSet = m.parallel != ""
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
		if m.aitune {
			m.benchmark = false
		}
	case "r", "R":
		if m.aitune {
			m.openCfgInput("aitune", strconv.Itoa(m.aituneRounds), "AI tune rounds (1-30, default 8)")
		}
	case "b", "B":
		m.benchmark = !m.benchmark
		if m.benchmark {
			m.aitune = false
		}
	case "v", "V":
		m.vision = !m.vision
	case "x", "X":
		m.claudeCode = !m.claudeCode
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

func firstRunActions() []string {
	return []string{"recommend", "download", "modeldir", "update", "quit"}
}

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
	case "update":
		m.launchRequest = &LaunchRequest{Update: true}
		return m, tea.Quit
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
	case "u", "U":
		return m.doFirstRunAction("update")
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
				m.models = loadModels(m.modelDir, m.cacheDir, m.backend, m.caps)
				m.mainList = newMainList(m.models)
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

	b.WriteString(titleStyle.Render("═══ ggrun ═══") + "\n")
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
	b.WriteString(mutedStyle.Render("  Enter configure · / search · r downloads · s settings · u update"))

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
	b.WriteString(titleStyle.Render("═══ ggrun First Run ═══") + "\n")
	b.WriteString(fmt.Sprintf("  Hardware: %s\n", hwSummary(m.caps)))
	b.WriteString(fmt.Sprintf("  No runnable GGUF models found in: %s\n", m.modelDir))
	b.WriteString("  Start with Recommended; ggrun will choose a model and quant that fit.\n")
	b.WriteString("\n")

	actions := firstRunActions()
	labels := map[string]string{
		"recommend": "[r] Recommended downloads for this machine",
		"download":  "[d] Manual Hugging Face repository",
		"modeldir":  "[m] Point at an existing model directory",
		"update":    "[u] Update ggrun and backends",
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

func (m Model) viewModelConfig() string {
	if len(m.models) == 0 {
		return "No models"
	}
	model := m.models[m.selectedModel]
	var b strings.Builder

	b.WriteString(titleStyle.Render("⚙  Configure  ·  "+model.Name) + "\n")
	b.WriteString(mutedStyle.Render("  ↑/↓ move · ←/→ or Enter change · x Claude Code · Esc back") + "\n")

	rows := m.cfgRows()
	focused := ""
	if m.cfgCursor < len(rows) {
		focused = rows[m.cfgCursor]
	}
	line := func(key, label, value string) {
		if key == focused {
			b.WriteString(selectedStyle.Render(fmt.Sprintf("  ▸ %-26s %s", label, value)) + "\n")
		} else {
			b.WriteString(fmt.Sprintf("    %-26s ", label) + subtitleStyle.Render(value) + "\n")
		}
	}
	section := func(title string) {
		b.WriteString("\n" + recommendStyle.Render("  "+title) + "\n")
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
	parallelLabel := m.parallel
	if !m.parallelSet && (parallelLabel == "" || parallelLabel == "1") {
		parallelLabel = "automatic (1 normally; 2 for Claude Code)"
	}
	kvLabel := "auto (GPU KV first)"
	if m.kvPlacement == "gpu" {
		kvLabel = "gpu (best long-context decode)"
	} else if m.kvPlacement == "cpu" {
		kvLabel = "cpu (more GPU experts for short chat)"
	}
	tuneLabel := "auto"
	if m.tunePath != "" {
		tuneLabel = filepath.Base(m.tunePath)
	}

	kvQualityLabel := map[string]string{
		"high": "high (f16)", "mid": "mid (q8_0)", "low": "low (q4_0)",
	}[m.kvQuality]
	if kvQualityLabel == "" {
		kvQualityLabel = m.kvQuality
	}

	section("Context & memory")
	line("context", "[c] Context size", ctxLabel)
	line("parallel", "[p] Parallel slots", parallelLabel)
	line("kv", "[K] KV placement", kvLabel)
	line("kvq", "KV quality", kvQualityLabel+"  (change in Settings)")

	section("Tuning")
	line("tuned", "[t] Tuned config", tuneLabel)
	line("aitune", "[a] AI tune", boolLabel(m.aitune))
	if m.aitune {
		line("rounds", "[r] AI tune rounds", strconv.Itoa(m.aituneRounds))
	}

	section("Run mode")
	line("vision", "[v] Vision (mmproj)", boolLabel(m.vision))
	ccLabel := boolLabel(m.claudeCode)
	if m.claudeCode {
		ccLabel += " — serve + print Claude Code env (thinking on)"
	}
	line("claudecode", "[x] Claude Code", ccLabel)
	line("benchmark", "[b] Benchmark mode", boolLabel(m.benchmark))

	section("Actions")
	line("launch", "[L] Launch", "▶ start the server")
	line("dryrun", "[D] Dry run", "print the command, don't run")

	b.WriteString("\n" + mutedStyle.Render("  Enter on Launch to start · Esc to go back"))

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
	b.WriteString(fmt.Sprintf("  Backend:        %s\n", m.backend))
	if model.FitCtx > 0 {
		b.WriteString(fmt.Sprintf("  Fit estimate:   ~%d tokens\n", model.FitCtx))
	}
	if model.MaxCtx > 0 {
		b.WriteString(fmt.Sprintf("  Train max:      %d tokens\n", model.MaxCtx))
	}
	b.WriteString(fmt.Sprintf("  Parallel:       %s\n", func() string {
		if !m.parallelSet && (m.parallel == "" || m.parallel == "1") {
			if m.claudeCode {
				return "automatic (2 for Claude Code)"
			}
			return "automatic (1)"
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
	b.WriteString(fmt.Sprintf("  Claude Code:    %s\n", boolLabel(m.claudeCode)))
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
			if m.recHeadroomFocus != "" {
				m.recHeadroomFocus = ""
				m.message = ""
				return m, nil
			}
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
		case "left", "h":
			if m.recHeadroomFocus != "" {
				m.stepRecommendedHeadroom(-1)
				return m, nil
			}
		case "right", "l":
			if m.recHeadroomFocus != "" {
				m.stepRecommendedHeadroom(1)
				return m, nil
			}
		case "enter":
			if m.recHeadroomFocus != "" {
				m.recHeadroomFocus = ""
				return m, nil
			}
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
		case "v", "V":
			m.focusRecommendedHeadroom("vram")
			return m, nil
		case "m", "M":
			m.focusRecommendedHeadroom("ram")
			return m, nil
		}
	}
	return m, nil
}

func (m *Model) focusRecommendedHeadroom(kind string) {
	m.recHeadroomFocus = kind
	m.message = "Use ←/→ to reserve memory for desktop, browser, IDE, games, or other GPU/CPU work."
	m.messageType = "info"
}

func (m *Model) stepRecommendedHeadroom(dir int) {
	if m.recHeadroomFocus == "vram" {
		m.setRecommendedHeadroom("vram", stepHeadroomMB(m.vramHeadroomMB, recommendedVRAMHeadroomSteps(m.caps, m.vramHeadroomMB), dir))
	} else if m.recHeadroomFocus == "ram" {
		m.setRecommendedHeadroom("ram", stepHeadroomMB(m.ramHeadroomMB, recommendedRAMHeadroomSteps(m.caps, m.ramHeadroomMB), dir))
	}
}

func (m *Model) setRecommendedHeadroom(kind string, mb int) {
	val := ""
	if mb > 0 {
		val = formatHeadroomMB(mb)
	}
	label := "VRAM reserve"
	if kind == "ram" {
		label = "RAM reserve"
	}
	if err := persistConfig(func(c *config.Config) {
		if kind == "vram" {
			c.VRAMHeadroom = val
		} else {
			c.RAMHeadroom = val
		}
	}); err != nil {
		m.message = fmt.Sprintf("%s set for this session — save failed: %v", label, err)
		m.messageType = "warning"
	} else {
		m.message = fmt.Sprintf("Saved: %s = %s", label, formatHeadroomMB(mb))
		m.messageType = "info"
	}
	if kind == "vram" {
		m.vramHeadroomMB = mb
	} else {
		m.ramHeadroomMB = mb
	}
	m.refreshRecommendations()
}

func stepHeadroomMB(current int, steps []int, dir int) int {
	if len(steps) == 0 {
		return current
	}
	if dir > 0 {
		for _, step := range steps {
			if step > current {
				return step
			}
		}
		return steps[len(steps)-1]
	}
	for i := len(steps) - 1; i >= 0; i-- {
		if steps[i] < current {
			return steps[i]
		}
	}
	return steps[0]
}

func recommendedVRAMHeadroomSteps(caps *detect.Capabilities, current int) []int {
	steps := []int{0, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 36864, 40960}
	max := 0
	if caps != nil {
		max = caps.TotalVRAM()
	}
	return smartHeadroomSteps(steps, current, max)
}

func recommendedRAMHeadroomSteps(caps *detect.Capabilities, current int) []int {
	steps := []int{0, 4096, 8192, 16384, 32768, 49152, 65536, 98304}
	max := 0
	if caps != nil {
		max = caps.RAM.TotalMB
	}
	return smartHeadroomSteps(steps, current, max)
}

func smartHeadroomSteps(base []int, current, max int) []int {
	steps := append([]int(nil), base...)
	if current > 0 {
		steps = append(steps, current)
	}
	if max > 0 {
		// Leave at least a little memory visible to the recommender; reserving
		// everything is not useful as a preset.
		limit := max - min(2048, max/4)
		filtered := steps[:0]
		for _, step := range steps {
			if step <= limit {
				filtered = append(filtered, step)
			}
		}
		steps = filtered
	}
	sort.Ints(steps)
	uniq := steps[:0]
	last := -1
	for _, step := range steps {
		if step != last {
			uniq = append(uniq, step)
			last = step
		}
	}
	return uniq
}

func formatHeadroomMB(mb int) string {
	if mb <= 0 {
		return "0"
	}
	if mb%1024 == 0 {
		return fmt.Sprintf("%dG", mb/1024)
	}
	return fmt.Sprintf("%dM", mb)
}

func (m *Model) refreshRecommendations() {
	m.recommendationGroups = recommend.TopCategories(detect.ApplyRAMHeadroom(detect.ApplyVRAMHeadroom(m.caps, m.vramHeadroomMB), m.ramHeadroomMB), 4)
	m.recommendations = flattenRecommendationCategories(m.recommendationGroups)
	if len(m.recommendations) == 0 {
		m.selectedRecommendation = 0
		return
	}
	if m.selectedRecommendation < 0 {
		m.selectedRecommendation = 0
	}
	if m.selectedRecommendation >= len(m.recommendations) {
		m.selectedRecommendation = len(m.recommendations) - 1
	}
}

func (m Model) viewRecommended() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ Recommended Downloads ═══") + "\n")
	b.WriteString(fmt.Sprintf("  Hardware: %s\n", hwSummary(m.caps)))
	b.WriteString("  " + m.recommendedHeadroomControls() + "\n")
	b.WriteString("  " + recommend.CatalogAttribution() + "\n\n")

	if len(m.recommendations) == 0 {
		b.WriteString(warningStyle.Render("  No safe recommendation fits the detected RAM/VRAM."))
		b.WriteString("\n  [v] VRAM reserve  [m] RAM reserve  [d] Manual Hugging Face repository  [Esc] Back\n")
		if m.recHeadroomFocus != "" {
			b.WriteString("  ←/→ smart reserve steps · Enter/Esc done\n")
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
			b.WriteString("\n")
		}
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
				tps = fmt.Sprintf("~%.0f t/s", rec.PredictedTPS)
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
	b.WriteString(mutedStyle.Render("  Speeds are estimates; Benchmark measures this exact machine.") + "\n")
	b.WriteString(mutedStyle.Render("  Fit uses installed capacity; launch rechecks memory currently free.") + "\n\n")

	b.WriteString(highlightStyle.Render("  [Enter] Download selected"))
	b.WriteString("\n  [v] VRAM reserve  [m] RAM reserve  [d] Manual repo  [Esc] Back  [↑/↓] Navigate\n")
	if m.recHeadroomFocus != "" {
		b.WriteString("  ←/→ smart reserve steps · Enter/Esc done\n")
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
		b.WriteString("\n")
	}
	return b.String()
}

func (m Model) recommendedHeadroomControls() string {
	vram := fmt.Sprintf("[v] VRAM %s", formatHeadroomMB(m.vramHeadroomMB))
	ram := fmt.Sprintf("[m] RAM %s", formatHeadroomMB(m.ramHeadroomMB))
	if m.recHeadroomFocus == "vram" {
		vram = selectedStyle.Render("▸ " + vram + " ◂")
	}
	if m.recHeadroomFocus == "ram" {
		ram = selectedStyle.Render("▸ " + ram + " ◂")
	}
	return fmt.Sprintf("Reserve for other apps: %s   %s", vram, ram)
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
	parts := make([]string, 0, len(caps.GPUs))
	for _, g := range caps.GPUs {
		name := strings.TrimPrefix(g.Name, "NVIDIA GeForce ")
		parts = append(parts, fmt.Sprintf("%s %.0fG", name, float64(g.VRAMTotalMB)/1024))
	}
	return fmt.Sprintf("%s · %dGB RAM · %d cores", strings.Join(parts, " + "), ramGB, caps.CPU.Cores)
}

func boolLabel(v bool) string {
	if v {
		return "on"
	}
	return "off"
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

		modelKey := filepath.Join(filepath.Dir(path), baseName)
		if seen[modelKey] {
			return nil
		}
		seen[modelKey] = true

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

func loadModels(dir, cacheDir, backend string, caps *detect.Capabilities) []ModelItem {
	models := discoverModels(dir)
	totalSysMemMB := 0
	if caps != nil {
		totalSysMemMB = caps.TotalVRAM() + caps.RAM.TotalMB
	}
	backendTag := strings.TrimSpace(backend)
	switch backendTag {
	case "", "auto", "llama":
		backendTag = "llama"
	case "ik_llama":
		backendTag = "ik"
	}
	for i := range models {
		modelBackendTag := backendTag
		if info, err := gguf.Parse(models[i].Path); err == nil {
			models[i].MaxCtx = info.ContextLength
			models[i].IsMoE = info.IsMoE
			if info.Architecture != "" {
				models[i].Arch = info.Architecture
			}
			if info.IsMoE {
				models[i].Arch += " · MoE"
			}
			if (backend == "" || backend == "auto") && info.Architecture != "" {
				if routed := backends.ForArch(info.Architecture); routed != nil {
					modelBackendTag = routed.Tag
					models[i].AutoBackend = routed.Tag
				}
			}
			models[i].FitCtx = probe.EstimateFitCtxForInfo(models[i].Path, cacheDir, info, totalSysMemMB)
		}
		models[i].Tuned = tune.CountTunedConfigs(cacheDir, models[i].Name, modelBackendTag)
	}
	return models
}

func (m Model) backendTag() string {
	backend := strings.TrimSpace(m.backend)
	switch backend {
	case "ik_llama":
		return "ik"
	case "":
		return "llama"
	case "auto":
		return "llama"
	default:
		return backend
	}
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
		modelTag := tag
		if (m.backend == "" || m.backend == "auto") && m.models[i].AutoBackend != "" {
			modelTag = m.models[i].AutoBackend
		}
		m.models[i].Tuned = tune.CountTunedConfigs(m.cacheDir, m.models[i].Name, modelTag)
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
		{label: "Backend", kind: "enum", options: backendOptions(),
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
		{label: "VRAM headroom", kind: "text",
			get: func(c *config.Config) string {
				if strings.TrimSpace(c.VRAMHeadroom) == "" {
					return "0"
				}
				return c.VRAMHeadroom
			},
			set: func(c *config.Config, v string) { c.VRAMHeadroom = strings.TrimSpace(v) }},
		{label: "RAM headroom", kind: "text",
			get: func(c *config.Config) string {
				if strings.TrimSpace(c.RAMHeadroom) == "" {
					return "0"
				}
				return c.RAMHeadroom
			},
			set: func(c *config.Config, v string) { c.RAMHeadroom = strings.TrimSpace(v) }},
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
	case "Context":
		m.setCtx(m.settingsCfg.CtxValue())
	case "KV placement":
		m.kvPlacement = val
	case "KV quality":
		// Sync the live session too — otherwise the saved value only applies
		// after a TUI restart while the current session keeps launching with
		// the startup-time quality.
		m.kvQuality = val
	case "VRAM headroom":
		m.vramHeadroomMB = config.ParseBudgetMB(val)
		m.refreshRecommendations()
	case "RAM headroom":
		m.ramHeadroomMB = config.ParseBudgetMB(val)
		m.refreshRecommendations()
	case "Backend":
		m.backend = val
		m.refreshTunedCounts()
	case "Model directory":
		m.modelDir = val
		m.models = loadModels(val, m.cacheDir, m.backend, m.caps)
		m.mainList = newMainList(m.models)
		if m.messageType != "warning" {
			m.message = fmt.Sprintf("Saved: Model directory = %s (%d models)", val, len(m.models))
		}
	case "Vision":
		m.vision = m.settingsCfg.Vision
	case "Port":
		m.port = m.settingsCfg.Port
	case "Parallel":
		m.parallel = ""
		m.parallelSet = false
		if m.settingsCfg.Parallel > 0 {
			m.parallel = strconv.Itoa(m.settingsCfg.Parallel)
		}
	case "AI-tune rounds":
		m.aituneRounds = m.settingsCfg.TuneRounds
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

// backendOptions lists the selectable backends: the built-ins plus any
// registered fork backends (so they show up in the TUI backend picker).
func backendOptions() []string {
	opts := []string{"auto", "llama", "ik_llama"}
	opts = append(opts, backends.Tags()...)
	return opts
}

// openBackendChoice shows the arrow-select backend chooser, persisting the
// choice and returning to ret afterwards.
func (m *Model) openBackendChoice(ret Screen) {
	m.openChoice("Backend", backendOptions(), m.backend, ret, func(mm *Model, v string) {
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
	// Default to auto-fit (ctx=0): placement.Compute finds the max context that
	// actually fits — the same path the CLI uses. The old default fed the crude
	// computeRecommendation heuristic straight to the backend, which produced a
	// wrong context for big MoE models. Only an explicit max/manual choice
	// overrides auto-fit.
	ctx := 0
	ctxFlag := "fit"
	if m.ctxMode == "max" {
		ctxFlag = "max"
	} else if m.ctxMode == "manual" && m.ctxSize != "" {
		ctxFlag = m.ctxSize
		if n, err := strconv.Atoi(m.ctxSize); err == nil {
			ctx = n
		}
	}
	parallel := 1
	parallelSet := m.parallelSet
	if m.parallel != "" {
		if n, err := strconv.Atoi(m.parallel); err == nil {
			parallel = n
		}
	}
	return &LaunchRequest{
		ModelPath:   model.Path,
		Port:        m.port,
		CtxSize:     ctx,
		CtxFlag:     ctxFlag,
		KVPlacement: m.kvPlacement,
		// The configured KV quality, not a hardcoded default: passing a fixed
		// "mid" here overrode the user's saved setting with --kv-quality mid
		// on every TUI launch (settings appeared to save but never applied).
		KVQuality:    m.kvQuality,
		FlashAttn:    true,
		Parallel:     parallel,
		ParallelSet:  parallelSet,
		Vision:       m.vision,
		Backend:      m.backend,
		TuneCache:    m.tunePath,
		AITune:       m.aitune,
		AITuneRounds: m.aituneRounds,
		Benchmark:    m.benchmark,
		ClaudeCode:   m.claudeCode,
	}
}

func (m Model) buildArgs() []string {
	req := m.buildLaunchRequest()
	if req == nil {
		return nil
	}
	return append([]string{"ggrun", "dry-run"}, req.LaunchArgs()...)
}

// LaunchRequest is returned when the user chooses to launch a model.
type LaunchRequest struct {
	Update        bool
	DownloadRepo  string
	DownloadQuant string
	ModelPath     string
	Port          int
	CtxSize       int
	CtxFlag       string
	KVPlacement   string
	KVQuality     string
	FlashAttn     bool
	Parallel      int
	ParallelSet   bool // user typed a parallel value (claude-code mode must not override)
	Vision        bool
	Backend       string
	TuneCache     string
	AITune        bool
	AITuneRounds  int
	Benchmark     bool
	ClaudeCode    bool
}

func (req *LaunchRequest) LaunchArgs() []string {
	if req == nil {
		return nil
	}
	args := []string{req.ModelPath}
	if req.Port > 0 {
		args = append(args, "--port", strconv.Itoa(req.Port))
	}
	if req.CtxFlag != "" {
		args = append(args, "--ctx-size", req.CtxFlag)
	} else if req.CtxSize > 0 {
		args = append(args, "--ctx-size", strconv.Itoa(req.CtxSize))
	} else {
		args = append(args, "--ctx-size", "fit")
	}
	if req.KVPlacement != "" {
		args = append(args, "--kv-placement", req.KVPlacement)
	}
	if req.KVQuality != "" {
		args = append(args, "--kv-quality", req.KVQuality)
	}
	if req.Vision {
		args = append(args, "--vision")
	}
	if req.Backend != "" && req.Backend != "auto" {
		args = append(args, "--backend", req.Backend)
	}
	if req.TuneCache != "" {
		args = append(args, "--tune-cache", req.TuneCache)
	}
	if req.AITune && req.AITuneRounds > 0 {
		args = append(args, "--rounds", strconv.Itoa(req.AITuneRounds))
	}
	if req.ParallelSet && req.Parallel > 0 {
		args = append(args, "--parallel", strconv.Itoa(req.Parallel))
	}
	if req.Benchmark {
		args = append(args, "--benchmark")
	}
	if req.ClaudeCode {
		args = append(args, "--claude-code")
	}
	return args
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
