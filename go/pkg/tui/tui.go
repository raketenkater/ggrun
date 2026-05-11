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
)

var (
	titleStyle        = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#7D56F4"))
	subtitleStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("#A0A0A0"))
	selectedStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("#7D56F4")).Bold(true)
	highlightStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#00FF00"))
	warningStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFAA00"))
	errorStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF0000"))
)

// Screen represents the current TUI screen.
type Screen int

const (
	ScreenMain Screen = iota
	ScreenModelConfig
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

	// Main menu list
	mainList    list.Model

	// Model config
	selectedModel int
	ctxSize       string
	ctxMode       string
	kvPlacement   string
	parallel      string
	aitune        bool
	aituneRounds  int
	benchmark     bool
	vision        bool
	keepalive     bool

	// Inputs
	input         textinput.Model
	inputMode     string

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
}

type ModelsLoadedMsg []ModelItem
type CapabilitiesMsg *detect.Capabilities

func InitialModel() Model {
	m := Model{
		screen:       ScreenMain,
		backend:      "ik_llama",
		modelDir:     os.Getenv("HOME") + "/ai_models",
		settingsPath: os.Getenv("HOME") + "/.config/llm-server/config",
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
	// Add action items
	items = append(items, mainItem{title: "c. Advanced configure", desc: "Configure model settings", isAction: true, action: "configure"})
	items = append(items, mainItem{title: "s. Settings", desc: "Edit configuration file", isAction: true, action: "settings"})
	items = append(items, mainItem{title: "d. Download model", desc: "Download from Hugging Face", isAction: true, action: "download"})
	items = append(items, mainItem{title: "b. Backend", desc: "Select backend (llama/ik_llama)", isAction: true, action: "backend"})
	items = append(items, mainItem{title: "m. Model directory", desc: "Change model search path", isAction: true, action: "modeldir"})
	items = append(items, mainItem{title: "u. Check updates", desc: "Check for llm-server updates", isAction: true, action: "update"})
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
	case ScreenModelConfig:
		return m.updateModelConfig(msg)
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
					m.screen = ScreenModelConfig
					return m, nil
				}
				switch item.action {
				case "configure":
					if len(m.models) > 0 {
						m.selectedModel = 0
						m.screen = ScreenModelConfig
					}
				case "settings":
					m.screen = ScreenSettings
					m.inputMode = "settings"
					m.input.SetValue("")
					m.input.Placeholder = "Press Enter to open in $EDITOR"
					m.input.Focus()
				case "download":
					m.screen = ScreenDownload
					m.inputMode = "download"
					m.input.SetValue("")
					m.input.Placeholder = "Hugging Face repo (e.g. unsloth/Llama-3.2-1B-Instruct)"
					m.input.Focus()
				case "backend":
					m.screen = ScreenBackend
					m.inputMode = "backend"
					m.input.SetValue(m.backend)
					m.input.Placeholder = "llama or ik_llama"
					m.input.Focus()
				case "modeldir":
					m.screen = ScreenBackend
					m.inputMode = "modeldir"
					m.input.SetValue(m.modelDir)
					m.input.Placeholder = "Path to model directory"
					m.input.Focus()
				case "update":
					m.message = "Update check not yet implemented in Go TUI"
					m.messageType = "info"
				case "quit":
					return m, tea.Quit
				}
			}
		}
	}

	var cmd tea.Cmd
	m.mainList, cmd = m.mainList.Update(msg)
	return m, cmd
}

func (m Model) updateModelConfig(msg tea.Msg) (tea.Model, tea.Cmd) {
	if len(m.models) == 0 {
		m.screen = ScreenMain
		return m, nil
	}
	model := m.models[m.selectedModel]

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
			m.message = fmt.Sprintf("Launch %s (not yet implemented)", model.Name)
			m.messageType = "info"
		case "d", "D":
			m.message = fmt.Sprintf("Dry run %s (not yet implemented)", model.Name)
			m.messageType = "info"
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
	case ScreenModelConfig:
		return m.viewModelConfig()
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
	b.WriteString(fmt.Sprintf("  [c] Context size       %s\n", ctxLabel))
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
	b.WriteString(fmt.Sprintf("  [a] AI tune            %s\n", boolLabel(m.aitune)))
	if m.aitune {
		b.WriteString(fmt.Sprintf("  [r] AI tune rounds     %d\n", m.aituneRounds))
	}
	b.WriteString(fmt.Sprintf("  [v] Vision (mmproj)    %s\n", boolLabel(m.vision)))
	b.WriteString(fmt.Sprintf("  [b] Benchmark mode     %s\n", boolLabel(m.benchmark)))
	b.WriteString(fmt.Sprintf("  [k] Keep-alive restart %s\n", boolLabel(m.keepalive)))
	b.WriteString("\n")
	b.WriteString("  [L] Launch    [D] Dry run    [<] Back (Esc)\n")

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
			for _, m := range matches {
				st, err := os.Stat(m)
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

// Run starts the TUI.
func Run() error {
	p := tea.NewProgram(InitialModel(), tea.WithAltScreen())
	_, err := p.Run()
	return err
}
