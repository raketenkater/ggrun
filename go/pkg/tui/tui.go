package tui

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/chattemplate"
	"github.com/raketenkater/ggrun/pkg/claudesession"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/gguf"
	modelstore "github.com/raketenkater/ggrun/pkg/models"
	"github.com/raketenkater/ggrun/pkg/modelusage"
	"github.com/raketenkater/ggrun/pkg/placement"
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
	ScreenLoading
)

// Model is the Bubble Tea model.
type Model struct {
	screen Screen
	width  int
	height int

	// Data
	caps   *detect.Capabilities
	models []ModelItem
	// modelUsage holds per-model launch usage (count + last-used), used to sort
	// the main list so the most-used and most-recent models surface first. It is
	// loaded once at startup; launch records are written by the CLI launcher.
	modelUsage             map[string]modelusage.Record
	backend                string
	modelDir               string
	settingsPath           string
	cacheDir               string
	port                   int
	recommendationGroups   recommend.Categories
	recommendations        []recommend.Recommendation
	selectedRecommendation int
	// pendingDownload holds the repo/quant already chosen while the destination
	// directory is being asked for, so a download is only emitted once both
	// halves are known.
	pendingDownload *LaunchRequest

	// Main menu list
	mainList list.Model

	// Quick launch / smart predictions
	selectedModel int

	// Advanced config
	ctxSize        string
	ctxMode        string
	kvPlacement    string
	kvQuality      string
	swaFull        bool
	vramHeadroomMB int
	ramHeadroomMB  int
	parallel       string
	parallelSet    bool
	aitune         bool
	aituneRounds   int
	benchmark      bool
	vision         bool
	claudeCode     bool
	// claudeReviewer picks the local worker/reviewer model for Claude Code:
	// "auto" (and "qwen") resolve to the Qwen3.5-4B worker/reviewer, "qwen2b"
	// forces the small/light Qwen3.5-2B review-only profile, and "nanbeige"
	// forces the NanoBeige4.2 big-MoE worker. Empty means auto. "off" seats no
	// reviewer at all, so the main model reviews its own output.
	claudeReviewer string
	supportExpert  string
	supportOnline  bool
	// supportSet carries a replayed launch's explicit support choice. A normal
	// launch leaves it false so cmdLaunch applies the saved setting exactly as
	// the CLI does: sending the saved value as --no-support-online made it a
	// user instruction and blocked online research on an escalated failure.
	supportSet bool
	// noCachedConfig derives this launch fresh, ignoring cached placement/probe
	// measurements (--no-cached-config). Per-launch; never persisted.
	noCachedConfig bool
	// claudeProfile is deliberately per-launch: an empty value preserves the
	// CLI default instead of writing a scheduling policy into user config.
	claudeProfile string
	// chatTemplate forces a corrected chat template from the data-driven catalog
	// (pkg/chattemplate), mirroring the CLI's --chat-template <name>. Empty means
	// "auto" — the launch auto-matches known-broken templates. Per-launch only.
	chatTemplate string

	// cfgMemo caches the two expensive per-render Advanced-config computations
	// (the Full-SWA KV estimate and the Chat-template label). bubbletea calls
	// View() after every Update, so on the model-config screen every keystroke —
	// arrow-down row cycling and free-text typing included — used to re-run them
	// synchronously on the single event-loop goroutine. They live behind a shared
	// pointer so the cache survives the value copies of Model that bubbletea
	// makes between Update and View, and are invalidated only by the setters that
	// change the inputs they depend on (setCtx, cycleCfgRow, openCfgInput).
	cfgMemo *configMemo

	// Tuned config
	tunedConfigs []tune.ConfigEntry
	tunedIndex   int // -1 = auto, 0+ = selected
	tunePath     string

	// Inputs
	input     textinput.Model
	inputMode string

	// Settings screen (arrow-navigable list of all config options)
	settingsCfg      *config.Config
	swaFullTouched   bool
	kvQualityTouched bool
	settingsCursor   int
	ramLimitPercent  int

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

	// Strong typed confirmation for deleting a model discovered outside the
	// primary directory. The user must type "yes" into the input screen before
	// removeModelAt may call modelstore.RemoveExternal on that row.
	pendingExternalRemoveIdx int

	// Resumable Claude Code session for the working directory, discovered when
	// the pre-launch screen is opened.
	resumeSession string
	resumeRun     string
	resumeCached  int

	// Launch request (set when user chooses to launch)
	launchRequest *LaunchRequest
	// replayRequest is the exact last TUI request while its pre-launch review
	// screen is open. It is kept separate from the editable Model fields so
	// confirming a replay cannot silently regenerate different arguments.
	replayRequest *LaunchRequest
	replaySavedAt time.Time
	// backendRouteBypass is set only when the user explicitly chooses to try a
	// model once without its reviewed architecture backend. It is per-selection
	// runtime state and is never persisted.
	backendRouteBypass        bool
	backendRouteBypassBackend string

	// Messages
	message        string
	messageType    string // info, warning, error
	scanningModels bool
	spinner        spinner.Model
}

// ModelItem represents a discovered GGUF model.
type ModelItem struct {
	Name          string
	Path          string
	Tuned         int
	SizeGB        float64
	Arch          string
	Architecture  string // exact GGUF architecture, without display suffixes
	IsMoE         bool
	AutoBackend   string                  // installed route selected for this architecture
	BackendRecipe string                  // reviewed route recipe when no installed route exists
	MaxCtx        int                     // trained max context from GGUF
	FitCtx        int                     // empirically proven fit context from probes
	External      bool                    // found outside the configured primary model directory
	KVProfile     *placement.ModelProfile // metadata for the live Full-SWA KV estimate
	// ChatTemplate is the model's embedded tokenizer.chat_template, captured ONCE
	// at enrich time (like KVProfile) so the config screen's chat-template row
	// never re-opens the .gguf and walks its metadata on every render.
	ChatTemplate string
}

type modelScanFinishedMsg struct {
	result   modelstore.ScanResult
	cacheErr error
}

func scanComputerModels(modelDir, cacheDir string) tea.Cmd {
	return func() tea.Msg {
		result := modelstore.ScanComputer(modelDir)
		return modelScanFinishedMsg{
			result:   result,
			cacheErr: modelstore.SaveDiscoveredScan(cacheDir, result),
		}
	}
}

func InitialModel() Model {
	m := sessionModel()
	m.loadHardwareAndModels()
	return m
}

func loadingModel() Model {
	m := sessionModel()
	m.screen = ScreenLoading
	return m
}

func sessionModel() Model {
	cfg, err := config.Load()
	configErr := err
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
	ctxMode := cfg.CtxMode()
	parallel := ""
	if cfg.Parallel > 0 {
		parallel = strconv.Itoa(cfg.Parallel)
	}
	// Config.Save serializes the built-in one-slot default. Loading it must
	// not turn automatic Claude scheduling into --parallel 1. Environment
	// overrides and selections made in the per-model editor remain explicit.
	parallelExplicit := cfg.Parallel > 0 && cfg.IsExplicit("PARALLEL") &&
		(cfg.Parallel > 1 || os.Getenv("LLM_PARALLEL") != "")
	spin := spinner.New()
	spin.Spinner = spinner.MiniDot
	spin.Style = titleStyle
	m := Model{
		screen:          ScreenMain,
		cfgMemo:         &configMemo{},
		backend:         backend,
		modelDir:        cfg.ModelDir,
		settingsPath:    settingsPath,
		cacheDir:        cfg.CacheDir,
		port:            cfg.Port,
		ctxSize:         ctxValue,
		ctxMode:         ctxMode,
		kvPlacement:     cfg.KVPlacement,
		kvQuality:       cfg.KVQuality,
		swaFull:         cfg.SWAFull,
		parallel:        parallel,
		parallelSet:     parallelExplicit,
		vision:          cfg.Vision,
		supportExpert:   cfg.SupportExpert,
		supportOnline:   cfg.SupportOnline,
		aituneRounds:    rounds,
		ramLimitPercent: cfg.RAMLimitPercent,
		spinner:         spin,
	}
	m.modelUsage = modelusage.Load(cfg.CacheDir)
	if m.port <= 0 {
		m.port = 8081
	}
	m.pendingExternalRemoveIdx = -1
	if m.ctxSize == "" {
		m.ctxSize = "fit"
	}
	if m.kvPlacement == "" {
		m.kvPlacement = "auto"
	}
	if m.kvQuality == "" {
		m.kvQuality = "auto"
	}

	m.input = textinput.New()
	m.input.Placeholder = ""
	m.input.Focus()
	m.vramHeadroomMB = config.ParseBudgetMB(cfg.VRAMHeadroom)
	m.ramHeadroomMB = config.ParseBudgetMB(cfg.RAMHeadroom)
	m.mainList = newMainList(nil, m.modelUsage)
	if configErr != nil {
		m.message = fmt.Sprintf("Warning: Configuration error: %v. Fix it with ggrun config edit or reset before launching.", configErr)
		m.messageType = "warning"
	}
	return m
}

func (m *Model) loadHardwareAndModels() {
	caps, _ := detect.Detect()
	m.caps = caps
	m.models = loadRecognizedModels(m.modelDir, m.cacheDir, m.backend, caps)
	m.refreshRecommendations()
	if len(m.models) == 0 {
		m.screen = ScreenFirstRun
	} else {
		m.screen = ScreenMain
	}
	m.mainList = newMainList(m.models, m.modelUsage)
}

type startupReadyMsg struct {
	caps      *detect.Capabilities
	models    []ModelItem
	recGroups recommend.Categories
	recs      []recommend.Recommendation
}

func loadHardwareAndModelsCmd(modelDir, cacheDir, backend string, ramLimit, vramHeadroomMB, ramHeadroomMB int) tea.Cmd {
	return func() tea.Msg {
		caps, _ := detect.Detect()
		models := loadRecognizedModels(modelDir, cacheDir, backend, caps)
		tmp := Model{
			caps:            caps,
			ramLimitPercent: ramLimit,
			vramHeadroomMB:  vramHeadroomMB,
			ramHeadroomMB:   ramHeadroomMB,
		}
		tmp.refreshRecommendations()
		return startupReadyMsg{
			caps: caps, models: models,
			recGroups: tmp.recommendationGroups, recs: tmp.recommendations,
		}
	}
}

func flattenRecommendationCategories(cats recommend.Categories) []recommend.Recommendation {
	total := len(cats.Balanced) + len(cats.Smartest) + len(cats.Fastest)
	rows := make([]recommend.Recommendation, 0, total)
	rows = append(rows, cats.Balanced...)
	rows = append(rows, cats.Smartest...)
	rows = append(rows, cats.Fastest...)
	return rows
}

// mainModelDesc renders one model's list description. Usage (Launches /
// LastUsedAt) already drives the sort order, so it is shown here as a badge —
// otherwise a non-alphabetical list looks random. The same-basename ambiguity
// case (two model-Q4.gguf in different directories) is disambiguated by always
// showing a location hint: the parent dir for primary models, "discovered" for
// external ones.
func mainModelDesc(m ModelItem, usage *modelusage.Record) string {
	desc := fmt.Sprintf("%.1fGB · %s", m.SizeGB, m.Arch)
	if m.AutoBackend != "" {
		desc += " · backend " + m.AutoBackend
	} else if m.BackendRecipe != "" {
		desc += " · backend " + m.BackendRecipe
	}
	if m.Tuned > 0 {
		desc += fmt.Sprintf(" · tuned %d", m.Tuned)
	}
	// The fits estimate is the single most useful datapoint for dense models.
	if m.FitCtx > 0 {
		desc += fmt.Sprintf(" · fits ~%dk", m.FitCtx/1000)
	}
	if usage != nil && usage.Launches > 0 {
		desc += fmt.Sprintf(" · %dx", usage.Launches)
		if !usage.LastUsedAt.IsZero() {
			desc += " " + timeAgo(usage.LastUsedAt)
		}
	}
	if m.External {
		desc += " · discovered"
	} else if dir := filepath.Base(filepath.Dir(m.Path)); dir != "." && dir != "" && dir != filepath.Base(m.Path) {
		desc += " · " + dir
	}
	return desc
}

// timeAgo renders a last-used timestamp compactly ("3d ago"), so the usage
// badge stays short enough for a 80-column terminal.
func timeAgo(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Hour:
		mins := int(d.Minutes())
		if mins < 1 {
			mins = 1
		}
		return fmt.Sprintf("%dm ago", mins)
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	case d < 30*24*time.Hour:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	default:
		return t.Local().Format("2006-01-02")
	}
}

func newMainList(models []ModelItem, usageArg ...map[string]modelusage.Record) list.Model {
	var usage map[string]modelusage.Record
	if len(usageArg) > 0 {
		usage = usageArg[0]
	}
	items := []list.Item{
		mainItem{title: "r. Recommended downloads", desc: "Best models and quants that fit this computer", isAction: true, action: "recommend"},
		mainItem{title: "l. Run latest configuration", desc: "Review and replay the exact previous TUI launch", isAction: true, action: "latest"},
		mainItem{title: "p. Scan computer for models", desc: "Find GGUFs on all local disks and remember their paths", isAction: true, action: "scan"},
	}
	for i, m := range models {
		var rec *modelusage.Record
		if usage != nil {
			if r, ok := usage[modelIdentity(m.Path)]; ok && r.Launches > 0 {
				rec = &r
			}
		}
		items = append(items, mainItem{
			title:   m.Name,
			desc:    mainModelDesc(m, rec),
			index:   i,
			isModel: true,
		})
	}
	// Minimal action items
	items = append(items, mainItem{title: "d. Download model", desc: "Get from Hugging Face", isAction: true, action: "download"})
	items = append(items, mainItem{title: "m. Model directory", desc: "Change search path", isAction: true, action: "modeldir"})
	items = append(items, mainItem{title: "b. Backend", desc: "Auto-select or choose an installed backend", isAction: true, action: "backend"})
	items = append(items, mainItem{title: "f. Backend forks", desc: "Install or manage llama.cpp-compatible forks", isAction: true, action: "backend-forks"})
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

// rebuildMainList replaces m.mainList (e.g. after the model set changes) and
// re-applies the real terminal size plus the outgoing list's cursor position
// and active search filter. newMainList's own list.New call always starts a
// brand-new list.Model at index 0, unfiltered, with a placeholder 40x20 size;
// without re-applying all three, any mid-session rebuild (a model delete, a
// model-directory change, a tuned-count refresh after switching backends)
// silently resets the user's scroll position and drops whatever they were
// searching for, on top of the terminal-size regression this helper was
// first written to fix. Before the first WindowSizeMsg arrives (m.width/
// m.height still zero, e.g. during initial Model construction), this
// intentionally leaves the placeholder size in place — the upcoming
// WindowSizeMsg sizes it correctly on its own; there's no prior selection or
// filter to restore at that point either.
func (m *Model) rebuildMainList() {
	prevIndex := m.mainList.Index()
	prevFilterState := m.mainList.FilterState()
	prevFilterValue := m.mainList.FilterValue()

	sortModels(m.models, m.modelUsage)
	m.mainList = newMainList(m.models, m.modelUsage)
	if m.width > 0 {
		m.mainList.SetWidth(m.width - 4)
	}
	if m.height > 0 {
		m.mainList.SetHeight(m.height - 12)
	}

	if prevFilterState != list.Unfiltered && prevFilterValue != "" {
		m.mainList.SetFilterText(prevFilterValue)
	}
	// Clamp against VisibleItems(), not Items(): once a filter is applied,
	// list.Select()/the paginator index into the FILTERED set, and clamping
	// against the raw (unfiltered) count can leave prevIndex pointing past
	// the last filtered page — the next View() then slices out of range and
	// panics, killing the whole TUI session.
	if itemCount := len(m.mainList.VisibleItems()); itemCount > 0 {
		if prevIndex >= itemCount {
			prevIndex = itemCount - 1
		}
		if prevIndex > 0 {
			m.mainList.Select(prevIndex)
		}
	}
}

// openTunedPicker rebuilds the tuned-config list for the current model and
// opens the picker with the cursor on whatever config is actually active
// (m.tunePath), instead of always defaulting to "Auto". Without this,
// reopening the picker just to look — then reflexively pressing Enter to
// close it — silently reverted an already-chosen tuned config back to Auto.
func (m *Model) openTunedPicker() {
	m.tunedConfigs = tune.ListTunedConfigs(m.cacheDir, m.models[m.selectedModel].Name, m.backendTag(), false)
	m.tunedIndex = -1
	if m.tunePath != "" {
		for i, c := range m.tunedConfigs {
			if c.Path == m.tunePath {
				m.tunedIndex = i
				break
			}
		}
	}
	m.screen = ScreenTunedPicker
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
		fmt.Fprint(w, selectedStyle.Render("> "+i.title)+"\n  "+subtitleStyle.Render(i.desc))
	} else {
		fmt.Fprint(w, "  "+i.title+"\n  "+subtitleStyle.Render(i.desc))
	}
}

func (m Model) Init() tea.Cmd {
	if m.screen != ScreenLoading {
		return nil
	}
	return tea.Batch(
		m.spinner.Tick,
		loadHardwareAndModelsCmd(m.modelDir, m.cacheDir, m.backend, m.ramLimitPercent, m.vramHeadroomMB, m.ramHeadroomMB),
	)
}

func (m Model) updateLoading(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case startupReadyMsg:
		m.caps = msg.caps
		m.models = msg.models
		m.recommendationGroups = msg.recGroups
		m.recommendations = msg.recs
		if len(m.models) == 0 {
			m.screen = ScreenFirstRun
		} else {
			m.screen = ScreenMain
		}
		m.rebuildMainList()
		return m, nil
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if m.screen == ScreenLoading {
		return m.updateLoading(msg)
	}
	switch msg := msg.(type) {
	case modelScanFinishedMsg:
		m.scanningModels = false
		// A scan can finish while the user is already configuring a model. Keep
		// both that active model and the highlighted Main-menu row attached to
		// their paths while the newly found models are merged and sorted.
		selectedModelID := ""
		if m.selectedModel >= 0 && m.selectedModel < len(m.models) {
			selectedModelID = modelIdentity(m.models[m.selectedModel].Path)
		}
		focusedModelID, focusedAction := "", ""
		if raw := m.mainList.SelectedItem(); raw != nil {
			if item, ok := raw.(mainItem); ok {
				focusedAction = item.action
				if item.isModel && item.index >= 0 && item.index < len(m.models) {
					focusedModelID = modelIdentity(m.models[item.index].Path)
				}
			}
		}
		// Use this scan's in-memory result even if persistence failed; a cache
		// write problem must not discard paths we just found for this session.
		items := mergeModelItems(discoverModels(m.modelDir), discoverModelsFromPaths(msg.result.Paths))
		m.models = enrichModelItems(items, m.cacheDir, m.backend, m.caps)
		m.rebuildMainList()
		if selectedModelID != "" {
			for i := range m.models {
				if modelIdentity(m.models[i].Path) == selectedModelID {
					m.selectedModel = i
					break
				}
			}
		}
		for i, raw := range m.mainList.VisibleItems() {
			item, ok := raw.(mainItem)
			if !ok {
				continue
			}
			if focusedModelID != "" && item.isModel && item.index >= 0 && item.index < len(m.models) && modelIdentity(m.models[item.index].Path) == focusedModelID {
				m.mainList.Select(i)
				break
			}
			if focusedModelID == "" && focusedAction != "" && item.action == focusedAction {
				m.mainList.Select(i)
				break
			}
		}
		external := 0
		for _, model := range m.models {
			if model.External {
				external++
			}
		}
		seconds := msg.result.Duration.Round(100 * time.Millisecond)
		m.message = fmt.Sprintf("Computer scan complete: %d runnable model(s), %d outside the primary directory (%s)", len(m.models), external, seconds)
		m.messageType = "info"
		if msg.result.Truncated {
			m.message += "; scan limit reached, showing partial results"
			m.messageType = "warning"
		}
		if msg.cacheErr != nil {
			m.message += fmt.Sprintf("; results could not be saved: %v", msg.cacheErr)
			m.messageType = "warning"
		}
		if m.screen == ScreenFirstRun && len(m.models) > 0 {
			m.screen = ScreenMain
		}
		return m, nil

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
			// Cancel any in-progress free-text edit. On ScreenModelConfig/
			// ScreenSettings this stays put - there's a row menu on the same
			// screen to fall back into. ScreenDownload/ScreenBackend are pure
			// single-field prompts with no such fallback (updateInputScreen
			// has no "esc" case of its own), so clearing the input there also
			// has to leave the screen in the same keypress, or it dead-ends on
			// a blurred, unresponsive input box until a second Esc/Enter.
			if m.inputMode != "" {
				m.inputMode = ""
				m.input.Blur()
				if m.screen == ScreenDownload || m.screen == ScreenBackend {
					if m.screen == ScreenDownload {
						m.pendingExternalRemoveIdx = -1
					}
					m.screen = ScreenMain
					m.message = ""
				}
				return m, nil
			}
			// These screens own their own "esc" case (back a level, or defocus a
			// control before leaving) - let the per-screen dispatch below run
			// instead of jumping straight to Main and skipping it.
			switch m.screen {
			case ScreenPrelaunch, ScreenTunedPicker, ScreenRecommended:
			case ScreenMain:
			default:
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
		case "p", "P":
			return m.startComputerModelScan()
		case "l", "L":
			return m.openLatestLaunch()
		case "enter":
			if item, ok := m.mainList.SelectedItem().(mainItem); ok {
				if item.isModel {
					// Open the config screen first so settings (context, KV,
					// Claude Code, …) are discoverable. The recommended defaults
					// are pre-filled, so launching is one more keypress: press L
					// (or Enter on the Launch row) to start.
					m.selectedModel = item.index
					m.backendRouteBypass = false
					m.replayRequest = nil
					m.replaySavedAt = time.Time{}
					m.cfgCursor = 0
					m.screen = ScreenModelConfig
					return m, nil
				}
				switch item.action {
				case "recommend":
					m.selectedRecommendation = 0
					m.screen = ScreenRecommended
					return m, nil
				case "scan":
					return m.startComputerModelScan()
				case "latest":
					return m.openLatestLaunch()
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
				case "backend-forks":
					m.openBackendManager(ScreenMain)
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
		case "f", "F":
			m.openBackendManager(ScreenMain)
		case "c", "C":
			if item, ok := m.mainList.SelectedItem().(mainItem); ok && item.isModel {
				m.selectedModel = item.index
				m.backendRouteBypass = false
				m.replayRequest = nil
				m.replaySavedAt = time.Time{}
				m.cfgCursor = 0
				m.screen = ScreenModelConfig
				return m, nil
			}
		case "x", "X":
			if item, ok := m.mainList.SelectedItem().(mainItem); ok && item.isModel {
				m.openRemoveModelChoice(item.index)
				return m, nil
			}
		}
	}

	var cmd tea.Cmd
	m.mainList, cmd = m.mainList.Update(msg)
	return m, cmd
}

func (m Model) startComputerModelScan() (tea.Model, tea.Cmd) {
	if m.scanningModels {
		m.message = "Computer model scan is already running"
		m.messageType = "info"
		return m, nil
	}
	m.scanningModels = true
	m.message = "Scanning local disks for GGUF models… You can keep using the TUI."
	m.messageType = "info"
	return m, scanComputerModels(m.modelDir, m.cacheDir)
}

// openLatestLaunch loads the last request emitted by the TUI, restores its
// presentation fields, and opens the normal pre-launch screen. Enter then
// returns the saved request byte-for-byte at the field level; Esc converts it
// back into an editable configuration instead of launching anything.
func (m Model) openLatestLaunch() (tea.Model, tea.Cmd) {
	req, savedAt, err := LoadLatestLaunch(m.cacheDir)
	if err != nil {
		if errors.Is(err, ErrNoLatestLaunch) {
			m.message = "No previous TUI launch configuration has been saved yet"
		} else {
			m.message = fmt.Sprintf("Could not load latest TUI configuration: %v", err)
		}
		m.messageType = "warning"
		return m, nil
	}
	if info, err := os.Stat(req.ModelPath); err != nil || info.IsDir() {
		m.message = fmt.Sprintf("Latest configuration is unavailable because its model is missing: %s", req.ModelPath)
		m.messageType = "warning"
		return m, nil
	}
	if req.Backend == "" {
		m.backend = "auto"
	} else {
		m.backend = req.Backend
	}

	if !m.selectModelPath(req.ModelPath) {
		m.message = fmt.Sprintf("Latest configuration model is not a runnable GGUF: %s", req.ModelPath)
		m.messageType = "warning"
		return m, nil
	}

	m.applyLaunchRequestFields(req)
	m.backendRouteBypass = false
	copyReq := *req
	copyReq.BackendArgs = append([]string(nil), req.BackendArgs...)
	m.replayRequest = &copyReq
	m.replaySavedAt = savedAt
	m.message = ""
	m.messageType = ""
	m.screen = ScreenPrelaunch
	return m, nil
}

// selectModelPath attaches a request to an already recognized model or adds
// that one existing GGUF without requiring another whole-computer scan.
func (m *Model) selectModelPath(path string) bool {
	want := modelIdentity(path)
	selected := -1
	for i := range m.models {
		if modelIdentity(m.models[i].Path) == want {
			selected = i
			break
		}
	}
	if selected < 0 {
		items := mergeModelItems(m.models, discoverModelsFromPaths([]string{path}))
		m.models = enrichModelItems(items, m.cacheDir, m.backend, m.caps)
		m.rebuildMainList()
		for i := range m.models {
			if modelIdentity(m.models[i].Path) == want {
				selected = i
				break
			}
		}
	}
	if selected < 0 {
		return false
	}
	m.selectedModel = selected
	return true
}

func (m *Model) applyLaunchRequestFields(req *LaunchRequest) {
	if req == nil {
		return
	}
	m.port = req.Port
	m.ctxMode, m.ctxSize = "fit", "fit"
	ctxFlag := strings.TrimSpace(req.CtxFlag)
	switch ctxFlag {
	case "", "fit", "auto":
	case "max", "native":
		m.ctxMode, m.ctxSize = "max", "max"
	default:
		m.ctxMode, m.ctxSize = "manual", ctxFlag
	}
	if ctxFlag == "" && req.CtxSize > 0 {
		m.ctxMode, m.ctxSize = "manual", strconv.Itoa(req.CtxSize)
	}
	if req.KVPlacement != "" {
		m.kvPlacement = req.KVPlacement
	}
	if req.KVQuality != "" {
		m.kvQuality = req.KVQuality
	}
	m.swaFull = req.SWAFull
	m.parallel, m.parallelSet = "", req.ParallelSet
	if req.ParallelSet && req.Parallel > 0 {
		m.parallel = strconv.Itoa(req.Parallel)
	}
	m.vision = req.Vision
	m.tunePath = req.TuneCache
	m.aitune = req.AITune
	m.aituneRounds = req.AITuneRounds
	m.benchmark = req.Benchmark
	m.claudeCode = req.ClaudeCode
	m.claudeProfile = req.ClaudeProfile
	m.claudeReviewer = req.ClaudeReviewerOverride
	if req.SupportSet || req.SupportExpert != "" {
		m.supportExpert = req.SupportExpert
		m.supportOnline = req.SupportOnline
	}
	m.supportSet = req.SupportSet
	m.noCachedConfig = req.NoCachedConfig
	m.chatTemplate = req.ChatTemplate
	m.resumeSession, m.resumeRun, m.resumeCached = req.ResumeSession, "", 0
	m.refreshTunedCounts()
}

// cfgRows returns the ordered focusable rows of the Advanced config screen.
func (m Model) cfgRows() []string {
	rows := []string{}
	if m.selectedBackendRecipe() != nil {
		rows = append(rows, "backend-install")
	}
	rows = append(rows, "context", "parallel", "kv", "kvq", "swa", "tuned", "aitune")
	if m.aitune {
		rows = append(rows, "rounds")
	}
	rows = append(rows, "vision", "claudecode")
	if m.claudeCode {
		rows = append(rows, "claudereviewer", "claudeprofile")
	}
	rows = append(rows, "chattemplate")
	return append(rows, "benchmark", "launch", "dryrun")
}

func (m Model) selectedBackendRecipe() *backends.Recipe {
	if m.backendRouteBypass || m.selectedModel < 0 || m.selectedModel >= len(m.models) {
		return nil
	}
	model := m.models[m.selectedModel]
	if model.AutoBackend != "" || model.BackendRecipe == "" {
		return nil
	}
	return backends.RecipeByName(model.BackendRecipe)
}

// effectiveBackend returns the model-specific installed route when one exists;
// otherwise it preserves the configured default backend. This keeps a Laguna,
// Hy3, or MiniMax fork scoped to the model that needs it instead of changing the
// user's global backend setting for every model.
func (m Model) effectiveBackend() string {
	if m.backendRouteBypass {
		if fallback := strings.TrimSpace(m.backendRouteBypassBackend); fallback != "" {
			return fallback
		}
	}
	if m.selectedModel >= 0 && m.selectedModel < len(m.models) {
		if routed := strings.TrimSpace(m.models[m.selectedModel].AutoBackend); routed != "" {
			return routed
		}
	}
	return m.backend
}

// requestBackend is the backend a launch asks for. With the default "auto"
// setting a model's registered route is exactly what automatic selection picks,
// so it stays a display value: sending it as --backend made the default TUI
// launch an explicit backend choice, which the CLI default is not, and that
// switches off route fallback, the reviewed-recipe and fork offers and the
// CUDA-build offer. A configured global backend would override the route in
// cmdLaunch, so there the route is still sent to keep a fork scoped to the
// model that needs it.
func (m Model) requestBackend() string {
	if m.backendRouteBypass {
		if fallback := strings.TrimSpace(m.backendRouteBypassBackend); fallback != "" {
			return fallback
		}
	}
	configured := strings.TrimSpace(m.backend)
	if configured == "" || strings.EqualFold(configured, "auto") {
		return configured
	}
	return m.effectiveBackend()
}

// openSelectedBackendInstall asks before the network clone/build. Confirming
// returns a normal backend CLI request carrying the model's current launch
// settings; cmdGUI can then reopen the same model at pre-launch with the newly
// registered route selected.
func (m *Model) openSelectedBackendInstall() bool {
	recipe := m.selectedBackendRecipe()
	if recipe == nil || m.selectedModel < 0 || m.selectedModel >= len(m.models) {
		return false
	}
	model := m.models[m.selectedModel]
	arch := model.Architecture
	if arch == "" {
		arch = model.Arch
	}
	installLabel := "Install " + recipe.Name + " and select it"
	continueBackend := strings.TrimSpace(m.backend)
	if required := backends.RequiredBackendForArch(arch); required != "" {
		continueBackend = required
	}
	if continueBackend == "" {
		continueBackend = "auto"
	}
	continueLabel := "Continue once with " + continueBackend
	m.openChoice(
		"Backend for "+arch,
		[]string{"Cancel", installLabel, continueLabel},
		"Cancel",
		ScreenModelConfig,
		func(mm *Model, value string) {
			switch value {
			case installLabel:
				req := mm.buildLaunchRequest()
				if req == nil {
					return
				}
				req.Backend = recipe.Tag
				req.BackendArgs = []string{"install", recipe.Name}
				mm.launchRequest = req
			case continueLabel:
				mm.backendRouteBypass = true
				mm.backendRouteBypassBackend = continueBackend
				mm.message = "Continuing once with " + continueBackend + "; the model may fail if that backend lacks " + arch
				mm.messageType = "warning"
				mm.loadResumableSession()
				mm.screen = ScreenPrelaunch
			}
		},
	)
	return true
}

func (m *Model) openCfgInput(mode, val, placeholder string) {
	m.inputMode = mode
	m.input.SetValue(val)
	m.input.Placeholder = placeholder
	m.input.Focus()
	// Entering free-text edit mode signals that a config value may follow; clear
	// the memo so the next render recomputes the Full-SWA KV / template rows.
	m.invalidateConfigMemo()
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
		n, err := strconv.Atoi(val)
		if err != nil || n < 1 {
			m.message = "Warning: Context must be fit, max, or a positive token count"
			m.messageType = "warning"
			return
		}
		m.ctxMode = "manual"
		m.ctxSize = strconv.Itoa(n)
	}
	// Context is an input to the memoized Full-SWA KV estimate and Chat-template
	// label, so a change forces the next render to recompute them.
	m.invalidateConfigMemo()
}

// cycleCfgRow changes the focused Advanced-config row with ←/→ (dir -1/+1).
// kvQualityOptions is the canonical KV-quality enum shared by the Settings
// screen and the Advanced-config cycle, so one vocabulary exists everywhere.
// "low" is an alias for q4_0 and "high" for f16; presenting both would make
// cycling visually do nothing, so the aliases are dropped and every option is
// one distinct type.
func kvQualityOptions() []string {
	return []string{"auto", "high", "bf16", "mid", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"}
}

// kvQualityLabel renders one KV-quality value with a friendly hint on both the
// Settings screen and the Advanced-config cycle, so the two screens agree on
// the vocabulary. Legacy aliases (low, f16) are still recognized so a saved
// config value never renders raw or empty.
func kvQualityLabel(v string) string {
	labels := map[string]string{
		"auto": "auto", "high": "high (f16)", "mid": "mid (q8_0)", "low": "low (q4_0)",
		"q4_1": "q4_1", "iq4_nl": "iq4_nl", "q5_0": "q5_0", "q5_1": "q5_1",
		"bf16": "bf16", "f16": "f16", "f32": "f32", "q4_0": "q4_0", "q8_0": "q8_0",
	}
	if l, ok := labels[v]; ok {
		return l
	}
	return v
}

func (m *Model) cycleCfgRow(row string, dir int) {
	// Rowing mutates one of the memoized Advanced-config inputs (kvQuality,
	// swaFull, context, claudeCode, chatTemplate, ...), so the next render must
	// recompute the Full-SWA KV estimate and Chat-template label rather than
	// reuse the now-stale memo.
	m.invalidateConfigMemo()
	switch row {
	case "kv":
		order := []string{"auto", "gpu", "cpu"}
		if dir < 0 {
			m.kvPlacement = prevOption(order, m.kvPlacement)
		} else {
			m.kvPlacement = nextOption(order, m.kvPlacement)
		}
	case "kvq":
		// Same ordering as the Settings screen's "KV quality" enum, so arrow
		// cycling and the saved setting agree.
		order := kvQualityOptions()
		if dir < 0 {
			m.kvQuality = prevOption(order, m.kvQuality)
		} else {
			m.kvQuality = nextOption(order, m.kvQuality)
		}
		// Cycling this row is a deliberate per-launch choice, so it goes out as
		// an explicit flag. Merely inheriting the saved setting does not: that
		// stays in config, where ggrun still applies it but may withdraw it on a
		// memory failure.
		m.kvQualityTouched = true
	case "swa":
		m.swaFull = !m.swaFull
		// Cycling this row is a deliberate per-launch choice, so it goes out as
		// an explicit flag. Merely inheriting the saved setting does not: that
		// stays in config, where ggrun still applies it but may withdraw it on a
		// memory failure.
		m.swaFullTouched = true
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
	case "claudereviewer":
		order := []string{"auto", "qwen", "qwen2b", "nanbeige", "off"}
		if dir < 0 {
			m.claudeReviewer = prevOption(order, m.claudeReviewer)
		} else {
			m.claudeReviewer = nextOption(order, m.claudeReviewer)
		}
	case "claudeprofile":
		profiles := []string{"", "agent-interactive", "agent-parallel"}
		if dir < 0 {
			m.claudeProfile = prevOption(profiles, m.claudeProfile)
		} else {
			m.claudeProfile = nextOption(profiles, m.claudeProfile)
		}
	case "chattemplate":
		opts := chatTemplateOrdered()
		cur := m.chatTemplate
		if cur == "" {
			cur = "auto"
		}
		if dir < 0 {
			m.chatTemplate = prevOption(opts, cur)
		} else {
			m.chatTemplate = nextOption(opts, cur)
		}
		if m.chatTemplate == "auto" {
			m.chatTemplate = ""
		}
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
	case "backend-install":
		m.openSelectedBackendInstall()
	case "context":
		m.openCfgInput("ctx", m.ctxSize, "fit, max, or token count")
	case "parallel":
		m.openCfgInput("parallel", m.parallel, "Parallel slots (blank = let placement decide)")
	case "kv":
		m.cycleCfgRow("kv", 1)
	case "kvq":
		m.cycleCfgRow("kvq", 1)
	case "swa":
		m.cycleCfgRow("swa", 1)
	case "tuned":
		m.openTunedPicker()
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
	case "claudereviewer":
		m.cycleCfgRow("claudereviewer", 1)
	case "claudeprofile":
		m.cycleCfgRow("claudeprofile", 1)
	case "chattemplate":
		m.cycleCfgRow("chattemplate", 1)
	case "benchmark":
		m.benchmark = !m.benchmark
		if m.benchmark {
			m.aitune = false
		}
	case "launch":
		if m.openSelectedBackendInstall() {
			return m, nil
		}
		m.replayRequest = nil
		m.replaySavedAt = time.Time{}
		m.loadResumableSession()
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
				val = strings.TrimSpace(val)
				if val == "" {
					m.parallel = ""
					m.parallelSet = false
				} else if n, err := strconv.Atoi(val); err == nil && n > 0 {
					m.parallel = strconv.Itoa(n)
					m.parallelSet = true
				} else {
					m.message = "Warning: Parallel must be a positive integer"
					m.messageType = "warning"
				}
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
	case "k":
		m.cycleCfgRow("kvq", 1)
	case "w", "W":
		m.cycleCfgRow("swa", 1)
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
		if m.openSelectedBackendInstall() {
			return m, nil
		}
		m.replayRequest = nil
		m.replaySavedAt = time.Time{}
		m.loadResumableSession()
		m.screen = ScreenPrelaunch
	case "d", "D":
		m.message = fmt.Sprintf("Dry run: %s", strings.Join(m.buildArgs(), " "))
		m.messageType = "info"
	case "i", "I":
		m.openSelectedBackendInstall()
	case "t", "T":
		m.openTunedPicker()
	case "e", "E":
		if m.claudeCode {
			m.cycleCfgRow("claudereviewer", 1)
		}
	case "z", "Z":
		if m.claudeCode {
			m.cycleCfgRow("claudeprofile", 1)
		}
	case "n", "N":
		m.cycleCfgRow("chattemplate", 1)
	case "y", "Y":
		m.openClearCachesChoice(m.selectedModel)
	case "g", "G":
		m.noCachedConfig = !m.noCachedConfig
	case "q", "Q":
		return m, tea.Quit
	}
	return m, nil
}

// promptDownloadDir asks where a chosen download should land, pre-filled with
// the configured model directory so Enter keeps the existing behaviour.
func (m Model) promptDownloadDir(req *LaunchRequest) (tea.Model, tea.Cmd) {
	m.pendingDownload = req
	m.screen = ScreenDownload
	m.inputMode = "downloaddir"
	m.input.SetValue(m.modelDir)
	m.input.Placeholder = "Destination directory"
	m.input.Focus()
	m.message = fmt.Sprintf("Destination for %s (Enter keeps %s)", req.DownloadRepo, m.modelDir)
	m.messageType = "info"
	return m, nil
}

func firstRunActions() []string {
	return []string{"recommend", "latest", "scan", "download", "modeldir", "backends", "update", "quit"}
}

func (m Model) doFirstRunAction(action string) (tea.Model, tea.Cmd) {
	switch action {
	case "recommend":
		m.selectedRecommendation = 0
		m.screen = ScreenRecommended
	case "latest":
		return m.openLatestLaunch()
	case "scan":
		return m.startComputerModelScan()
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
	case "backends":
		m.openBackendManager(ScreenFirstRun)
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
	case "l", "L":
		return m.doFirstRunAction("latest")
	case "p", "P":
		return m.doFirstRunAction("scan")
	case "d", "D":
		return m.doFirstRunAction("download")
	case "m", "M":
		return m.doFirstRunAction("modeldir")
	case "f", "F":
		return m.doFirstRunAction("backends")
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
		case "external-remove-confirm":
			idx := m.pendingExternalRemoveIdx
			m.inputMode = ""
			m.input.Blur()
			m.screen = ScreenMain
			if strings.TrimSpace(val) != "yes" {
				m.message = "External model removal cancelled."
				m.messageType = ""
				return m, cmd
			}
			m.removeModelAt(idx)
			return m, cmd
		case "download":
			val = strings.TrimSpace(val)
			if val != "" {
				return m.promptDownloadDir(&LaunchRequest{DownloadRepo: val})
			}
			m.message = "Warning: Enter a Hugging Face GGUF repository"
			m.messageType = "warning"
		case "downloaddir":
			req := m.pendingDownload
			if req == nil {
				m.inputMode = ""
				m.screen = ScreenFirstRun
				return m, nil
			}
			if dir := strings.TrimSpace(val); dir != "" && dir != m.modelDir {
				req.DownloadDir = dir
			}
			m.launchRequest = req
			return m, tea.Quit
		case "modeldir":
			val = strings.TrimSpace(val)
			if val != "" {
				m.modelDir = val
				m.models = loadRecognizedModels(m.modelDir, m.cacheDir, m.backend, m.caps)
				m.rebuildMainList()
				if err := persistConfig(func(c *config.Config) { c.ModelDir = val }); err != nil {
					m.message = fmt.Sprintf("Warning: Using %s for this session — could not save config: %v", val, err)
					m.messageType = "warning"
				} else {
					m.message = fmt.Sprintf("Model directory saved: %s (%d models)", val, len(m.models))
					m.messageType = "info"
				}
			}
		case "backend-add":
			fields := strings.Fields(val)
			if len(fields) == 0 {
				m.message = "Warning: Enter a Git repository URL"
				m.messageType = "warning"
				return m, cmd
			}
			args := []string{"add", fields[0]}
			if len(fields) > 1 {
				args = append(args, "--tag", fields[1])
			}
			if len(fields) > 2 {
				args = append(args, "--route-arch", fields[2])
			}
			m.launchRequest = &LaunchRequest{BackendArgs: args}
			return m, tea.Quit
		case "backend-register":
			fields := strings.Fields(val)
			if len(fields) < 2 {
				m.message = "Warning: Enter a tag and llama-server binary path"
				m.messageType = "warning"
				return m, cmd
			}
			args := []string{"register", "--tag", fields[0], "--path", fields[1]}
			if len(fields) > 2 {
				args = append(args, "--route-arch", fields[2])
			}
			m.launchRequest = &LaunchRequest{BackendArgs: args}
			return m, tea.Quit
		}
		m.inputMode = ""
		m.screen = ScreenMain
	}
	return m, cmd
}

func (m Model) View() string {
	if m.screen == ScreenLoading {
		return m.viewLoading()
	}
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

func (m Model) viewLoading() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ ggrun ═══") + "\n\n")
	b.WriteString("  " + m.spinner.View() + "  " + subtitleStyle.Render("Starting up…") + "\n")
	b.WriteString(mutedStyle.Render("  Detecting hardware and reading GGUF headers") + "\n")
	b.WriteString(mutedStyle.Render("  This can take a few seconds with a large model directory") + "\n")
	body := b.String()
	if m.width > 0 && m.height > 0 {
		return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, body)
	}
	return body
}

func (m Model) viewMain() string {
	var b strings.Builder
	external := 0
	for _, model := range m.models {
		if model.External {
			external++
		}
	}

	b.WriteString(titleStyle.Render("═══ ggrun ═══") + "\n")
	b.WriteString(fmt.Sprintf("  Backend:  %s\n", m.backend))
	b.WriteString(fmt.Sprintf("  Hardware: %s\n", hwSummary(m.caps)))
	b.WriteString(fmt.Sprintf("  Models:   %d recognized (%d elsewhere)\n", len(m.models), external))
	b.WriteString(fmt.Sprintf("  Primary:  %s\n", m.modelDir))
	b.WriteString(fmt.Sprintf("  Settings: %s\n", m.settingsPath))
	b.WriteString("\n")

	if len(m.models) == 0 {
		b.WriteString("  (no GGUF models found)\n")
	}

	b.WriteString(m.mainList.View())

	b.WriteString("\n")
	if m.modelUsage != nil && hasAnyUsage(m.modelUsage) {
		b.WriteString(mutedStyle.Render("  sorted by usage: most-used and most-recent first") + "\n")
	}
	b.WriteString(mutedStyle.Render("  Enter configure · / or type to filter · x delete"))

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
		"latest":    "[l] Run latest saved configuration",
		"scan":      "[p] Scan all local disks for existing GGUF models",
		"download":  "[d] Manual Hugging Face repository",
		"modeldir":  "[m] Point at an existing model directory",
		"backends":  "[f] Install or manage backend forks",
		"update":    "[u] Update ggrun and backends",
		"quit":      "[q] Quit",
	}
	for i, a := range actions {
		if i == m.menuCursor {
			b.WriteString(selectedStyle.Render("> "+labels[a]) + "\n")
		} else {
			b.WriteString("  " + labels[a] + "\n")
		}
	}
	if m.message != "" {
		b.WriteString("\n  ")
		if m.messageType == "warning" {
			b.WriteString(warningStyle.Render(m.message))
		} else if m.messageType == "error" {
			b.WriteString(errorStyle.Render(m.message))
		} else {
			b.WriteString(highlightStyle.Render(m.message))
		}
		b.WriteString("\n")
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
			b.WriteString(selectedStyle.Render(fmt.Sprintf("  > %-26s %s", label, value)) + "\n")
		} else {
			b.WriteString(fmt.Sprintf("    %-26s ", label) + subtitleStyle.Render(value) + "\n")
		}
	}
	// statline renders a read-only row that is NOT focusable (no arrow key row,
	// no Enter action): the action is only reachable by its letter hotkey. The
	// muted render signals "this is state, not a menu row" so a user does not
	// arrow to it expecting a cursor.
	statline := func(label, value string) {
		b.WriteString(mutedStyle.Render(fmt.Sprintf("    %-26s ", label)) + subtitleStyle.Render(value) + "\n")
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
		parallelLabel = "automatic (1 normally; up to 4 for Claude Code)"
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

	kvQualityLabel := kvQualityLabel(m.kvQuality)

	section("Backend, context & memory")
	switch {
	case model.AutoBackend != "":
		statline("Backend", model.AutoBackend+" (auto-selected for "+model.Architecture+")")
	case m.selectedBackendRecipe() != nil:
		line("backend-install", "[i] Backend", "install "+model.BackendRecipe+" for "+model.Architecture)
	case m.backendRouteBypass && model.BackendRecipe != "":
		statline("Backend", m.effectiveBackend()+" (unsupported-route check bypassed once)")
	default:
		statline("Backend", m.backend)
	}
	line("context", "[c] Context size", ctxLabel)
	line("parallel", "[p] Parallel slots", parallelLabel)
	line("kv", "[K] KV placement", kvLabel)
	line("kvq", "[k] KV quality", kvQualityLabel)
	swaLabel := m.swaLabel(model)
	line("swa", "[w] Full SWA cache", swaLabel)

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
	if m.claudeCode {
		line("claudereviewer", "[e] Reviewer/worker", claudeReviewerLabel(m.claudeReviewer))
		line("claudeprofile", "[z] Claude profile", claudeProfileLabel(m.claudeProfile))
	}
	line("chattemplate", "[n] Chat template", m.chatTemplateLabel(model))
	line("benchmark", "[b] Benchmark mode", boolLabel(m.benchmark))

	section("Launch optimizer")
	optLabel := placement.FormatCalibrationStatus(nil)
	if d, err := placement.LatestCalibrationDecisionForModel(m.cacheDir, model.Name); err == nil {
		optLabel = placement.FormatCalibrationStatus(d)
	}
	statline("Measured config", optLabel)
	statline("Inspect", "ggrun status "+model.Name)

	section("Cache & launch")
	nocacheLabel := "off (reuse cached placement/probes)"
	if m.noCachedConfig {
		nocacheLabel = "on (derive fresh, ignore cached config)"
	}
	statline("[g] Launch without cached config", nocacheLabel)
	statline("[y] Clear caches", "drop cached placement/calibration for this model (keep GGUF)")
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

	// Keep the focused row visible on short terminals: clip the body but add a
	// hint that more rows exist above/below.
	if body, clipped := clipToFocused(b.String(), m.viewportLines()); clipped {
		return body + "\n" + mutedStyle.Render("  (use ↑/↓ — more rows above/below)")
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
			if m.replayRequest != nil {
				req := *m.replayRequest
				req.BackendArgs = append([]string(nil), m.replayRequest.BackendArgs...)
				m.launchRequest = &req
			} else {
				m.launchRequest = m.buildLaunchRequest()
			}
			return m, tea.Quit
		case "c", "C":
			// Apply the suggested context from the KV hint: one keypress turns the
			// recommendation into the actual launch context.
			if model := m.currentModel(); model != nil {
				if hint := m.denseKVHint(*model); hint != nil && hint.suggestCtx > 0 {
					m.setCtx(strconv.Itoa(hint.suggestCtx))
					m.message = fmt.Sprintf("Context set to %d — review, then Enter to launch", hint.suggestCtx)
					m.messageType = "info"
				}
			}
			return m, nil
		case "r", "R":
			// Resume only makes sense in Claude Code mode with a recorded
			// session; otherwise fall through to no-op rather than launching
			// something the footer did not offer.
			if m.claudeCode && m.resumeSession != "" {
				req := m.buildLaunchRequest()
				if req != nil {
					req.ResumeSession = m.resumeSession
					m.launchRequest = req
					return m, tea.Quit
				}
			}
			return m, nil
		case "esc":
			m.replayRequest = nil
			m.replaySavedAt = time.Time{}
			m.screen = ScreenModelConfig
			return m, nil
		case "q", "Q":
			return m, tea.Quit
		}
	}
	return m, nil
}

// currentModel returns the selected ModelItem, or nil when out of range.
func (m Model) currentModel() *ModelItem {
	if m.selectedModel >= 0 && m.selectedModel < len(m.models) {
		return &m.models[m.selectedModel]
	}
	return nil
}

// verifiedConfigLine reports whether a previously verified serving config
// exists for this model, so the prelaunch screen can say the launch will start
// directly from the flags that already proved themselves. The scope key is
// strategy-free and computed inside placement.Compute, so the TUI uses a
// lightweight heuristic: any verified-config record whose basename matches this
// model is surfaced. A miss is silent — this is purely informational.
func (m Model) verifiedConfigLine(model ModelItem) string {
	if m.cacheDir == "" {
		return ""
	}
	dir := filepath.Join(m.cacheDir, "verified-configs")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	basename := filepath.Base(model.Path)
	latest := time.Time{}
	for _, ent := range entries {
		if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, ent.Name()))
		if err != nil {
			continue
		}
		var vc placement.VerifiedConfig
		if json.Unmarshal(data, &vc) != nil || vc.ModelBasename == "" {
			continue
		}
		// The record's basename is filepath.Base(Path); match it directly and
		// via the base without the .gguf extension (writeTemplateFile-style
		// tolerance, mirroring ClearModelCaches' matcher).
		if vc.ModelBasename != basename && vc.ModelBasename != strings.TrimSuffix(basename, filepath.Ext(basename)) {
			continue
		}
		if t, err := time.Parse(time.RFC3339, vc.MeasuredAt); err == nil && t.After(latest) {
			latest = t
		}
	}
	if latest.IsZero() {
		return ""
	}
	return mutedStyle.Render("  verified config from " + latest.Local().Format("2006-01-02 15:04") + " may be reused (clear caches to force re-measure)")
}

// loadResumableSession looks for a Claude Code session recorded from this
// working directory, so the pre-launch screen can offer to continue it instead
// of starting a conversation and workflow from zero.
func (m *Model) loadResumableSession() {
	m.resumeSession, m.resumeRun, m.resumeCached = "", "", 0
	workDir, err := os.Getwd()
	if err != nil {
		return
	}
	rec, err := claudesession.Latest(m.cacheDir, workDir)
	if err != nil {
		return
	}
	m.resumeSession = rec.SessionID
	if wf, cached := claudesession.LatestRun(claudeProjectsDir(), rec.WorkDir, rec.SessionID); wf != nil {
		m.resumeRun, m.resumeCached = wf.RunID, cached
	}
}

// claudeProjectsDir mirrors the CLI's lookup of Claude Code's project state.
func claudeProjectsDir() string {
	if dir := os.Getenv("CLAUDE_CONFIG_DIR"); dir != "" {
		return filepath.Join(dir, "projects")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".claude", "projects")
}

// shortSessionID keeps the pre-launch line readable; the full ID is only needed
// by the resume itself.
func shortSessionID(id string) string {
	if len(id) <= 13 {
		return id
	}
	return id[:13] + "…"
}

func (m Model) viewPrelaunch() string {
	if len(m.models) == 0 {
		return "No model selected"
	}
	model := m.models[m.selectedModel]
	kvHint := m.denseKVHint(model)
	var b strings.Builder
	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ Pre-launch: %s ═══", model.Name)) + "\n\n")
	if m.replayRequest != nil {
		when := m.replaySavedAt.Local().Format("2006-01-02 15:04:05 MST")
		b.WriteString(highlightStyle.Render("  Latest saved TUI configuration") + "\n")
		b.WriteString(fmt.Sprintf("  Saved:          %s\n", when))
		b.WriteString("  Enter replays the exact saved request; Esc makes it editable.\n\n")
	}

	ctx := m.ctxSize
	if m.ctxMode == "fit" {
		ctx = "fit"
	}
	if model.FitCtx > 0 {
		ctx += fmt.Sprintf("  (fits ~%d)", model.FitCtx)
	}
	if model.MaxCtx > 0 {
		ctx += fmt.Sprintf("  (train max %d)", model.MaxCtx)
	}
	prelaunchBackend := m.effectiveBackend()
	if m.replayRequest != nil && m.replayRequest.Backend != "" {
		prelaunchBackend = m.replayRequest.Backend
	}

	b.WriteString("  Model:          " + model.Path + "\n")
	b.WriteString(fmt.Sprintf("  Backend:        %s\n", prelaunchBackend))
	b.WriteString(fmt.Sprintf("  Context:        %s\n", ctx))
	if m.port > 0 {
		b.WriteString(fmt.Sprintf("  Port:           %d\n", m.port))
	}
	b.WriteString(fmt.Sprintf("  Parallel:       %s\n", m.prelaunchParallelLabel()))
	b.WriteString(fmt.Sprintf("  KV placement:   %s\n", m.kvPlacement))
	b.WriteString(fmt.Sprintf("  KV quality:     %s\n", kvQualityLabel(m.kvQuality)))
	b.WriteString(fmt.Sprintf("  Full SWA cache: %s\n", m.swaLabel(model)))
	if m.aitune {
		b.WriteString(fmt.Sprintf("  AI tune:        %s (%d rounds)\n", boolLabel(m.aitune), m.aituneRounds))
	} else {
		b.WriteString(fmt.Sprintf("  AI tune:        %s\n", boolLabel(m.aitune)))
	}
	b.WriteString(fmt.Sprintf("  Vision:         %s\n", boolLabel(m.vision)))
	b.WriteString(fmt.Sprintf("  Benchmark:      %s\n", boolLabel(m.benchmark)))
	if m.chatTemplate != "" {
		b.WriteString(fmt.Sprintf("  Chat template:  forced %s\n", m.chatTemplate))
	} else if entry, ok := m.autoChatTemplate(model); ok {
		b.WriteString(fmt.Sprintf("  Chat template:  auto (%s corrected)\n", entry.Name))
	}
	if m.claudeCode {
		b.WriteString(fmt.Sprintf("  Claude Code:    %s\n", boolLabel(m.claudeCode)))
		b.WriteString(fmt.Sprintf("  Reviewer/worker: %s\n", claudeReviewerLabel(m.claudeReviewer)))
		b.WriteString(fmt.Sprintf("  Claude profile: %s\n", claudeProfileLabel(m.claudeProfile)))
	}
	if m.tunePath != "" {
		b.WriteString(fmt.Sprintf("  Tuned config:   %s\n", filepath.Base(m.tunePath)))
	}
	if m.noCachedConfig {
		b.WriteString("  Cached config:  ignored (fresh derive)\n")
	}
	if m.claudeCode && m.resumeSession != "" {
		b.WriteString(fmt.Sprintf("  Resumable:      session %s\n", shortSessionID(m.resumeSession)))
		if m.resumeRun != "" {
			b.WriteString(fmt.Sprintf("                  workflow %s, %d agents cached\n", m.resumeRun, m.resumeCached))
		}
	}
	// Support policy lines are only shown when they deviate from the defaults;
	// a user cannot change them on this screen and the defaults are noise.
	if m.supportExpert != "" && m.supportExpert != "auto" {
		b.WriteString(fmt.Sprintf("  Support expert: %s\n", supportExpertLabel(m.supportExpert)))
	}
	if m.supportOnline {
		b.WriteString("  Online research: on\n")
	}
	if vc := m.verifiedConfigLine(model); vc != "" {
		b.WriteString("  " + vc + "\n")
	}

	if kvHint != nil {
		b.WriteString("\n" + warningStyle.Render("  "+strings.Join(kvHint.lines, "\n  ")) + "\n")
	}

	b.WriteString("\n")
	b.WriteString(highlightStyle.Render("  [Enter] Confirm and launch"))
	b.WriteString("\n")
	if m.claudeCode && m.resumeSession != "" {
		b.WriteString(highlightStyle.Render("  [r] Resume that session and its workflow"))
		b.WriteString("\n")
	}
	if kvHint != nil && kvHint.suggestCtx > 0 {
		b.WriteString(highlightStyle.Render(fmt.Sprintf("  [c] Use suggested ctx %d", kvHint.suggestCtx)))
		b.WriteString("\n")
	}
	b.WriteString("  [Esc] Back to config\n")
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
	// On short terminals keep the launch actions visible; the tail-first clip
	// guarantees [Enter]/[r]/[Esc] are never silently cut off.
	if body, clipped := clipToFocused(b.String(), m.viewportLines()); clipped {
		return body + "\n" + mutedStyle.Render("  (use ↑/↓ to review the full summary)")
	}
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
		case "q", "Q":
			return m, tea.Quit
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
			b.WriteString(selectedStyle.Render("> [0] Auto / heuristic cache selection") + "\n")
		} else {
			b.WriteString("  [0] Auto / heuristic cache selection\n")
		}
		for i, entry := range m.tunedConfigs {
			if i == m.tunedIndex {
				b.WriteString(selectedStyle.Render(fmt.Sprintf("> [%d] %s", i+1, entry.Label)) + "\n")
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
				return m.promptDownloadDir(&LaunchRequest{DownloadRepo: rec.Repo, DownloadQuant: rec.QuantName})
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
		case "q", "Q":
			return m, tea.Quit
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
		m.message = fmt.Sprintf("Warning: %s set for this session — save failed: %v", label, err)
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
	caps := detect.ApplyRAMLimitPercent(m.caps, m.ramLimitPercent)
	caps = detect.ApplyVRAMHeadroom(caps, m.vramHeadroomMB)
	caps = detect.ApplyRAMHeadroom(caps, m.ramHeadroomMB)
	m.recommendationGroups = recommend.TopCategories(caps, 4)
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

// clipToFocused clips a rendered screen to maxLines, keeping the focused row
// (the line whose trimmed text starts with the "> " cursor marker) visible. A
// config screen can run to ~30 lines while a 24-row terminal only shows ~20;
// without this the bottom is silently cut off and Launch becomes unreachable.
// When no focused marker exists (e.g. the prelaunch screen) it keeps the tail,
// so the launch actions stay visible. It returns (clipped, clippedFlag).
func clipToFocused(s string, maxLines int) (string, bool) {
	if maxLines <= 0 {
		return s, false
	}
	lines := strings.Split(s, "\n")
	if len(lines) <= maxLines {
		return s, false
	}
	focused := -1
	for i, l := range lines {
		if strings.HasPrefix(strings.TrimSpace(l), "> ") {
			focused = i
			break
		}
	}
	var start int
	if focused < 0 {
		start = len(lines) - maxLines
	} else {
		start = focused - maxLines/2
		if start < 0 {
			start = 0
		}
		if start+maxLines > len(lines) {
			start = len(lines) - maxLines
		}
	}
	return strings.Join(lines[start:start+maxLines], "\n"), true
}

// viewportLines returns how many rows a full-screen render may use. The TUI
// runs on an alternate screen where the full terminal height is available, but
// a 24-row terminal is common; rows below this clip are reachable with ↑/↓.
func (m Model) viewportLines() int {
	if m.height <= 0 {
		return 0 // unknown size: never clip, tests and pre-WindowSizeMsg renders
	}
	// Reserve a couple of rows of margin so the clip hint itself fits.
	if m.height <= 4 {
		return m.height
	}
	return m.height - 2
}

// wordWrap greedily packs words from s into lines no wider than width,
// breaking only at spaces. width <= 0 falls back to 78 columns.
func wordWrap(s string, width int) []string {
	if width <= 0 {
		width = 78
	}
	words := strings.Fields(s)
	if len(words) == 0 {
		return nil
	}
	lines := []string{words[0]}
	for _, w := range words[1:] {
		last := lines[len(lines)-1]
		if len(last)+1+len(w) > width {
			lines = append(lines, w)
		} else {
			lines[len(lines)-1] = last + " " + w
		}
	}
	return lines
}

func (m Model) viewRecommended() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("═══ Recommended Downloads ═══") + "\n")
	b.WriteString(fmt.Sprintf("  Hardware: %s\n", hwSummary(m.caps)))
	b.WriteString("  " + m.recommendedHeadroomControls() + "\n")
	for _, line := range wordWrap(recommend.CatalogAttribution(), m.width) {
		b.WriteString("  " + line + "\n")
	}
	b.WriteString("\n")

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
				prefix = selectedStyle.Render("> ")
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
			line := fmt.Sprintf("%-34s %-9s %-11s %5.1fG %3.0f%% %7s",
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
		vram = selectedStyle.Render("> " + vram + " ◂")
	}
	if m.recHeadroomFocus == "ram" {
		ram = selectedStyle.Render("> " + ram + " ◂")
	}
	return fmt.Sprintf("Reserve for other apps: %s   %s", vram, ram)
}

func (m Model) viewInputScreen() string {
	var b strings.Builder
	var title string
	switch m.inputMode {
	case "download":
		title = "Download Model"
	case "downloaddir":
		title = "Download Destination"
	case "modeldir":
		title = "Model Directory"
	case "backend-add":
		title = "Add llama.cpp Fork"
	case "backend-register":
		title = "Register Built Backend"
	case "external-remove-confirm":
		title = "Delete Outside the Primary Directory"
	default:
		title = "Input"
	}
	b.WriteString(titleStyle.Render(fmt.Sprintf("═══ %s ═══", title)) + "\n\n")
	b.WriteString(m.input.View())
	if m.inputMode == "external-remove-confirm" {
		b.WriteString("\n\n  Type exactly 'yes' to permanently delete this file, Esc to cancel")
	} else {
		b.WriteString("\n\n  Press Enter to confirm, Esc to cancel")
	}
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

func isAuxiliaryModel(name, arch string) bool {
	if strings.EqualFold(strings.TrimSpace(arch), "dflash") {
		return true
	}
	for _, token := range strings.FieldsFunc(strings.ToLower(filepath.Base(name)), func(r rune) bool {
		return (r < 'a' || r > 'z') && (r < '0' || r > '9')
	}) {
		switch token {
		case "dflash", "draft", "mtp", "speculator":
			return true
		}
	}
	return false
}

func discoverModels(dir string) []ModelItem {
	var items []ModelItem
	seen := make(map[string]bool)

	_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		item, key, ok := modelItemFromPath(path, info, false)
		if !ok || seen[key] {
			return nil
		}
		seen[key] = true
		items = append(items, item)
		return nil
	})

	return items
}

func modelItemFromPath(path string, info os.FileInfo, external bool) (ModelItem, string, bool) {
	if info == nil || info.IsDir() {
		return ModelItem{}, "", false
	}
	name := info.Name()
	lower := strings.ToLower(name)
	if !strings.HasSuffix(lower, ".gguf") || strings.Contains(lower, "mmproj") || isAuxiliaryModel(name, "") {
		return ModelItem{}, "", false
	}
	shardFiles, isMultiPart, shardErr := modelstore.ResolveGGUFShardFiles(path)
	if shardErr != nil {
		return ModelItem{}, "", false
	}

	baseName := name
	if shard := strings.Index(lower, "-00001-of-"); shard > 0 {
		baseName = name[:shard] + ".gguf"
	} else if strings.Contains(lower, "-of-") {
		return ModelItem{}, "", false
	}

	dirPath := filepath.Dir(path)
	modelKey := filepath.Join(dirPath, baseName)
	totalBytes := info.Size()
	if isMultiPart {
		totalBytes = 0
		for _, shardPath := range shardFiles {
			if st, err := os.Stat(shardPath); err == nil {
				totalBytes += st.Size()
			}
		}
	}

	arch := "dense"
	if strings.Contains(name, "A") && strings.Contains(name, "B") {
		// Check A[0-9]+B pattern for a cheap pre-header MoE hint.
		for i := 0; i < len(name)-1; i++ {
			if name[i] != 'A' && name[i] != 'a' {
				continue
			}
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
	return ModelItem{
		Name:     baseName,
		Path:     filepath.Clean(path),
		SizeGB:   float64(totalBytes) / (1024 * 1024 * 1024),
		Arch:     arch,
		External: external,
	}, modelKey, true
}

func discoverModelsFromPaths(paths []string) []ModelItem {
	items := make([]ModelItem, 0, len(paths))
	seen := make(map[string]bool)
	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			continue
		}
		item, key, ok := modelItemFromPath(path, info, true)
		if !ok || seen[key] {
			continue
		}
		seen[key] = true
		items = append(items, item)
	}
	return items
}

func modelIdentity(path string) string {
	path = filepath.Clean(path)
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		path = resolved
	}
	if runtime.GOOS == "windows" {
		path = strings.ToLower(path)
	}
	return path
}

// hasAnyUsage reports whether any usage record exists, so the footer legend
// ("sorted by usage") only appears when the sort actually used usage data.
func hasAnyUsage(usage map[string]modelusage.Record) bool {
	for _, r := range usage {
		if r.Launches > 0 {
			return true
		}
	}
	return false
}

// sortModels orders the list for display. When usage records are available the
// most-used and most-recently-used models float to the top; everything still
// falls back to a stable name sort so the list stays deterministic.
func sortModels(items []ModelItem, usage map[string]modelusage.Record) {
	sort.SliceStable(items, func(i, j int) bool {
		var left, right modelusage.Record
		if usage != nil {
			left = usage[modelIdentity(items[i].Path)]
			right = usage[modelIdentity(items[j].Path)]
		}
		if left.Launches != right.Launches {
			return left.Launches > right.Launches
		}
		if !left.LastUsedAt.Equal(right.LastUsedAt) {
			return left.LastUsedAt.After(right.LastUsedAt)
		}
		lname, rname := strings.ToLower(items[i].Name), strings.ToLower(items[j].Name)
		if lname != rname {
			return lname < rname
		}
		return strings.ToLower(items[i].Path) < strings.ToLower(items[j].Path)
	})
}

func mergeModelItems(groups ...[]ModelItem) []ModelItem {
	seen := make(map[string]bool)
	var merged []ModelItem
	for _, group := range groups {
		for _, item := range group {
			key := modelIdentity(item.Path)
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			merged = append(merged, item)
		}
	}
	sortModels(merged, nil)
	return merged
}

func enrichModelItems(models []ModelItem, cacheDir, backend string, caps *detect.Capabilities) []ModelItem {
	visible := models[:0]
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
			if isAuxiliaryModel(models[i].Name, info.Architecture) {
				continue
			}
			models[i].MaxCtx = info.ContextLength
			models[i].IsMoE = info.IsMoE
			models[i].Architecture = info.Architecture
			models[i].KVProfile = kvProfileFromGGUF(info)
			// Capture the embedded chat template once here, alongside KVProfile, so
			// the config screen's per-render chat-template row never re-opens the
			// .gguf and walks its metadata KV table (~700ms on a 17GB model).
			models[i].ChatTemplate = gguf.ChatTemplate(models[i].Path)
			if info.Architecture != "" {
				models[i].Arch = info.Architecture
			}
			if info.IsMoE {
				models[i].Arch += " · MoE"
			}
			if info.Architecture != "" {
				// Architecture routes are per-model and therefore more specific than
				// the configured default backend. An installed reviewed/custom fork
				// is selected for this model even when the user's default is pinned to
				// mainline or generic ik_llama.
				if routed := backends.ForArch(info.Architecture); routed != nil {
					modelBackendTag = routed.Tag
					models[i].AutoBackend = routed.Tag
				} else if helper := backends.SoleHelperForArch(info.Architecture); helper != nil {
					// The CLI routes an architecture served only by a helper-only
					// fork to that fork (selectBackendForModel). Mirror it, or an
					// installed helper recipe keeps being offered for install.
					modelBackendTag = helper.Tag
					models[i].AutoBackend = helper.Tag
				} else if recipes := backends.RecipesForArch(info.Architecture); len(recipes) > 0 {
					// Catalog order is preference order. For architectures with an
					// alternative recipe (Laguna's upstream target-only fork), the first
					// entry is the reviewed full-featured default.
					models[i].BackendRecipe = recipes[0].Name
				}
			}
			models[i].FitCtx = probe.EstimateFitCtxForInfo(models[i].Path, cacheDir, info, totalSysMemMB)
		}
		models[i].Tuned = tune.CountTunedConfigs(cacheDir, models[i].Name, modelBackendTag)
		visible = append(visible, models[i])
	}
	return visible
}

func kvProfileFromGGUF(info *gguf.Info) *placement.ModelProfile {
	if info == nil {
		return nil
	}
	return &placement.ModelProfile{
		NumLayers:        info.BlockCount,
		HeadCountKV:      info.HeadCountKV,
		KeyLength:        info.KeyLength,
		ValueLength:      info.ValueLength,
		KVLoraRank:       info.KVLoraRank,
		RopeDim:          info.NRot,
		HasSSM:           info.SSM,
		FullAttnInterval: info.FullAttnInterval,
		SlidingWindow:    info.SlidingWindow,
		ModelArch:        info.Architecture,
		// The KV rate/geometry cache (kvCachePath) is keyed on the model's exact
		// byte size. Without it, the profile built here produces kv_<basename>_0
		// which never matches the kv_<basename>_<actualsize> written at launch,
		// so the TUI "Clear caches" action silently leaves the stale KV rate.
		SizeBytes: info.NonExpertBytes + info.ExpertBytes,
	}
}

func (m Model) swaEstimateContext(model ModelItem) int {
	value := strings.TrimSpace(m.ctxSize)
	if n, err := strconv.Atoi(value); err == nil && n > 0 {
		return n
	}
	if value == "max" || value == "native" {
		return model.MaxCtx
	}
	if m.claudeCode {
		ctx := model.MaxCtx
		if ctx > 1048576 {
			ctx = 1048576
		}
		if ctx <= 0 {
			ctx = 131072
		}
		return ctx
	}
	return model.FitCtx
}

// configMemo caches the two expensive per-render Advanced-config computations:
// the Full-SWA KV extra-MiB estimate (swaExtraKVMB) and the Chat-template
// label (chatTemplateLabel). bubbletea calls View() after every Update, so on
// the model-config screen every keystroke — arrow-down row cycling and free-text
// typing included — used to re-run both on the single event-loop goroutine. The
// template re-opened the .gguf and walked its metadata KV table to read
// tokenizer.chat_template (~700ms per keystroke on a 17GB Qwen3.8-27B), and the
// KV row re-ran the full placement estimate. Their results depend only on
// (modelPath, ctxSize, kvQuality, swaFull, claudeCode), so they are memoized on
// this shared pointer (which survives the value copies of Model bubbletea makes
// between Update and View) and recomputed only when one of those inputs changes.
// The setters that mutate those inputs (setCtx, cycleCfgRow, openCfgInput) call
// invalidateConfigMemo to force the next render to recompute.
type configMemo struct {
	key      string // cfgMemoKeyFor of the currently cached values
	template string // rendered Chat-template label, see chatTemplateLabel
	kvMB     int    // Full-SWA KV extra MiB, see swaExtraKVMB
	kvMBSet  bool
}

// cfgMemoKeyFor returns the key identifying the model-config inputs the memoized
// values depend on. When unchanged, a render reads the cached values in O(1)
// instead of re-opening the .gguf or re-running the placement estimate.
func (m Model) cfgMemoKeyFor(model ModelItem) string {
	return model.Path + "\x00" + m.ctxSize + "\x00" + m.kvQuality +
		"\x00" + strconv.FormatBool(m.swaFull) + "\x00" + strconv.FormatBool(m.claudeCode)
}

// invalidateConfigMemo clears the memo so the next render recomputes it. Called
// by every setter that changes an input the memo depends on.
func (m *Model) invalidateConfigMemo() {
	if m.cfgMemo == nil {
		return
	}
	m.cfgMemo.key = ""
	m.cfgMemo.kvMBSet = false
}

// ensure recomputes the memoized values when cfgMemoKeyFor changed since the
// last computation. While the key is unchanged it is a no-op, so a render that
// did not change the config is O(1).
func (c *configMemo) ensure(m Model, model ModelItem) {
	key := m.cfgMemoKeyFor(model)
	if c.key == key {
		return
	}
	c.key = key

	// Chat-template label, mirroring chatTemplateLabel/autoChatTemplate. Uses the
	// template captured once at enrich time (model.ChatTemplate), never reopening
	// the .gguf here.
	if v := strings.TrimSpace(m.chatTemplate); v != "" {
		c.template = "forced: " + v
	} else {
		arch := model.Architecture
		if arch == "" {
			arch = model.Arch
		}
		basename := filepath.Base(model.Path)
		if entry, ok := chattemplate.Resolve(arch, basename, model.ChatTemplate, true); ok {
			c.template = "auto (" + entry.Name + ")"
		} else {
			c.template = "auto"
		}
	}

	// Full-SWA KV extra MiB.
	c.kvMB = swaExtraKVMBComputed(m, model)
	c.kvMBSet = true
}

// swaExtraKVMB returns the extra MiB a Full-SWA cache needs over a plain KV
// cache for this model at the current config, or -1 when it cannot be estimated.
// The result is memoized on the config memo; see configMemo. When the memo is
// not initialized (rare, e.g. a hand-built Model in tests), it computes directly.
func (m Model) swaExtraKVMB(model ModelItem) int {
	if m.cfgMemo == nil {
		return swaExtraKVMBComputed(m, model)
	}
	m.cfgMemo.ensure(m, model)
	return m.cfgMemo.kvMB
}

// swaExtraKVMBComputed is the uncached estimate used to populate the memo.
func swaExtraKVMBComputed(m Model, model ModelItem) int {
	if model.KVProfile == nil {
		return -1
	}
	ctx := m.swaEstimateContext(model)
	if ctx <= 0 {
		return -1
	}
	kvType, err := placement.NormalizeKVType(m.kvQuality)
	if err != nil {
		return -1
	}
	plain := placement.EstimateKVCacheMB(model.KVProfile, ctx, kvType, false)
	full := placement.EstimateKVCacheMB(model.KVProfile, ctx, kvType, true)
	if full < plain {
		return 0
	}
	return full - plain
}

func (m Model) swaLabel(model ModelItem) string {
	extraMB := m.swaExtraKVMB(model)
	if extraMB < 0 {
		if m.swaFull {
			return "on (more cache hits; memory estimate after dry run)"
		}
		return "off (smaller KV cache)"
	}
	if extraMB == 0 {
		if m.swaFull {
			return "on (no extra KV for this model)"
		}
		return "off (this model has no priced SWA delta)"
	}
	delta := fmt.Sprintf("+%.1f GiB KV", float64(extraMB)/1024.0)
	if m.swaFull {
		return "on (more cache hits; " + delta + ")"
	}
	return "off (enable cache hits; " + delta + ")"
}

// kvHint is the result of denseKVHint: the text to render on the prelaunch
// screen and, when the current context would spill off GPU, a suggested context
// the user can apply with one keypress ([c] on the prelaunch screen).
type kvHint struct {
	lines      []string
	suggestCtx int // >0: a rung that fits all-GPU; apply it with setCtx
}

// denseKVHint suggests useful context/KV steps for a DENSE model whose current
// context would push it into host offload: the 131k/64k/32k rungs with their KV
// sizes, plus the max-vs-fit tradeoff. It returns nil for MoE models, models
// with no KV geometry, or when the estimated context already fits entirely on
// GPU (the pre-launch Fit estimate already covers that case). It mirrors
// placement's denseMultiGPUFit boundary: model + (CUDA overhead + compute
// floor) per GPU + KV <= free VRAM.
func (m Model) denseKVHint(model ModelItem) *kvHint {
	if model.IsMoE || model.KVProfile == nil || m.caps == nil || len(m.caps.GPUs) == 0 {
		return nil
	}
	kvType, err := placement.NormalizeKVType(m.kvQuality)
	if err != nil {
		kvType = "q8_0"
	}
	baseMB := model.KVProfile.TotalSizeMB
	if baseMB <= 0 && model.KVProfile.SizeBytes > 0 {
		baseMB = int(model.KVProfile.SizeBytes / 1048576)
	}
	if baseMB <= 0 {
		return nil
	}

	numGPUs := len(m.caps.GPUs)
	totalFree := 0
	for _, g := range m.caps.GPUs {
		totalFree += g.VRAMFreeMB()
	}
	// computeFloorMB (1024 MiB per GPU) mirrors placement's cited llama.cpp
	// compute-graph floor; kept as a literal here because the constant is not
	// exported. It is deliberately NOT the measured compute-buffer figure — a
	// hint should be conservative so it suggests a rung that really fits.
	overheadMB := placement.SystemCUDAOverheadMB(m.cacheDir, m.caps.GPUs) + 1024*numGPUs

	kvAt := func(ctx int) int {
		if ctx <= 0 {
			return 0
		}
		return placement.EstimateKVCacheMB(model.KVProfile, ctx, kvType, m.swaFull)
	}
	fits := func(ctx int) bool {
		if ctx <= 0 {
			return false
		}
		return baseMB+overheadMB+kvAt(ctx) <= totalFree
	}

	cur := m.swaEstimateContext(model)
	if cur <= 0 {
		cur = model.FitCtx
	}
	if cur <= 0 || fits(cur) {
		return nil
	}

	rungs := placement.DenseContextRungs(cur + 1)
	rungParts := make([]string, 0, 3)
	bestFit := 0
	for _, r := range rungs {
		if len(rungParts) >= 3 {
			break
		}
		rungParts = append(rungParts, fmt.Sprintf("ctx %d → ~%s KV", r, gibString(kvAt(r))))
		if fits(r) && bestFit == 0 {
			bestFit = r
		}
	}
	hint := &kvHint{}
	if bestFit > 0 {
		deficit := baseMB + overheadMB + kvAt(cur) - totalFree
		if deficit < 0 {
			deficit = 0
		}
		hint.lines = []string{
			fmt.Sprintf("ctx %d would spill ~%s to system RAM", cur, gibString(deficit)),
			"try " + strings.Join(rungParts, " · "),
			fmt.Sprintf("[c] use ctx %d (fits all GPU, KV ~%s)", bestFit, gibString(kvAt(bestFit))),
		}
		hint.suggestCtx = bestFit
	} else {
		hint.lines = []string{
			fmt.Sprintf("ctx %d would spill to system RAM", cur),
			"does not fit all-GPU even at " + strings.Join(rungParts, " · "),
		}
	}
	return hint
}

// gibString renders an MiB figure as a compact "N.N GB" string.
func gibString(mb int) string {
	return fmt.Sprintf("%.1f GB", float64(mb)/1024.0)
}

func loadModels(dir, cacheDir, backend string, caps *detect.Capabilities) []ModelItem {
	return enrichModelItems(discoverModels(dir), cacheDir, backend, caps)
}

func loadRecognizedModels(dir, cacheDir, backend string, caps *detect.Capabilities) []ModelItem {
	items := mergeModelItems(
		discoverModels(dir),
		discoverModelsFromPaths(modelstore.LoadDiscoveredPaths(cacheDir)),
	)
	items = enrichModelItems(items, cacheDir, backend, caps)
	// Re-apply the usage sort after enrichment: display order must reflect real
	// launch history, not discovery order.
	sortModels(items, modelusage.Load(cacheDir))
	return items
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
		if m.models[i].AutoBackend != "" {
			modelTag = m.models[i].AutoBackend
		}
		m.models[i].Tuned = tune.CountTunedConfigs(m.cacheDir, m.models[i].Name, modelTag)
	}
	m.rebuildMainList()
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
				_ = c.SetCtxValue(v)
			}},
		{label: "KV placement", kind: "enum", options: []string{"auto", "gpu", "cpu"},
			get: func(c *config.Config) string { return c.KVPlacement },
			set: func(c *config.Config, v string) { c.KVPlacement = v }},
		{label: "KV quality", kind: "enum", options: kvQualityOptions(),
			get: func(c *config.Config) string { return c.KVQuality },
			set: func(c *config.Config, v string) { c.KVQuality = v }},
		{label: "Full SWA cache", kind: "bool",
			get: func(c *config.Config) string { return boolLabel(c.SWAFull) },
			set: func(c *config.Config, v string) { c.SWAFull = v == "on" }},
		{label: "Remember live probes", kind: "bool",
			get: func(c *config.Config) string { return boolLabel(c.AllowLiveMemoryProbe) },
			set: func(c *config.Config, v string) { c.AllowLiveMemoryProbe = v == "on" }},
		{label: "Support expert / optimizer", kind: "enum", options: []string{"auto", "on", "off"},
			get: func(c *config.Config) string { return c.SupportExpert },
			set: func(c *config.Config, v string) { c.SupportExpert = v }},
		{label: "Support online research", kind: "bool",
			get: func(c *config.Config) string { return boolLabel(c.SupportOnline) },
			set: func(c *config.Config, v string) { c.SupportOnline = v == "on" }},
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
		{label: "RAM limit percent", kind: "text",
			get: func(c *config.Config) string { return strconv.Itoa(c.RAMLimitPercent) },
			set: atoiSet(func(c *config.Config, n int) { c.RAMLimitPercent = n })},
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
	if err := validateSettingValue(row.label, val); err != nil {
		m.message = fmt.Sprintf("Warning: %s was not changed: %v", row.label, err)
		m.messageType = "warning"
		return
	}
	row.set(m.settingsCfg, val)
	if err := m.settingsCfg.Save(); err != nil {
		m.message = fmt.Sprintf("Warning: %s set to %s for this session — save failed: %v", row.label, val, err)
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
	case "Full SWA cache":
		m.swaFull = m.settingsCfg.SWAFull
	case "Support expert / optimizer":
		m.supportExpert = m.settingsCfg.SupportExpert
	case "Support online research":
		m.supportOnline = m.settingsCfg.SupportOnline
	case "VRAM headroom":
		m.vramHeadroomMB = config.ParseBudgetMB(val)
		m.refreshRecommendations()
	case "RAM headroom":
		m.ramHeadroomMB = config.ParseBudgetMB(val)
		m.refreshRecommendations()
	case "RAM limit percent":
		m.ramLimitPercent = m.settingsCfg.RAMLimitPercent
		m.refreshRecommendations()
	case "Backend":
		m.backend = val
		m.refreshTunedCounts()
	case "Model directory":
		m.modelDir = val
		m.models = loadRecognizedModels(val, m.cacheDir, m.backend, m.caps)
		m.rebuildMainList()
		if m.messageType != "warning" {
			m.message = fmt.Sprintf("Saved: Model directory = %s (%d models)", val, len(m.models))
		}
	case "Vision":
		m.vision = m.settingsCfg.Vision
	case "Port":
		m.port = m.settingsCfg.Port
	case "Parallel":
		m.parallel = ""
		m.parallelSet = m.settingsCfg.Parallel > 0
		if m.settingsCfg.Parallel > 0 {
			m.parallel = strconv.Itoa(m.settingsCfg.Parallel)
		}
	case "AI-tune rounds":
		m.aituneRounds = m.settingsCfg.TuneRounds
	}
}

func validateSettingValue(label, val string) error {
	switch label {
	case "Port":
		_, err := config.ParsePort(val)
		return err
	case "Parallel", "AI-tune rounds":
		n, err := strconv.Atoi(strings.TrimSpace(val))
		if err != nil || n < 0 {
			return fmt.Errorf("must be a non-negative integer")
		}
	case "VRAM headroom", "RAM headroom":
		_, err := config.ParseBudgetMBStrict(val)
		return err
	case "RAM limit percent":
		_, err := config.ParseRAMLimitPercent(val)
		return err
	}
	return nil
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
			mm.message = fmt.Sprintf("Warning: Backend set to %s for this session — save failed: %v", v, err)
			mm.messageType = "warning"
		} else {
			mm.message = "Backend saved: " + v
			mm.messageType = "info"
		}
		mm.refreshTunedCounts()
	})
}

func backendManagerOptions() []string {
	opts := make([]string, 0, len(backends.Recipes())+2+len(backends.Load()))
	for _, recipe := range backends.Recipes() {
		opts = append(opts, "Install reviewed: "+recipe.Name)
	}
	opts = append(opts, "Add custom fork from Git URL", "Register an existing llama-server binary")
	for _, backend := range backends.Load() {
		opts = append(opts, "Remove installed: "+backend.Tag)
	}
	return opts
}

func (m *Model) openBackendManager(ret Screen) {
	m.openChoice("Backend forks", backendManagerOptions(), "", ret, func(mm *Model, value string) {
		switch {
		case strings.HasPrefix(value, "Install reviewed: "):
			name := strings.TrimSpace(strings.TrimPrefix(value, "Install reviewed: "))
			mm.launchRequest = &LaunchRequest{BackendArgs: []string{"install", name}}
		case value == "Add custom fork from Git URL":
			mm.screen = ScreenBackend
			mm.inputMode = "backend-add"
			mm.input.SetValue("")
			mm.input.Placeholder = "Git URL [tag] [model architecture]"
			mm.input.Focus()
		case value == "Register an existing llama-server binary":
			mm.screen = ScreenBackend
			mm.inputMode = "backend-register"
			mm.input.SetValue("")
			mm.input.Placeholder = "tag /path/to/llama-server [model architecture]"
			mm.input.Focus()
		case strings.HasPrefix(value, "Remove installed: "):
			tag := strings.TrimSpace(strings.TrimPrefix(value, "Remove installed: "))
			mm.openChoice("Remove "+tag+"?", []string{"Cancel", "Confirm: remove " + tag}, "Cancel", ret, func(cm *Model, confirm string) {
				if confirm == "Cancel" {
					return
				}
				cm.launchRequest = &LaunchRequest{BackendArgs: []string{"remove", tag}}
			})
		}
	})
}

// openRemoveModelChoice confirms before deleting the selected model's GGUF
// file(s) from disk, reusing the same confirm-first choice screen as backend
// fork removal (default cursor on Cancel). A model discovered outside the
// primary directory needs a second, stronger type-to-confirm step afterwards
// (see openExternalRemoveConfirm).
func (m *Model) openRemoveModelChoice(idx int) {
	if idx < 0 || idx >= len(m.models) {
		return
	}
	name := m.models[idx].Name
	if m.models[idx].External {
		m.openChoice("Delete "+name+" outside the primary directory?", []string{"Cancel", "Remove file " + name}, "Cancel", ScreenMain, func(cm *Model, confirm string) {
			if confirm == "Cancel" {
				return
			}
			cm.openExternalRemoveConfirm(idx)
		})
		return
	}
	m.openChoice("Delete "+name+"?", []string{"Cancel", "Confirm: delete " + name}, "Cancel", ScreenMain, func(cm *Model, confirm string) {
		if confirm == "Cancel" {
			return
		}
		cm.removeModelAt(idx)
	})
}

// openExternalRemoveConfirm opens the strong typed confirmation screen for a
// model outside the primary directory. Unlike the in-dir path, which deletes
// after one arrow-select confirm, this needs an explicit typed "yes": the file
// lives elsewhere on disk, and a single stray Enter must not destroy it. The
// hidden prompt is filled in by the user (the same input harness used by every
// other text screen); updateInputScreen validates it in the
// "external-remove-confirm" mode below. Esc on this screen falls through the
// generic input-screen path and discards the pending removal.
func (m *Model) openExternalRemoveConfirm(idx int) {
	m.pendingExternalRemoveIdx = idx
	m.screen = ScreenDownload
	m.inputMode = "external-remove-confirm"
	m.input.SetValue("")
	m.input.Placeholder = "type yes to permanently delete"
	m.input.Focus()
	m.message = fmt.Sprintf("This file is OUTSIDE the primary model directory (%s) and will be permanently deleted:\n%s\nType yes to confirm, or back out and make this its directory primary ('m' on the Main screen) instead.",
		m.modelDir, m.models[idx].Path)
	m.messageType = "warning"
}

// removeModelAt deletes a model's GGUF file(s) from disk — the same removal
// path as `ggrun models rm` — then refreshes the Main list. It matches by
// file path rather than display name: two models in different directories
// can share the same basename (see
// TestDiscoverModelsKeepsSameBasenameInDifferentDirectories), so the display
// name alone cannot be trusted to pick the right one.
func (m *Model) removeModelAt(idx int) {
	if idx < 0 || idx >= len(m.models) {
		return
	}
	item := m.models[idx]
	// A model outside the primary directory leaves the in-dir removal path: it
	// is not listed by modelstore.List(m.modelDir), so the match-by-relative-path
	// logic below could never find it. Removal is gated on the strong typed
	// "yes" confirmation (openExternalRemoveConfirm): updateInputScreen sets
	// pendingExternalRemoveIdx to this exact index only after the user typed
	// "yes", and removes it again afterwards. Anything else reaching this branch
	// (a stray call, a stale index) is refused.
	if item.External {
		if m.pendingExternalRemoveIdx != idx {
			m.message = "External model removal requires the typed confirmation."
			m.messageType = "warning"
			return
		}
		m.pendingExternalRemoveIdx = -1
		removed, err := modelstore.RemoveExternal(m.modelDir, item.Path)
		if err != nil {
			m.message = fmt.Sprintf("Error removing %s: %v", item.Name, err)
			m.messageType = "error"
			return
		}
		m.models = loadRecognizedModels(m.modelDir, m.cacheDir, m.backend, m.caps)
		m.rebuildMainList()
		m.message = fmt.Sprintf("Removed %s (%.1fGB freed).", removed.Name, float64(removed.Bytes)/(1024*1024*1024))
		m.messageType = "info"
		return
	}
	rel, err := filepath.Rel(m.modelDir, item.Path)
	if err != nil {
		m.message = fmt.Sprintf("Error removing %s: %v", item.Name, err)
		m.messageType = "error"
		return
	}
	rel = filepath.Clean(rel)
	all, err := modelstore.List(m.modelDir)
	if err != nil {
		m.message = fmt.Sprintf("Error removing %s: %v", item.Name, err)
		m.messageType = "error"
		return
	}
	var name string
matched:
	for _, candidate := range all {
		for _, f := range candidate.Files {
			if f == rel {
				name = candidate.Name
				break matched
			}
		}
	}
	if name == "" {
		m.message = fmt.Sprintf("Model not found on disk: %s", item.Name)
		m.messageType = "error"
		return
	}
	removed, err := modelstore.Remove(m.modelDir, name)
	if err != nil {
		m.message = fmt.Sprintf("Error removing %s: %v", item.Name, err)
		m.messageType = "error"
		return
	}
	m.models = loadRecognizedModels(m.modelDir, m.cacheDir, m.backend, m.caps)
	m.rebuildMainList()
	m.message = fmt.Sprintf("Removed %s (%.1fGB freed).", removed.Name, float64(removed.Bytes)/(1024*1024*1024))
	m.messageType = "info"
}

// openClearCachesChoice confirms before removing the selected model's cached
// measurements (probe caches, KV rate/geometry, calibration decisions,
// placement caches) while keeping the GGUF itself. Default cursor on Cancel.
func (m *Model) openClearCachesChoice(idx int) {
	if idx < 0 || idx >= len(m.models) {
		return
	}
	name := m.models[idx].Name
	m.openChoice("Clear caches for "+name+"?", []string{"Cancel", "Confirm: clear caches for " + name}, "Cancel", ScreenModelConfig, func(cm *Model, confirm string) {
		if confirm == "Cancel" {
			return
		}
		cm.clearModelCachesAt(idx)
	})
}

// clearModelCachesAt removes the selected model's cached configs via
// placement.ClearModelCaches (probe/KV/calibration/placement caches), keeps the
// GGUF, and reports how many files were removed. Discovery can be stale (the
// model may no longer be on disk), so build the model profile from the path
// with whatever metadata the scanner already parsed; ClearModelCaches only
// needs the basename identity to match cache headers.
func (m *Model) clearModelCachesAt(idx int) {
	if idx < 0 || idx >= len(m.models) {
		return
	}
	item := m.models[idx]
	profile := &placement.ModelProfile{Path: item.Path, Basename: filepath.Base(item.Path)}
	if item.KVProfile != nil {
		copy := *item.KVProfile
		copy.Path = item.Path
		if copy.Basename == "" {
			copy.Basename = filepath.Base(item.Path)
		}
		profile = &copy
	}
	removed, err := placement.ClearModelCaches(m.cacheDir, profile)
	if err != nil {
		m.message = fmt.Sprintf("Error clearing caches for %s: %v", item.Name, err)
		m.messageType = "error"
		return
	}
	m.message = fmt.Sprintf("Cleared %d cached config(s) for %s. Next launch re-measures placement.", removed, item.Name)
	m.messageType = "info"
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
				m.screen = m.choiceReturn
				m.choiceApply(&m, m.choiceOptions[m.choiceCursor])
			}
			if m.launchRequest != nil && len(m.launchRequest.BackendArgs) > 0 {
				return m, tea.Quit
			}
		case "q", "Q":
			return m, tea.Quit
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
	case "q", "Q":
		return m, tea.Quit
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
		val := row.get(m.settingsCfg)
		if row.label == "KV quality" {
			val = kvQualityLabel(val)
		}
		line := fmt.Sprintf("%-17s %s", row.label+":", val)
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
		KVQualitySet: m.kvQualityTouched,
		SWAFull:      m.swaFull,
		// Only emit --swa-full/--no-swa-full when this launch actually deviates
		// from the saved setting. Emitting it unconditionally turned a stored
		// preference into a command-line flag, and ggrun treats a typed flag as
		// inviolable: userExplicitBackendFlag reads OriginalArgs, so the
		// recovery ladder, the advisory notice and the support expert's
		// remove_generated_feature action were all locked out of it. Measured
		// 2026-08-03: --swa-full cost 5.3 GiB of KV on CUDA0 (6196 MiB vs 871),
		// which is exactly what made the launch unfittable — and nothing was
		// permitted to drop it, because the TUI had "typed" it. A setting is a
		// preference; only a human at the command line is an instruction.
		SWAFullSet:    m.swaFullTouched,
		FlashAttn:     true,
		Parallel:      parallel,
		ParallelSet:   parallelSet,
		Vision:        m.vision,
		Backend:       m.requestBackend(),
		TuneCache:     m.tunePath,
		AITune:        m.aitune,
		AITuneRounds:  m.aituneRounds,
		Benchmark:     m.benchmark,
		ClaudeCode:    m.claudeCode,
		ClaudeProfile: m.claudeProfile,
		// Only emit the reviewer override when the user explicitly deviated from
		// the automatic choice; an empty value keeps the CLI default.
		ClaudeReviewerOverride: m.claudeReviewer,
		SupportExpert:          m.supportExpert,
		SupportOnline:          m.supportOnline,
		SupportSet:             m.supportSet,
		NoCachedConfig:         m.noCachedConfig,
		ChatTemplate:           m.chatTemplate,
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
	Update      bool
	BackendArgs []string
	// BackendInstallError is set by the caller when a model-aware backend
	// install failed; RunAfterBackendInstall reports it on the model screen.
	BackendInstallError string
	DownloadRepo        string
	DownloadQuant       string
	// DownloadDir sends this one download somewhere other than the configured
	// model directory, without changing that setting. Large quants routinely
	// have to land on a different disk than the default one, and making the
	// user repoint ModelDir and then remember to put it back is both tedious
	// and easy to get wrong.
	DownloadDir   string
	ModelPath     string
	Port          int
	CtxSize       int
	CtxFlag       string
	KVPlacement   string
	KVQuality     string
	KVQualitySet  bool // TUI explicitly cycled KV quality; emit an explicit override
	SWAFull       bool
	SWAFullSet    bool // TUI explicitly selected on/off; emit an override either way
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
	ClaudeProfile string
	// ClaudeReviewerOverride picks the local worker/reviewer model in Claude
	// Code mode: empty/"auto" and "qwen" resolve to the Qwen3.5-4B
	// worker/reviewer, "qwen2b" forces the small/light Qwen3.5-2B review-only
	// profile, and "nanbeige" forces the NanoBeige4.2 big-MoE worker. "off"
	// seats no reviewer at all (main-model self-review).
	ClaudeReviewerOverride string
	ResumeSession          string // reopen this recorded Claude Code session
	SupportExpert          string // optional native support/optimizer policy: off, auto, on
	SupportOnline          bool   // allow typed official llama.cpp research
	SupportSet             bool   // TUI explicitly selected the support and research policy
	// NoCachedConfig derives this launch fresh, skipping cached placement/probe
	// measurements without deleting them (the "launch without cached config"
	// escape hatch for a stale placement or probe).
	NoCachedConfig bool
	// ChatTemplate forces a corrected chat template from the data-driven
	// catalog (pkg/chattemplate), mirroring the CLI's --chat-template <name>.
	// Empty means auto-match. Per-launch only.
	ChatTemplate string
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
	// Only emit --kv-quality when this launch actually deviates from the saved
	// setting, mirroring --swa-full above: a stored preference belongs in config,
	// where ggrun still applies it but may withdraw it on a memory failure. A
	// cycled row is an explicit per-launch choice and goes out as a typed flag.
	if req.KVQualitySet && req.KVQuality != "" {
		args = append(args, "--kv-quality", req.KVQuality)
	}
	if req.SWAFullSet {
		if req.SWAFull {
			args = append(args, "--swa-full")
		} else {
			args = append(args, "--no-swa-full")
		}
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
	if req.SupportExpert != "" {
		args = append(args, "--support-expert", req.SupportExpert)
	}
	if req.SupportSet {
		if req.SupportOnline {
			args = append(args, "--support-online")
		} else {
			args = append(args, "--no-support-online")
		}
	}
	if req.NoCachedConfig {
		args = append(args, "--no-cached-config")
	}
	// Only emit the chat-template override when the user explicitly forced one;
	// empty keeps the CLI's automatic catalog matching.
	if v := strings.TrimSpace(req.ChatTemplate); v != "" {
		args = append(args, "--chat-template", v)
	}
	if req.ClaudeCode {
		args = append(args, "--claude-code")
		if req.ClaudeProfile != "" {
			args = append(args, "--claude-profile", req.ClaudeProfile)
		}
		// Only emit the reviewer override when the user explicitly picked one;
		// an empty or "auto" value keeps the CLI default.
		if v := strings.TrimSpace(req.ClaudeReviewerOverride); v != "" && v != "auto" {
			args = append(args, "--claude-reviewer", v)
		}
		// Resume goes through the same launch argv as the CLI, so the TUI and
		// the command line share one implementation.
		if req.ResumeSession != "" {
			args = append(args, "--claude-resume", req.ResumeSession)
		}
	}
	return args
}

// chatTemplateLabel renders the Chat template row. Empty is "auto" — the
// launch auto-matches known-broken templates from the data-driven catalog.
// The resolved override is shown when the model's own template would already
// be fixed by the catalog, so a user launching that model sees the active fix.
func (m Model) chatTemplateLabel(model ModelItem) string {
	if m.cfgMemo != nil {
		m.cfgMemo.ensure(m, model)
		return m.cfgMemo.template
	}
	return m.chatTemplateLabelComputed(model)
}

// chatTemplateLabelComputed renders the Chat-template label without consulting
// the config memo (used as the memo's fallback when it is not initialized).
func (m Model) chatTemplateLabelComputed(model ModelItem) string {
	if v := strings.TrimSpace(m.chatTemplate); v != "" {
		return "forced: " + v
	}
	if entry, ok := m.autoChatTemplate(model); ok {
		return "auto (" + entry.Name + ")"
	}
	return "auto"
}

// autoChatTemplate returns the catalog entry the launch would auto-apply to
// this model: a user override (m.chatTemplate) wins unconditionally; otherwise
// the first enabled entry whose arch/basename matches and whose embedded
// template carries the raise_exception guard (mirrors catalogTemplateArgs in
// the CLI). It reports ok=false when no override applies, so the TUI only
// surfaces the row when there is something to see or force.
func (m Model) autoChatTemplate(model ModelItem) (chattemplate.Entry, bool) {
	if v := strings.TrimSpace(m.chatTemplate); v != "" {
		return chattemplate.ResolveOverride(v)
	}
	arch := model.Architecture
	if arch == "" {
		arch = model.Arch
	}
	basename := filepath.Base(model.Path)
	// Use the template captured once at enrich time (model.ChatTemplate) instead
	// of reopening the .gguf here, which used to happen on every config-screen
	// render and cost ~700ms per keystroke. Models not captured at enrich (e.g.
	// hand-built rows in tests) carry "" here, the same value gguf.ChatTemplate
	// would return for an absent path.
	return chattemplate.Resolve(arch, basename, model.ChatTemplate, true)
}

// chatTemplateOrdered returns the cycle order for the Chat template row. "auto"
// is the empty value; the forced choices are the catalog entry names.
func chatTemplateOrdered() []string {
	return append([]string{"auto"}, chattemplate.Names()...)
}

func supportExpertLabel(mode string) string {
	switch strings.TrimSpace(mode) {
	case "off":
		return "off"
	case "on":
		return "on (required, ephemeral)"
	default:
		return "auto (installed-only, ephemeral)"
	}
}

// claudeReviewerLabel describes the reviewer/worker selector value without
// converting its empty default into an explicit launch flag. The profiles are
// presented by tradeoff rather than raw model names: the "big/fast worker"
// profile is Nanbeige4.2 — a WORKER model that both reviews and does real work
// (heavier, stays resident); the "small/light" profile is Qwen3.5-4B, which —
// like Nanbeige — serves both worker and reviewer roles, just lighter.
// Only "qwen2b" is review-only. "auto" keeps ggrun's automatic choice.
func claudeReviewerLabel(reviewer string) string {
	switch strings.TrimSpace(reviewer) {
	case "qwen":
		return "small/light (Qwen3.5-4B, worker+reviewer)"
	case "qwen2b":
		return "lightest (Qwen3.5-2B, review-only)"
	case "nanbeige":
		return "big/fast worker (Nanbeige4.2, reviews + works)"
	case "off":
		return "off (no reviewer, main model self-reviews)"
	default:
		return "auto (automatic)"
	}
}

// claudeProfileLabel describes the selector value without converting its empty
// default into an explicit launch flag.
func claudeProfileLabel(profile string) string {
	switch profile {
	case "agent-interactive":
		return "agent-interactive (1 foreground agent)"
	case "agent-parallel":
		return "agent-parallel (4 workflow slots)"
	default:
		return "default (automatic)"
	}
}

// prelaunchParallelLabel mirrors the Claude scheduling policy closely enough
// for the confirmation screen. Context fitting may still reduce an automatic
// slot count; explicit --parallel remains authoritative.
func (m Model) prelaunchParallelLabel() string {
	if m.parallelSet && m.parallel != "" {
		return m.parallel + " (explicit)"
	}
	if m.claudeCode {
		switch m.claudeProfile {
		case "agent-interactive":
			return "agent-interactive (1 foreground slot)"
		case "agent-parallel":
			return "agent-parallel (4 workflow slots)"
		default:
			if m.parallel == "" || m.parallel == "1" {
				return "automatic (Claude Code policy; target 4 slots)"
			}
		}
	}
	if !m.parallelSet && (m.parallel == "" || m.parallel == "1") {
		return "automatic (1)"
	}
	return m.parallel
}

// noTerminalError is what a bare "ggrun" prints when there is no TTY to drive
// the interactive UI (cron jobs, CI, piped stdin). The caller only wraps this in
// "Error: %v", so the subcommand guidance has to travel inside the message.
func noTerminalError() error {
	return errors.New("no terminal available for the interactive UI; use subcommands instead, e.g. ggrun recommend | ggrun models list | ggrun <model.gguf>")
}

func runModel(initial Model) (*LaunchRequest, error) {
	p := tea.NewProgram(initial, tea.WithAltScreen())
	m, err := p.Run()
	if err != nil {
		return nil, err
	}
	if model, ok := m.(Model); ok && model.launchRequest != nil {
		return model.launchRequest, nil
	}
	return nil, nil
}

// Run starts the TUI and returns a launch request if the user chose to launch.
// The terminal check happens before any GGUF scan. Hardware detection and header
// reads then run behind a loading screen so a large model directory is not a
// blank wait.
func Run() (*LaunchRequest, error) {
	if !terminalAvailable() {
		return nil, noTerminalError()
	}
	return runModel(loadingModel())
}

// RunAfterBackendInstall reopens the same model and settings after a
// model-aware backend install: at pre-launch when it succeeded, at the model
// configuration with the error when it failed. The recipe tag stays scoped to
// this request; it does not overwrite the user's default backend for unrelated
// models.
func RunAfterBackendInstall(req *LaunchRequest) (*LaunchRequest, error) {
	if !terminalAvailable() {
		return nil, noTerminalError()
	}
	return runModel(afterBackendInstallModel(req))
}

// afterBackendInstallModel builds the screen shown after a model-aware backend
// install, successful or not.
func afterBackendInstallModel(req *LaunchRequest) Model {
	m := InitialModel()
	if req == nil {
		return m
	}
	// Do not copy req.Backend (the recipe tag) into m.backend: that is the
	// session default, so backing out and picking an unrelated model sent
	// --backend <recipe> as an explicit pin. The rescan in InitialModel already
	// routes this model to the fork through its AutoBackend.
	if !m.selectModelPath(req.ModelPath) {
		m.message = "Backend installed, but the selected model is no longer available: " + req.ModelPath
		if failure := strings.TrimSpace(req.BackendInstallError); failure != "" {
			m.message = "Backend install failed: " + failure
			m.messageType = "error"
		} else {
			m.messageType = "warning"
		}
		m.screen = ScreenMain
		return m
	}
	m.applyLaunchRequestFields(req)
	m.backendRouteBypass = false
	m.replayRequest = nil
	m.replaySavedAt = time.Time{}
	if failure := strings.TrimSpace(req.BackendInstallError); failure != "" {
		m.message = "Backend install failed: " + failure + ". Nothing was changed; choose another backend or retry the install."
		m.messageType = "error"
		m.screen = ScreenModelConfig
		return m
	}
	m.message = "Backend installed and auto-selected for this model. Review the launch, then press Enter."
	m.messageType = "info"
	m.screen = ScreenPrelaunch
	return m
}
