package backends

import (
	"fmt"
	"strings"

	"github.com/raketenkater/ggrun/pkg/gguf"
)

// SupportState is the answer to "can this machine load this model, and with
// which backend?". ResolveSupport is the one place that answers it for the
// recommender and the TUI. A launch does not read it: backend selection for a
// launch is its own code (selectBackendForModel and friends).
type SupportState string

const (
	// SupportReady: an installed backend loads the model (Support.Backend).
	SupportReady SupportState = "ready"
	// SupportInstall: no installed backend loads it, and a supported build
	// does (Support.Install says which and how).
	SupportInstall SupportState = "install"
	// SupportNoEncoding: the file uses a tensor encoding a catalog entry
	// provides, no installed backend has it, and no build can start here.
	SupportNoEncoding SupportState = "no_encoding"
	// SupportUnsupported: no installed backend loads the architecture and no
	// supported route builds one here.
	SupportUnsupported SupportState = "unsupported"
	// SupportUnknownArch: no installed backend loads the architecture, and
	// neither the backend catalog nor the model catalog knows it.
	SupportUnknownArch SupportState = "unknown_arch"
	// SupportUnknown: it cannot be told (no architecture, no installed
	// backend, an unreadable backend, or encodings that could not be checked).
	SupportUnknown SupportState = "unknown"
)

// InstalledBackend is one backend a default launch can pick, with what is
// already known about it. Nothing here starts a probe.
type InstalledBackend struct {
	Name string // registered tag, or a display name
	Path string
	// LoadsArch and Probed are BackendSupportsArch's answer for the model's
	// architecture.
	LoadsArch, Probed bool
	// Encodings is a cached ProbeBackendEncodings result for this build; nil
	// when none is cached, which means "not known", never "absent".
	Encodings *EncodingProbe
}

// SupportInput is everything ResolveSupport combines.
type SupportInput struct {
	Arch string
	// Types is the model's tensor type histogram (gguf.Info.TensorTypes or
	// gguf.ReadTypeHistogramModel); nil when unknown.
	Types     []gguf.TypeUsage
	Installed []InstalledBackend
	// MainlineRegisters is the model catalog's stamp that current mainline
	// llama.cpp registers the architecture; MainlineKnown is false when no
	// stamp exists.
	MainlineRegisters, MainlineKnown bool
	// BuildBlocked says why a source build cannot start here ("" when it can):
	// a pinned backend, missing build tools, a platform without source builds.
	BuildBlocked string
}

// Support is ResolveSupport's answer.
type Support struct {
	State SupportState `json:"state"`
	// Backend is the installed backend that loads the model (ready only).
	Backend string `json:"backend,omitempty"`
	// Install is what gets the model running when State is install: the
	// first launch's offer, and a command where one exists.
	Install string `json:"install,omitempty"`
	// Requirement is the catalog's verdict on the architecture and types.
	Requirement Requirement `json:"-"`
	// Reason is the full, actionable sentence both surfaces print.
	Reason string `json:"reason"`
}

// Note is the full wording of the state, shared by `ggrun recommend` and the
// TUI.
func (s Support) Note() string { return s.Reason }

// Short is the list-row wording of the state ("" for ready and unknown, which
// a list row does not flag).
func (s Support) Short() string {
	switch s.State {
	case SupportInstall:
		return "no installed backend loads it; first launch offers a build"
	case SupportUnsupported:
		return "no backend here loads it"
	case SupportNoEncoding:
		return "needs a tensor encoding no installed backend has"
	case SupportUnknownArch:
		return "architecture unknown to ggrun"
	}
	return ""
}

// ResolveSupport combines the model's architecture and tensor types, the
// embedded catalog (ResolveRequirement and its recipes) and the installed
// backends into one state with an actionable reason.
func ResolveSupport(in SupportInput) Support { return loadedCatalog.resolveSupport(in) }

const buildMinutes = "about 20-40 min"

func (c *parsedCatalog) resolveSupport(in SupportInput) Support {
	arch := strings.ToLower(strings.TrimSpace(in.Arch))
	if arch == "" {
		return Support{State: SupportUnknown, Reason: "the model's architecture is not known (GGUF metadata missing), so backend support cannot be checked"}
	}
	req := c.resolve(arch, in.Types)
	s := Support{Requirement: req}

	var loaders []InstalledBackend
	allProbed := len(in.Installed) > 0
	for _, b := range in.Installed {
		if b.Probed && b.LoadsArch {
			loaders = append(loaders, b)
		}
		if !b.Probed {
			allProbed = false
		}
	}

	if req.Recipe != nil {
		return c.resolveEncodings(s, arch, *req.Recipe, in, loaders, allProbed)
	}

	switch {
	case len(loaders) > 0 && len(req.Unchecked) > 0:
		s.State = SupportUnknown
		s.Reason = fmt.Sprintf("installed backend %s loads %s, but the catalog entries %s need tensor encodings and this file's are not known yet; download the file or run a launch to check",
			loaders[0].Name, arch, strings.Join(req.Unchecked, ", "))
		return s
	case len(loaders) > 0:
		s.State, s.Backend = SupportReady, loaders[0].Name
		s.Reason = fmt.Sprintf("loads with installed backend %s", loaders[0].Name)
		return s
	case len(in.Installed) == 0:
		s.State = SupportUnknown
		s.Reason = "no installed backend was found, so loading " + arch + " cannot be checked; run `ggrun backend recipes` or reinstall ggrun"
		return s
	case !allProbed:
		s.State = SupportUnknown
		s.Reason = "an installed backend could not be read, so loading " + arch + " cannot be checked"
		return s
	}

	if offer, command, ok := c.buildRoute(arch, req.Family, in); ok {
		s.State, s.Install = SupportInstall, offer
		if command != "" {
			s.Install += " (or run `" + command + "`)"
		}
		s.Reason = "no installed backend loads " + arch + "; " + s.Install
		return s
	}
	s.State = SupportUnsupported
	s.Reason = "no installed backend loads " + arch + " and no supported route builds one here"
	if why := c.noRouteDetail(arch, req.Family, in); why != "" {
		s.Reason += "; " + why
	} else if !c.knowsArch(arch) && !in.MainlineKnown {
		s.State = SupportUnknownArch
		s.Reason = "no installed backend loads " + arch + ", and neither the backend catalog nor the model catalog knows this architecture; install a backend that registers it (`ggrun backend install <recipe>`, see `ggrun backend recipes`)"
	}
	return s
}

// resolveEncodings answers for a model whose file uses the tensor encodings a
// catalog entry provides.
func (c *parsedCatalog) resolveEncodings(s Support, arch string, rec Recipe, in SupportInput, loaders []InstalledBackend, allProbed bool) Support {
	names := c.encodingNamesOf(rec.Name)
	list := strings.Join(names, ", ")
	for _, b := range loaders {
		if strings.EqualFold(b.Name, rec.Tag) {
			s.State, s.Backend = SupportReady, b.Name
			s.Reason = fmt.Sprintf("loads with installed backend %s, the catalog entry for the tensor encodings %s", b.Name, list)
			return s
		}
	}
	allLack := allProbed
	for _, b := range loaders {
		if b.Encodings == nil {
			allLack = false
			continue
		}
		has, lacks := true, false
		for _, n := range names {
			if !b.Encodings.hasName(n) {
				has = false
			}
			if b.Encodings.Absent(n) {
				lacks = true
			}
		}
		if has {
			s.State, s.Backend = SupportReady, b.Name
			s.Reason = fmt.Sprintf("loads with installed backend %s, which carries the tensor encodings %s", b.Name, list)
			return s
		}
		if !lacks {
			allLack = false
		}
	}
	command := ""
	if !rec.HelperOnly {
		command = "ggrun backend install " + rec.Name
	}
	if !allLack {
		s.State = SupportUnknown
		s.Reason = fmt.Sprintf("the file uses the tensor encodings %s; whether an installed backend has them is not known yet", list)
		if command != "" {
			s.Install = "`" + command + "`"
			s.Reason += "; the reviewed " + rec.Name + " backend provides them (" + s.Install + ")"
		}
		return s
	}
	if in.BuildBlocked == "" && command != "" {
		s.State = SupportInstall
		s.Install = fmt.Sprintf("build the reviewed %s backend, %s (run `%s`)", rec.Name, buildMinutes, command)
		s.Reason = fmt.Sprintf("the file uses the tensor encodings %s, which no installed backend has; %s", list, s.Install)
		return s
	}
	s.State = SupportNoEncoding
	s.Reason = fmt.Sprintf("the file uses the tensor encodings %s, which no installed backend has", list)
	switch {
	case command == "":
		s.Reason += "; the catalog entry " + rec.Name + " that provides them is a helper-only build"
	default:
		s.Reason += "; the reviewed " + rec.Name + " backend provides them, but " + in.BuildBlocked
	}
	return s
}

// buildRoute mirrors the routes a first launch offers when no installed
// backend loads an architecture: a reviewed recipe, an accelerated build of a
// reviewed helper recipe's pinned commit, or current mainline when mainline
// registers the architecture and no name rule requires another family.
func (c *parsedCatalog) buildRoute(arch, family string, in SupportInput) (offer, command string, ok bool) {
	if in.BuildBlocked != "" || c == nil {
		return "", "", false
	}
	for _, r := range c.recipes {
		if !r.HelperOnly && strings.EqualFold(strings.TrimSpace(r.RouteArch), arch) {
			return "the first launch offers to build the reviewed " + r.Name + " backend, " + buildMinutes, "ggrun backend install " + r.Name, true
		}
	}
	for _, r := range c.recipes {
		if r.HelperOnly && r.Commit != "" && strings.EqualFold(strings.TrimSpace(r.RouteArch), arch) {
			return "the first launch offers to build an accelerated backend from the reviewed " + r.Name + " source, " + buildMinutes, "", true
		}
	}
	if in.MainlineRegisters && family == "" {
		return "the first launch offers to build current llama.cpp, " + buildMinutes, "", true
	}
	return "", "", false
}

// noRouteDetail says why no route exists when something would otherwise offer
// one ("" when nothing would).
func (c *parsedCatalog) noRouteDetail(arch, family string, in SupportInput) string {
	if c != nil {
		for _, r := range c.recipes {
			if strings.EqualFold(strings.TrimSpace(r.RouteArch), arch) && (!r.HelperOnly || r.Commit != "") && in.BuildBlocked != "" {
				return "the reviewed " + r.Name + " recipe needs a source build, but " + in.BuildBlocked
			}
		}
	}
	if in.MainlineRegisters && family != "" {
		return "it needs the " + family + " backend family, and no reviewed recipe builds one"
	}
	if in.MainlineRegisters && in.BuildBlocked != "" {
		return "current llama.cpp registers it, but " + in.BuildBlocked
	}
	if family != "" {
		return "it needs the " + family + " backend family, and no reviewed recipe builds one"
	}
	return ""
}

// knowsArch reports whether any catalog recipe or name rule names arch.
func (c *parsedCatalog) knowsArch(arch string) bool {
	if c == nil {
		return false
	}
	for _, r := range c.recipes {
		if strings.EqualFold(strings.TrimSpace(r.RouteArch), arch) {
			return true
		}
	}
	return c.requiredBackend(arch) != ""
}

func (c *parsedCatalog) encodingNamesOf(recipe string) []string {
	var out []string
	if c == nil {
		return nil
	}
	for i, r := range c.recipes {
		if r.Name == recipe && i < len(c.encodings) {
			for _, e := range c.encodings[i] {
				out = append(out, e.Name)
			}
		}
	}
	return out
}

// hasName reports that the probe found name in the build's type table.
func (p EncodingProbe) hasName(name string) bool {
	for n := range p.Found {
		if strings.EqualFold(n, name) {
			return true
		}
	}
	return false
}
