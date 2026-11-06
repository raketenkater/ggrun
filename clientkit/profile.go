// Package clientkit defines ggrun's versioned remote Claude connection contract.
// It has no model runtime, filesystem or network-provider dependency.
package clientkit

import (
	"fmt"
	"net"
	"net/url"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

const ProtocolVersion = 1
const PolicyVersion = 1
const MaxTimeoutMS = 2147483647

type Profile struct {
	ProtocolVersion int    `json:"protocol_version"`
	ServerVersion   string `json:"server_version"`
	ModelID         string `json:"model_id"`
	ModelName       string `json:"model_name"`
	MainAlias       string `json:"main_alias"`
	UtilityAlias    string `json:"utility_alias"`
	ContextPerSlot  int    `json:"context_per_slot"`
	Slots           int    `json:"slots"`
	Vision          bool   `json:"vision"`
	SeparateWorker  bool   `json:"separate_worker"`
	Policy          Policy `json:"policy"`
}

// Policy contains only known, typed settings. A server cannot provide a shell
// command, path, arbitrary environment variable or permission override.
type Policy struct {
	Version        int    `json:"version"`
	CompactWindow  int    `json:"compact_window"`
	CompactPercent int    `json:"compact_percent"`
	Effort         string `json:"effort"`
	TimeoutMS      int    `json:"timeout_ms"`
	WorkflowHook   bool   `json:"workflow_hook"`
}

func (p Profile) Validate() error {
	if p.ProtocolVersion != ProtocolVersion || p.Policy.Version != PolicyVersion {
		return fmt.Errorf("unsupported ggrun connection protocol %d/policy %d; update ggrun-client or the server", p.ProtocolVersion, p.Policy.Version)
	}
	if p.ModelID == "" || len(p.ModelID) > 256 || strings.ContainsAny(p.ModelID, "\r\n\x00") || p.ContextPerSlot < 1 || p.Slots < 1 {
		return fmt.Errorf("server returned incomplete model/context metadata")
	}
	for _, s := range []string{p.MainAlias, p.UtilityAlias} {
		if s == "" || len(s) > 128 || strings.ContainsAny(s, "\r\n\x00") {
			return fmt.Errorf("server returned an invalid model alias")
		}
	}
	if p.Policy.CompactWindow < 1 || p.Policy.CompactWindow > p.ContextPerSlot || p.Policy.CompactPercent < 1 || p.Policy.CompactPercent > 95 {
		return fmt.Errorf("server returned a compaction policy outside the served slot context")
	}
	if p.Policy.TimeoutMS < 1 || p.Policy.TimeoutMS > MaxTimeoutMS {
		return fmt.Errorf("server returned an invalid inference timeout")
	}
	switch p.Policy.Effort {
	case "low", "medium", "high", "xhigh", "max", "auto":
	default:
		return fmt.Errorf("server returned an unsupported effort setting")
	}
	return nil
}

// ClientAlias accepts Claude's known tiers (mapped above) and served aliases.
// It permits explicit cheap-tier/subagent choices without permitting a cloud ID.
func (p Profile) ClientAlias(model string) bool {
	switch model {
	case p.MainAlias, p.UtilityAlias, "inherit", "haiku", "sonnet", "opus", "fable", "best":
		return true
	}
	return false
}

// BaseURL validates a gateway URL without guessing a private-network provider.
// It retains an explicit reverse-proxy path prefix. Credentials are separate.
func BaseURL(raw string, allowHTTP bool) (string, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u == nil || u.Hostname() == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Opaque != "" {
		return "", fmt.Errorf("use an HTTP(S) gateway base URL without credentials, query or fragment")
	}
	if u.Port() != "" {
		if n, err := strconv.Atoi(u.Port()); err != nil || n < 1 || n > 65535 {
			return "", fmt.Errorf("invalid gateway port")
		}
	}
	host := strings.ToLower(u.Hostname())
	ip := net.ParseIP(host)
	loopback := host == "localhost" || (ip != nil && ip.IsLoopback())
	if host == "0.0.0.0" || host == "::" {
		return "", fmt.Errorf("a wildcard listener is not a client destination; use the server's reachable address")
	}
	if u.Scheme == "http" && !loopback && !allowHTTP {
		return "", fmt.Errorf("remote cleartext HTTP requires explicit --allow-http; otherwise use HTTPS or a local tunnel")
	}
	u.Path = strings.TrimRight(u.Path, "/")
	if strings.HasSuffix(u.Path, "/v1") {
		return "", fmt.Errorf("use the gateway base URL without /v1; Claude Code appends the API path")
	}
	return strings.TrimRight(u.String(), "/"), nil
}

// Environment is shared by the server's local Claude launcher and the laptop
// connector. It changes only this child process; saved Claude settings survive.
func (p Profile) Environment(baseURL, token string, inherited []string) []string {
	old := map[string]string{}
	for _, pair := range inherited {
		if k, v, ok := strings.Cut(pair, "="); ok {
			if runtime.GOOS == "windows" {
				k = strings.ToUpper(k)
			}
			old[k] = v
		}
	}
	overrides := map[string]string{
		"ANTHROPIC_BASE_URL": baseURL, "ANTHROPIC_AUTH_TOKEN": token,
		"ANTHROPIC_MODEL": p.MainAlias, "ANTHROPIC_DEFAULT_OPUS_MODEL": p.MainAlias,
		"ANTHROPIC_DEFAULT_SONNET_MODEL": p.MainAlias, "ANTHROPIC_DEFAULT_FABLE_MODEL": p.MainAlias,
		"ANTHROPIC_DEFAULT_HAIKU_MODEL": p.UtilityAlias, "ANTHROPIC_SMALL_FAST_MODEL": p.UtilityAlias,
		"CLAUDE_CODE_SUBAGENT_MODEL":          p.MainAlias,
		"CLAUDE_CODE_ATTRIBUTION_HEADER":      "0",
		"API_TIMEOUT_MS":                      strconv.Itoa(p.Policy.TimeoutMS),
		"CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS": strconv.Itoa(p.Policy.TimeoutMS),
		"CLAUDE_ENABLE_BYTE_WATCHDOG":         "0", "CLAUDE_ENABLE_STREAM_WATCHDOG": "0", "API_FORCE_IDLE_TIMEOUT": "0",
		"CLAUDE_CODE_MAX_CONTEXT_TOKENS":  strconv.Itoa(p.ContextPerSlot),
		"CLAUDE_CODE_AUTO_COMPACT_WINDOW": strconv.Itoa(p.Policy.CompactWindow),
		"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": strconv.Itoa(p.Policy.CompactPercent),
		"CLAUDE_CODE_EFFORT_LEVEL":        p.Policy.Effort,
	}
	// These fixed headers bind a running client to the capacity it configured.
	// A recovery may reduce capacity; reject an old client before inference
	// rather than silently letting its old compaction window overflow the slot.
	var headers []string
	for _, line := range strings.Split(old["ANTHROPIC_CUSTOM_HEADERS"], "\n") {
		key, _, _ := strings.Cut(line, ":")
		if strings.EqualFold(strings.TrimSpace(key), "X-Ggrun-Model-ID") || strings.EqualFold(strings.TrimSpace(key), "X-Ggrun-Context") {
			continue
		}
		if strings.TrimSpace(line) != "" {
			headers = append(headers, line)
		}
	}
	headers = append(headers, "X-Ggrun-Model-ID: "+p.ModelID, "X-Ggrun-Context: "+strconv.Itoa(p.ContextPerSlot))
	overrides["ANTHROPIC_CUSTOM_HEADERS"] = strings.Join(headers, "\n")
	if v := old["CLAUDE_CODE_SUBAGENT_MODEL"]; v != "" && p.ClientAlias(v) {
		overrides["CLAUDE_CODE_SUBAGENT_MODEL"] = v
	}
	// Preserve explicit, valid client choices within the served capacity.
	for _, k := range []string{"CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "CLAUDE_CODE_EFFORT_LEVEL"} {
		v := old[k]
		if v == "" {
			continue
		}
		if k == "CLAUDE_CODE_AUTO_COMPACT_WINDOW" {
			n, e := strconv.Atoi(v)
			if e != nil || n < 1 || n > p.ContextPerSlot {
				continue
			}
		}
		if k == "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" {
			n, e := strconv.Atoi(v)
			if e != nil || n < 1 || n > 95 {
				continue
			}
		}
		overrides[k] = v
	}
	out := make([]string, 0, len(inherited)+len(overrides))
	for _, pair := range inherited {
		k, _, _ := strings.Cut(pair, "=")
		if runtime.GOOS == "windows" {
			k = strings.ToUpper(k)
		}
		if k == "ANTHROPIC_API_KEY" {
			continue
		}
		if _, ok := overrides[k]; !ok {
			out = append(out, pair)
		}
	}
	keys := make([]string, 0, len(overrides))
	for k := range overrides {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := overrides[k]
		out = append(out, k+"="+v)
	}
	return out
}
