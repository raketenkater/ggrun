package tui

import "strconv"

func (m *Model) cycleClaudeMode(dir int) {
	mode := 0
	if m.claudeCode {
		mode = 1
	}
	if m.claudeCode && m.claudeHeadless {
		mode = 2
	}
	mode = (mode + dir + 3) % 3
	m.claudeCode, m.claudeHeadless = mode != 0, mode == 2
	if mode != 1 {
		m.resumeSession, m.resumeRun, m.resumeCached = "", "", 0
	}
	m.invalidateConfigMemo()
}

func (m Model) claudeModeLabel() string {
	if !m.claudeCode {
		return "Off"
	}
	if m.claudeHeadless {
		return "Remote client — serve without launching Claude here"
	}
	return "On this machine — serve and launch Claude Code"
}

func (m Model) gatewayPortLabel() string {
	if m.gatewayPort <= 0 {
		return "8080"
	}
	return strconv.Itoa(m.gatewayPort)
}
func (m Model) gatewayHostLabel() string {
	if m.gatewayHost == "" {
		return "127.0.0.1"
	}
	return m.gatewayHost
}
func (m Model) gatewayURLLabel() string {
	if m.gatewayURL == "" {
		return "not set (use the reachable server/proxy URL on laptop)"
	}
	return m.gatewayURL
}
func (m Model) gatewayTokenLabel() string {
	if m.gatewayTokenFile == "" {
		return "automatic private credential file"
	}
	return m.gatewayTokenFile
}
