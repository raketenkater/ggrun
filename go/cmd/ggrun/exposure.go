package main

import (
	"fmt"
	"net"
	"strconv"
	"strings"
)

// exposedHostWarning is one line for a backend that will listen beyond
// loopback with no API key: anyone who can reach the port can use the model.
// It only informs; the launch proceeds unchanged.
func exposedHostWarning(host string, port int, args []string, getenv func(string) string) string {
	if isLoopbackHost(host) {
		return ""
	}
	for _, a := range args {
		if k, _, _ := strings.Cut(a, "="); k == "--api-key" || k == "--api-key-file" {
			return ""
		}
	}
	if getenv != nil && strings.TrimSpace(getenv("LLAMA_API_KEY")) != "" {
		return ""
	}
	return fmt.Sprintf("[launch] warning: the model server listens on %s with no API key; anyone who can reach this machine on port %d can use it. Use --host 127.0.0.1, or give the backend --api-key.",
		net.JoinHostPort(strings.Trim(host, "[]"), strconv.Itoa(port)), port)
}

func isLoopbackHost(host string) bool {
	host = strings.Trim(strings.TrimSpace(host), "[]")
	if host == "" || strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
