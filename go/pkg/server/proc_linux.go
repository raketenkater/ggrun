//go:build linux

package server

import "syscall"

// setParentDeathSignal makes an abruptly terminated ggrun process take its
// immediate launcher child down with it. Normal shutdown still uses Stop so
// the complete process group and any systemd scope are collected cleanly.
func setParentDeathSignal(attr *syscall.SysProcAttr) {
	attr.Pdeathsig = syscall.SIGTERM
}
