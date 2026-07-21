//go:build !windows && !linux

package server

import "syscall"

func setParentDeathSignal(*syscall.SysProcAttr) {}
