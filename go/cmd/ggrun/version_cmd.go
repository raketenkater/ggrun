package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
)

// buildIdentity is what an install check needs to prove which launcher it
// installed: the stamped version alone cannot, because a source build without
// release ldflags reports the package default.
type buildIdentity struct {
	Version    string `json:"version"`
	Revision   string `json:"revision,omitempty"`
	Modified   *bool  `json:"modified,omitempty"`
	CommitTime string `json:"commit_time,omitempty"`
	GoVersion  string `json:"go_version"`
	Platform   string `json:"platform"`
	Executable string `json:"executable,omitempty"`
}

func currentBuildIdentity() buildIdentity {
	id := buildIdentity{Version: version, GoVersion: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH}
	if exe, err := os.Executable(); err == nil {
		id.Executable = exe
	}
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, s := range info.Settings {
			switch s.Key {
			case "vcs.revision":
				id.Revision = s.Value
			case "vcs.modified":
				modified := s.Value == "true"
				id.Modified = &modified
			case "vcs.time":
				id.CommitTime = s.Value
			}
		}
	}
	return id
}

// cmdVersion keeps `ggrun --version` byte-for-byte unchanged (installers and
// update checks compare it) and adds `ggrun version --json` for identity.
func cmdVersion(args []string) {
	for _, a := range args {
		if a == "--json" {
			out, _ := json.Marshal(currentBuildIdentity())
			fmt.Println(string(out))
			return
		}
	}
	fmt.Println("ggrun", version)
}
