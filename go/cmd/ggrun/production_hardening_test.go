package main

import "testing"

func TestHotExpertEffectivePolicySharesTUIAndCLIIdentity(t *testing.T) {
	isolateConfig(t)
	cli, err := parseLaunchArgs([]string{"model.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	tui, err := parseLaunchArgs([]string{"model.gguf", "--hot-experts", cli.HotExperts})
	if err != nil {
		t.Fatal(err)
	}
	if requestedLaunchPolicyIdentity(cli, nil) != requestedLaunchPolicyIdentity(tui, nil) {
		t.Fatal("same effective policy split TUI and CLI evidence")
	}
	baseline := requestedLaunchPolicyIdentity(cli, nil)
	for _, policy := range []string{"off", "on", "8", "16"} {
		changed := *cli
		changed.HotExperts = policy
		if policy != cli.HotExperts && requestedLaunchPolicyIdentity(&changed, nil) == baseline {
			t.Fatalf("different policy %q shared evidence", policy)
		}
	}
	// Unlike spelling hot-experts=auto, explicitly pinning parallelism changes
	// which coordinates the optimizer may move and must remain in the identity.
	changed := *cli
	changed.ParallelSet = !cli.ParallelSet
	if requestedLaunchPolicyIdentity(&changed, nil) == baseline {
		t.Fatal("explicit parallel constraint lost its scope")
	}
}

func TestCalibrationRelaunchKeepsRecoveredBaselineWithoutPromotion(t *testing.T) {
	baseline := calibrationMeasurement{Name: "default", Args: []string{"server", "-b", "2048", "-cram", "7680"}}
	hot := calibrationMeasurement{Name: "hot-experts-16", Args: []string{"server", "-b", "2048", "--moe-expert-cache", "16"}}
	recovered := []string{"server", "-b", "2048", "-cram", "6656"}
	for _, tc := range []struct {
		name   string
		winner calibrationMeasurement
		args   []string
		want   calibrationRelaunchDisposition
	}{
		{"exact baseline", baseline, baseline.Args, calibrationExactRelaunch},
		{"recovered baseline stays serving", baseline, recovered, calibrationKeepRecoveredBaseline},
		{"exact cache winner", hot, hot.Args, calibrationExactRelaunch},
		{"cache fallback to measured baseline", hot, baseline.Args, calibrationUseMeasuredBaseline},
		{"changed cache winner must restore baseline", hot, recovered, calibrationRestoreMeasuredBaseline},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := classifyCalibrationRelaunch(tc.winner, baseline, tc.args); got != tc.want {
				t.Fatalf("got disposition %v, want %v", got, tc.want)
			}
		})
	}
}
