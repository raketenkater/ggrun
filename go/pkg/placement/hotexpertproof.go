package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Displacement proof for the hot-expert cache.
//
// The cache is not free. When residual VRAM cannot seat a useful cache, the
// planner frees whole GPU expert layers until one fits, so the cache is funded
// by evicting resident weights. Whether that trade pays is an empirical
// question about a particular model on particular hardware, and on 2026-09-08
// it did not pay here:
//
//	hot experts on   decode 6.39 tok/s (n=16)  2 expert layers resident
//	hot experts off  decode 7.26 tok/s (n=20)  5 expert layers resident
//
// Matched workload, same --n-cpu-moe, cache provably engaged in one arm and
// provably absent in the other. Decode fell 12% and the distributions do not
// overlap: the cache arm's best sample (6.71) is below the control's worst
// (6.96). Sixteen generations produced no warming, which is what a cache that
// mostly misses looks like -- 14 slots is 4.9% of 288 experts, and
// --moe-expert-cache-inserts uploads on every decode step regardless.
//
// The tempting fix is a minimum slot count. It is wrong twice: it hardcodes a
// threshold read off someone else's benchmark, and a slot count is not the
// quantity that matters. 14 of 288 experts is 4.9% coverage; 14 of 32 would be
// 44%. The same constant means opposite things on two models, which is exactly
// what invariant 8 forbids.
//
// So the gate is on displacement, measured per allocation identity:
//
//   - a cache that displaces nothing costs nothing and needs no proof;
//   - a cache that displaces resident layers must have beaten the packed
//     cache-free baseline on this identity before auto may choose it;
//   - an explicit `--hot-experts on` may serve once unproven, because that
//     serve is how the evidence gets recorded (invariant 3: an explicit user
//     choice is a constraint, not a preference to be overridden).
//
// Absent evidence, auto declines. A user who never asks for the cache never
// pays the 12%.

// HotExpertDisplacementProof is one identity's answer to "did the cache earn
// back the expert layers it evicted?".
type HotExpertDisplacementProof struct {
	Identity string
	// Demoted is how many GPU expert layers were freed to fund the cache.
	Demoted int
	Slots   int
	// CacheDecodeTPS and BaselineDecodeTPS are median decode throughput for the
	// cache-on plan and the packed cache-free baseline it displaced.
	CacheDecodeTPS    float64
	BaselineDecodeTPS float64
}

// Measured reports whether both arms of the comparison exist. One arm is not a
// comparison: invariant 7 refuses to promote on a baseline-only measurement.
func (p HotExpertDisplacementProof) Measured() bool {
	return p.CacheDecodeTPS > 0 && p.BaselineDecodeTPS > 0
}

// Earned reports whether the cache paid for what it displaced. Ties go against
// the cache: equal throughput for fewer resident weights is a worse plan, since
// the evicted layers would otherwise absorb a longer context or a larger batch.
func (p HotExpertDisplacementProof) Earned() bool {
	return p.Measured() && p.CacheDecodeTPS > p.BaselineDecodeTPS
}

// Summary renders the proof for a launch record.
func (p HotExpertDisplacementProof) Summary() string {
	if !p.Measured() {
		return "displacement unproven"
	}
	verdict := "lost"
	if p.Earned() {
		verdict = "won"
	}
	return fmt.Sprintf("%d slots displacing %d expert layer(s) %s: %.2f vs %.2f tok/s decode",
		p.Slots, p.Demoted, verdict, p.CacheDecodeTPS, p.BaselineDecodeTPS)
}

func hotExpertProofPath(cacheDir, modelPath string) string {
	base := filepath.Base(strings.TrimSpace(modelPath))
	if base == "" || base == "." || base == string(filepath.Separator) {
		base = "model"
	}
	return filepath.Join(cacheDir, "hotexpert-displacement-"+base+".proof")
}

// RecordHotExpertDisplacementProof stores one identity's comparison. Later
// results replace earlier ones for the same identity: hardware, backend build
// and workload all drift, and the freshest matched pair is the one that
// describes the machine as it is now (invariant 9).
func RecordHotExpertDisplacementProof(cacheDir, modelPath string, p HotExpertDisplacementProof) error {
	if cacheDir == "" || strings.TrimSpace(p.Identity) == "" {
		return nil
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return err
	}
	path := hotExpertProofPath(cacheDir, modelPath)
	kept := []string{}
	if data, err := os.ReadFile(path); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, p.Identity+"|") {
				continue
			}
			kept = append(kept, line)
		}
	}
	kept = append(kept, fmt.Sprintf("%s|%d|%d|%.4f|%.4f",
		p.Identity, p.Demoted, p.Slots, p.CacheDecodeTPS, p.BaselineDecodeTPS))
	return os.WriteFile(path, []byte(strings.Join(kept, "\n")+"\n"), 0o644)
}

// LoadHotExpertDisplacementProof returns the recorded comparison for one
// identity.
func LoadHotExpertDisplacementProof(cacheDir, modelPath, identity string) (HotExpertDisplacementProof, bool) {
	if cacheDir == "" || strings.TrimSpace(identity) == "" {
		return HotExpertDisplacementProof{}, false
	}
	data, err := os.ReadFile(hotExpertProofPath(cacheDir, modelPath))
	if err != nil {
		return HotExpertDisplacementProof{}, false
	}
	for _, line := range strings.Split(string(data), "\n") {
		parts := strings.Split(strings.TrimSpace(line), "|")
		if len(parts) != 5 || parts[0] != identity {
			continue
		}
		demoted, err1 := strconv.Atoi(parts[1])
		slots, err2 := strconv.Atoi(parts[2])
		cacheTPS, err3 := strconv.ParseFloat(parts[3], 64)
		baseTPS, err4 := strconv.ParseFloat(parts[4], 64)
		if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
			continue
		}
		return HotExpertDisplacementProof{
			Identity: identity, Demoted: demoted, Slots: slots,
			CacheDecodeTPS: cacheTPS, BaselineDecodeTPS: baseTPS,
		}, true
	}
	return HotExpertDisplacementProof{}, false
}

// hotExpertDisplacementAllowed decides whether a cache may be funded by
// evicting resident expert layers, and explains the decision for the record.
func hotExpertDisplacementAllowed(cacheDir, modelPath, identity string, demoted int, required bool) (bool, string) {
	if demoted <= 0 {
		// Residual VRAM funded this cache. It evicts nothing, so there is
		// nothing to prove: its cost is zero whatever its benefit turns out
		// to be.
		return true, "cache displaces no resident expert layer"
	}
	proof, ok := LoadHotExpertDisplacementProof(cacheDir, modelPath, identity)
	if ok && proof.Measured() {
		return proof.Earned(), "measured on this placement: " + proof.Summary()
	}
	if required {
		// The user asked for the cache by name. Serving it once is how the
		// comparison above ever gets recorded, and refusing an explicit request
		// on the grounds that it is unmeasured would make it permanently
		// unmeasurable.
		return true, fmt.Sprintf("displacing %d expert layer(s) unproven on this placement; "+
			"serving once because --hot-experts on was requested", demoted)
	}
	return false, fmt.Sprintf("declined: funding this cache evicts %d resident expert layer(s) "+
		"and no measurement on this placement shows the cache earns them back", demoted)
}
