package recommend

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// R-05 (review W12): the recommender sizes quants with its own copy of the
// planner's formulas. This test runs both on the same catalog rows and
// hardware fixtures and counts where they disagree about "fits". It changes no
// behaviour; it fails when either direction's share of the cases grows by more
// than a percentage point over the recorded baseline, so a change to either
// side that widens the gap is seen. Shares, not raw counts, so a catalog
// refresh that only adds rows does not fail it. Lower the baseline when a fix
// closes part of it.
//
// The planner gets a profile reconstructed from catalog geometry (expert
// bytes split by parameter share), so part of the count is that
// approximation, not a real disagreement; the bound still catches drift.

var agreementHardware = []struct {
	name string
	caps detect.Capabilities
}{
	{"cpu-32g", detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 32768, FreeMB: 30000}, CPU: detect.CPUInfo{Cores: 8, Threads: 16}}},
	{"gpu8-ram16", detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "gpu8", VRAMTotalMB: 8192, BandwidthMBps: 15760}},
		RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 14000}, CPU: detect.CPUInfo{Cores: 6, Threads: 12}}},
	{"gpu24-ram64", detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "gpu24", VRAMTotalMB: 24576, BandwidthMBps: 15760}},
		RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 60000}, CPU: detect.CPUInfo{Cores: 8, Threads: 16}}},
	{"gpu12x2-ram128", detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, Name: "gpu12a", VRAMTotalMB: 12288, BandwidthMBps: 15760},
		{Index: 1, Name: "gpu12b", VRAMTotalMB: 12288, BandwidthMBps: 7880}},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 120000}, CPU: detect.CPUInfo{Cores: 16, Threads: 32}}},
}

// profileFromCandidate reconstructs what the planner needs from catalog
// geometry. Expert bytes take the routed experts' share of the parameters.
func profileFromCandidate(c Candidate, q QuantOption) *placement.ModelProfile {
	sizeMB := int(q.SizeGB * 1024)
	p := &placement.ModelProfile{
		Path: c.Name + "-" + q.Name + ".gguf", Basename: c.Name, TotalSizeMB: sizeMB,
		SizeBytes: int64(sizeMB) << 20, NumLayers: c.Layers, IsMoE: c.MoE, NumExperts: c.Experts,
		HiddenSize: c.Embedding, EmbeddingLength: c.Embedding, FeedForwardLength: c.FeedForward,
		HeadCountKV: c.HeadCountKV, KeyLength: c.KeyLength, ValueLength: c.ValueLength,
		KVLoraRank: c.KVLoraRank, QLoraRank: c.QLoraRank, ContextSize: c.TrainCtx,
	}
	if c.MoE && c.Experts > 0 && c.ExpertFF > 0 && c.Embedding > 0 && c.Layers > 0 {
		moeLayers := max(c.Layers-c.LeadingDense, 1)
		expertParams := float64(moeLayers) * float64(c.Experts) * 3 * float64(c.Embedding) * float64(c.ExpertFF)
		total := c.TotalParamsB * 1e9
		share := 0.9
		if total > expertParams {
			share = expertParams / total
		}
		p.ExpertBytes = int64(float64(p.SizeBytes) * share)
		p.NonExpertBytes = p.SizeBytes - p.ExpertBytes
	} else {
		p.NonExpertBytes = p.SizeBytes
	}
	return p
}

func agreementCounts(t *testing.T) (recOnly, planOnly int, compared int, detail []string) {
	t.Helper()
	for _, hw := range agreementHardware {
		caps := hw.caps
		budget := hardware(&caps)
		for _, c := range Shortlist() {
			if !hasGeometry(c) {
				continue
			}
			for _, q := range quantOptions(c) {
				if q.SizeGB <= 0 {
					continue
				}
				_, _, _, _, recFits := fitQuant(budget, &caps, c, q)
				ctx := recommendAutoContext(&caps, c, int(q.SizeGB*1024))
				_, err := placement.Compute(&caps, profileFromCandidate(c, q), placement.Options{
					ContextSize: ctx, KVPlacement: "auto", KVQuality: "auto", Parallel: 1, CacheDir: t.TempDir(),
				})
				planFits := err == nil
				compared++
				if recFits != planFits {
					if recFits {
						recOnly++
					} else {
						planOnly++
					}
					detail = append(detail, fmt.Sprintf("%s %s %s: recommender %t planner %t", hw.name, c.Name, q.Name, recFits, planFits))
				}
			}
		}
	}
	sort.Strings(detail)
	return recOnly, planOnly, compared, detail
}

func TestRecommenderAndPlannerDisagreementDoesNotGrow(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	recOnly, planOnly, compared, detail := agreementCounts(t)
	path := filepath.Join("testdata", "planner_disagreement_baseline.txt")
	if os.Getenv("GGRUN_UPDATE_DISAGREEMENT_BASELINE") == "1" {
		body := fmt.Sprintf("%d %d\n# of %d (hardware, row, quant) cases: %d the recommender offers but the planner refuses, %d the planner places but the recommender does not offer\n%s\n",
			recOnly, planOnly, compared, recOnly, planOnly, strings.Join(detail, "\n"))
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var wantRec, wantPlan int
	if _, err := fmt.Sscanf(strings.SplitN(string(b), "\n", 2)[0], "%d %d", &wantRec, &wantPlan); err != nil {
		t.Fatal(err)
	}
	var wantCompared int
	for _, line := range strings.Split(string(b), "\n") {
		if _, err := fmt.Sscanf(line, "# of %d", &wantCompared); err == nil {
			break
		}
	}
	if wantCompared <= 0 || compared == 0 {
		t.Fatalf("baseline or comparison is empty (%d, %d)", wantCompared, compared)
	}
	share := func(n, of int) float64 { return float64(n) / float64(of) }
	t.Logf("of %d cases: %d offered but refused by the planner (baseline %d of %d), %d placed but not offered (baseline %d)", compared, recOnly, wantRec, wantCompared, planOnly, wantPlan)
	const tolerance = 0.01
	if share(recOnly, compared) > share(wantRec, wantCompared)+tolerance || share(planOnly, compared) > share(wantPlan, wantCompared)+tolerance {
		t.Fatalf("recommender/planner disagreement grew: offered-but-refused %d (was %d), placed-but-not-offered %d (was %d); GGRUN_UPDATE_DISAGREEMENT_BASELINE=1 rewrites testdata after a reviewed change:\n%s",
			recOnly, wantRec, planOnly, wantPlan, strings.Join(detail, "\n"))
	}
}
