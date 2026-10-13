package recommend

import "testing"

// main's catalog refresh of 2026-10-13 (written by a builder that still summed
// draft heads) listed MiMo-V2.6-Flash Q4_0/Q8_0/BF16 at 1.26/5.28/9.92 GB: the
// mtp and dflash files, beside the real 167 GB MXFP4. A runtime refresh made
// that Q8_0 the hosted CPU runners' top recommendation.
func TestCompanionSizedQuantsAreDropped(t *testing.T) {
	c := Candidate{Name: "MiMo-V2.6-Flash", Repo: "ggml-org/MiMo-V2.6-Flash-RL-GGUF", Quants: []QuantOption{
		{Name: "Q4_0", SizeBytes: 1264898272}, {Name: "Q8_0", SizeBytes: 5277661632},
		{Name: "BF16", SizeBytes: 9923508672}, {Name: "MXFP4", SizeBytes: 167369465152},
	}}
	got := dropCompanionSizedQuants(c)
	if len(got.Quants) != 1 || got.Quants[0].Name != "MXFP4" {
		t.Fatalf("kept %+v, want only MXFP4", got.Quants)
	}
	// gpt-oss keeps MXFP4 experts in its "BF16" file; it is still the model.
	oss := Candidate{Quants: []QuantOption{{Name: "MXFP4", SizeGB: 63.4}, {Name: "BF16", SizeGB: 65.4}, {Name: "Q2_K", SizeGB: 62.8}}}
	if got := dropCompanionSizedQuants(oss); len(got.Quants) != 3 {
		t.Fatalf("dropped part of gpt-oss: %+v", got.Quants)
	}
	// A normal ladder (Q2 to BF16 of one model) keeps every quant.
	ladder := Candidate{Quants: []QuantOption{{Name: "Q2_K", SizeGB: 10}, {Name: "Q4_K_M", SizeGB: 17}, {Name: "BF16", SizeGB: 54}}}
	if got := dropCompanionSizedQuants(ladder); len(got.Quants) != 3 {
		t.Fatalf("dropped part of a normal ladder: %+v", got.Quants)
	}
	if len(c.Quants) != 4 {
		t.Fatal("the input row was modified")
	}
}
