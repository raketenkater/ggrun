package placement

import "testing"

// mtpYardstick is a synthetic 64 + 1-block delta-net hybrid: 48 recurrent
// blocks of 817,152 f32 state (149.625 MiB per slot) and a 334.75 MiB NextN
// block.
func mtpYardstick() (*ModelProfile, *Strategy) {
	layers := make([]int64, 65)
	for i := range layers {
		layers[i] = 256 << 20
	}
	layers[64] = 351010816
	model := &ModelProfile{
		ModelArch: "qwen35", NumLayers: 65, NextNPredictLayers: 1, HasSSM: 1, FullAttnInterval: 4,
		HeadCountKV: 4, KeyLength: 256, ValueLength: 256, EmbeddingLength: 5120,
		SSMConvKernel: 4, SSMStateSize: 128, SSMGroupCount: 16, SSMInnerSize: 6144, SSMTimeStepRank: 48,
		NonExpertLayerBytes: layers,
	}
	s := &Strategy{ContextSize: 32768, BatchSize: 8192, Parallel: 1, KVPlacement: "gpu",
		Draft: &DraftConfig{Type: DraftMTP, SpecType: "draft-mtp", DraftMax: 2}}
	return model, s
}

func TestEmbeddedMTPExtraReservationTerms(t *testing.T) {
	model, s := mtpYardstick()
	got, err := EmbeddedMTPExtraReservation(model, s)
	if err != nil {
		t.Fatal(err)
	}
	// Rounded up: ceil(334.75), ceil(2 x 149.625), ceil(4 x 5120 x 8193 B).
	if want := (EmbeddedMTPExtra{NextNWeightsMB: 335, RSSnapshotMB: 300, HNextnMB: 161}); got != want {
		t.Fatalf("terms %+v, want %+v", got, want)
	}

	// No --spec-draft-n-max: llama.cpp's default n_max of 3. Two slots double
	// the snapshots and the h_nextn rows grow by one.
	s.Draft.DraftMax, s.Parallel = 0, 2
	if got, err = EmbeddedMTPExtraReservation(model, s); err != nil || got.RSSnapshotMB != 898 || got.HNextnMB != 161 {
		t.Fatalf("default n_max, two slots: %+v %v", got, err)
	}
	// A context below the batch caps n_batch.
	s.ContextSize = 4096
	if got, _ = EmbeddedMTPExtraReservation(model, s); got.HNextnMB != 81 {
		t.Fatalf("h_nextn at ctx 4096: %d", got.HNextnMB)
	}
}

// Recurrent state is charged whatever the architecture: whether a build
// rolls it back is not known here, and zero is the unsafe direction. A model
// without recurrent state gets no snapshot term.
func TestEmbeddedMTPExtraReservationRecurrentStateWhateverTheArch(t *testing.T) {
	model, s := mtpYardstick()
	model.ModelArch = "qwen3next" // a build may clamp n_rs_seq to 0 for it
	if got, err := EmbeddedMTPExtraReservation(model, s); err != nil || got.RSSnapshotMB != 300 {
		t.Fatalf("recurrent arch outside a rollback list: %+v %v", got, err)
	}
	model, s = mtpYardstick()
	model.ModelArch, model.HasSSM = "llama", 0
	if got, err := EmbeddedMTPExtraReservation(model, s); err != nil || got.RSSnapshotMB != 0 || got.NextNWeightsMB != 335 {
		t.Fatalf("attention-only model: %+v %v", got, err)
	}
}

// Every term that cannot be derived is an error, never a zero charge.
func TestEmbeddedMTPExtraReservationUnknownIsAnError(t *testing.T) {
	for name, change := range map[string]func(*ModelProfile, *Strategy){
		"no NextN block":           func(m *ModelProfile, _ *Strategy) { m.NextNPredictLayers = 0 },
		"no per-block bytes":       func(m *ModelProfile, _ *Strategy) { m.NonExpertLayerBytes = m.NonExpertLayerBytes[:10] },
		"empty NextN block":        func(m *ModelProfile, _ *Strategy) { m.NonExpertLayerBytes[64] = 0 },
		"MoE without expert bytes": func(m *ModelProfile, _ *Strategy) { m.IsMoE = true },
		"recurrent state, unsized": func(m *ModelProfile, _ *Strategy) { m.ModelArch = "nemotron_h" },
		"no embedding length":      func(m *ModelProfile, _ *Strategy) { m.EmbeddingLength = 0 },
		"no MTP plan":              func(_ *ModelProfile, s *Strategy) { s.Draft = nil },
	} {
		model, s := mtpYardstick()
		change(model, s)
		if got, err := EmbeddedMTPExtraReservation(model, s); err == nil {
			t.Errorf("%s: no error, terms %+v", name, got)
		}
	}
}

// The admission-rows component is part of the key only when set.
func TestCalibrationScopeKeyAdmissionRowsAppendedOnlyWhenSet(t *testing.T) {
	base := CalibrationScopeKey{ModelIdentity: "m", BackendIdentity: "b", HardwareID: "h", WorkloadProfile: "parallel"}
	gated := base
	gated.AdmissionRows = "noncuda-v1"
	if gated.String() == base.String() {
		t.Fatal("the gated scope must differ")
	}
	again := base
	again.AdmissionRows = ""
	if again.String() != base.String() {
		t.Fatal("an empty component changed the key")
	}
}
