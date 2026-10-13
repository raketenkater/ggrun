package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// Runtime graph growth is the VRAM a backend gains after load, from real
// requests. The post-launch recorder runs right after load, before any
// request, so it measured next to nothing; growth was only ever learned from
// crashes, as estimates. Measured on Qwen3.6-35B-A3B on one RTX 4070: 11,685
// MiB after load, 11,857 MiB at a 28k-token prompt, an abort when an earlier
// request had already pushed the pool to the card's 11,871 MiB.
//
// The serving recorder samples the backend process tree's own VRAM on each
// device (so companions and other applications are not counted) against its
// snapshot right after load, keeps the peak, and files it as measured growth
// for the launch key. Measured growth is carried to related context/ubatch
// keys, so later plans reserve it without having to crash first.

const (
	servingGrowthSampleEvery = 10 * time.Second
	servingGrowthFlushEvery  = time.Minute
	// Growth below this is pool noise, not a reserve worth re-planning for.
	servingGrowthMinMB = 32
	// A new record must exceed the last by this much.
	servingGrowthStepMB = 32
)

var loadedBackendVRAM = struct {
	sync.Mutex
	byPID map[int]map[int]int
}{byPID: map[int]map[int]int{}}

// noteBackendLoadedVRAM snapshots a freshly loaded backend's own VRAM per
// physical device, before any request reaches it.
func noteBackendLoadedVRAM(root int) {
	if root <= 0 {
		return
	}
	snapshot := backendTreeVRAMByPhysicalGPU(root)
	if len(snapshot) == 0 {
		return
	}
	loadedBackendVRAM.Lock()
	loadedBackendVRAM.byPID[root] = snapshot
	loadedBackendVRAM.Unlock()
}

func loadedBackendVRAMFor(root int) map[int]int {
	loadedBackendVRAM.Lock()
	defer loadedBackendVRAM.Unlock()
	return loadedBackendVRAM.byPID[root]
}

type servingGrowthRecorder struct {
	mu       sync.Mutex
	loaded   map[int]int // physical index -> MiB right after load
	peak     map[int]int // runtime index -> peak growth MiB
	recorded map[int]int // runtime index -> last recorded MiB
	physical map[int]int // runtime index -> physical index
	sample   func() map[int]int
	record   func(map[int]int) error
	stopCh   chan struct{}
	done     chan struct{}
}

// startServingGrowthRecorder begins sampling a serving backend. It returns nil
// when there is nothing to measure (CPU launch, no snapshot, no telemetry).
func startServingGrowthRecorder(cacheDir string, model *placement.ModelProfile, strategy *placement.Strategy, tag string,
	runtime *detect.Capabilities, visibleToPhysical map[int]int, root int,
) *servingGrowthRecorder {
	if cacheDir == "" || model == nil || strategy == nil || runtime == nil || len(runtime.GPUs) == 0 || root <= 0 {
		return nil
	}
	loaded := loadedBackendVRAMFor(root)
	if len(loaded) == 0 {
		return nil
	}
	gpus := runtime.GPUs
	r := newServingGrowthRecorder(loaded, gpus, visibleToPhysical,
		func() map[int]int { return backendTreeVRAMByPhysicalGPU(root) },
		func(growth map[int]int) error {
			return placement.RecordRuntimeGraphGrowth(cacheDir, model, strategy.ContextSize, strategy.UBatchSize,
				strategy.KVQuality, strategy.KVPlacement, tag, gpus, strategy.Parallel, growth)
		})
	go r.run(servingGrowthSampleEvery, servingGrowthFlushEvery)
	return r
}

func newServingGrowthRecorder(loaded map[int]int, gpus []detect.GPU, visibleToPhysical map[int]int,
	sample func() map[int]int, record func(map[int]int) error,
) *servingGrowthRecorder {
	physical := map[int]int{}
	for _, g := range gpus {
		phys := g.Index
		if mapped, ok := visibleToPhysical[g.Index]; ok {
			phys = mapped
		}
		physical[g.Index] = phys
	}
	return &servingGrowthRecorder{
		loaded: loaded, peak: map[int]int{}, recorded: map[int]int{}, physical: physical,
		sample: sample, record: record, stopCh: make(chan struct{}), done: make(chan struct{}),
	}
}

func (r *servingGrowthRecorder) run(sampleEvery, flushEvery time.Duration) {
	defer close(r.done)
	sampleTick := time.NewTicker(sampleEvery)
	defer sampleTick.Stop()
	flushTick := time.NewTicker(flushEvery)
	defer flushTick.Stop()
	for {
		select {
		case <-r.stopCh:
			r.observe()
			r.flush()
			return
		case <-sampleTick.C:
			r.observe()
		case <-flushTick.C:
			r.flush()
		}
	}
}

// observe takes one sample and keeps the per-device peak growth.
func (r *servingGrowthRecorder) observe() {
	current := r.sample()
	r.mu.Lock()
	defer r.mu.Unlock()
	for idx, phys := range r.physical {
		base, ok := r.loaded[phys]
		if !ok {
			continue
		}
		now, ok := current[phys]
		if !ok || now <= base {
			continue
		}
		if growth := now - base; growth > r.peak[idx] {
			r.peak[idx] = growth
		}
	}
}

// flush files peaks that grew past the last record. It never records less.
func (r *servingGrowthRecorder) flush() {
	r.mu.Lock()
	out := map[int]int{}
	for idx, peak := range r.peak {
		if peak >= servingGrowthMinMB && peak >= r.recorded[idx]+servingGrowthStepMB {
			out[idx] = peak
		}
	}
	r.mu.Unlock()
	if len(out) == 0 {
		return
	}
	if err := r.record(out); err != nil {
		fmt.Fprintf(os.Stderr, "[serve] could not record runtime growth: %v\n", err)
		return
	}
	r.mu.Lock()
	parts := make([]string, 0, len(out))
	for idx, mb := range out {
		r.recorded[idx] = mb
		parts = append(parts, fmt.Sprintf("CUDA%d=%d MiB", idx, mb))
	}
	r.mu.Unlock()
	fmt.Fprintf(os.Stderr, "[serve] recorded measured runtime growth %s; the next launch plans with it\n", strings.Join(parts, ", "))
}

// stop takes a last sample and flushes. Safe on nil and on repeat calls.
func (r *servingGrowthRecorder) stop() {
	if r == nil {
		return
	}
	select {
	case <-r.stopCh:
	default:
		close(r.stopCh)
	}
	<-r.done
}

// backendTreeVRAMByPhysicalGPU sums the VRAM nvidia-smi attributes to the
// processes under root, per physical GPU index. The launched pid is a scope
// wrapper; the backend is its child, and an abort can leave a forked helper.
func backendTreeVRAMByPhysicalGPU(root int) map[int]int {
	pids := processTreePIDs(root)
	if len(pids) == 0 {
		return nil
	}
	indexByUUID := map[string]int{}
	out, err := exec.Command("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader").Output()
	if err != nil {
		return nil
	}
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Split(line, ",")
		if len(f) != 2 {
			continue
		}
		if idx, err := strconv.Atoi(strings.TrimSpace(f[0])); err == nil {
			indexByUUID[strings.TrimSpace(f[1])] = idx
		}
	}
	apps, err := exec.Command("nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	used := map[int]int{}
	for _, line := range strings.Split(string(apps), "\n") {
		f := strings.Split(line, ",")
		if len(f) != 3 {
			continue
		}
		pid, err1 := strconv.Atoi(strings.TrimSpace(f[0]))
		mb, err2 := strconv.Atoi(strings.TrimSpace(f[2]))
		idx, known := indexByUUID[strings.TrimSpace(f[1])]
		if err1 != nil || err2 != nil || !known || !pids[pid] {
			continue
		}
		used[idx] += mb
	}
	return used
}

// processTreePIDs returns root and its descendants (Linux /proc). Elsewhere it
// returns nil and the recorder stays off.
func processTreePIDs(root int) map[int]bool {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil
	}
	parent := map[int]int{}
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		data, err := os.ReadFile(filepath.Join("/proc", e.Name(), "stat"))
		if err != nil {
			continue
		}
		stat := string(data)
		end := strings.LastIndexByte(stat, ')')
		if end < 0 {
			continue
		}
		fields := strings.Fields(stat[end+1:])
		if len(fields) < 2 {
			continue
		}
		if ppid, err := strconv.Atoi(fields[1]); err == nil {
			parent[pid] = ppid
		}
	}
	if _, ok := parent[root]; !ok {
		return nil
	}
	tree := map[int]bool{}
	for pid := range parent {
		for cur, hops := pid, 0; cur > 0 && hops < 64; cur, hops = parent[cur], hops+1 {
			if cur == root {
				tree[pid] = true
				break
			}
		}
	}
	return tree
}
