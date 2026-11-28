package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/devicelist"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// Memory admission for backends whose oracle rows are not CUDA rows.
//
// The no-alloc oracle prints one row per backend device, named by the
// backend's own ordinal ("Vulkan1"). cudaDeviceIndex reads only "CUDA<n>", so
// a Vulkan launch compared no device at all ("placement fits ()") and embedded
// MTP was always refused. On the gated path below every row is mapped to the
// launch's device through one table built from the backend's own listing
// under the launch's exact environment, each device's free memory comes from
// readers that say when they do not know, and anything unmapped or unknown
// refuses the attempt.
//
// The gate (nonCUDAAdmissionApplies) is GGRUN_EXPERIMENTAL_AMD=1 on Linux, a
// non-CUDA backend dialect, and an OS inventory with GPUs of more than one
// vendor. Every host without the opt-in, and every CUDA-family backend, reads
// nothing new. With the opt-in and a Vulkan backend each gate check reads the
// sysfs PCI inventory; on a single-vendor host that read is all, and the
// launch is today's byte for byte (no listing, no reader).

// admissionRowsScopeVersion is the calibration and verified-config scope of
// a gated launch (placement.CalibrationScopeKey.AdmissionRows).
const admissionRowsScopeVersion = "noncuda-v1"

// admissionRowPrefix is the device-name prefix the oracle and the listing use
// for a non-CUDA dialect, or "" for the CUDA-family dialects cudaDeviceIndex
// already reads. ROCm is set only by an AMD --gpus selection (amd_select.go).
func admissionRowPrefix(be *backendInfo) string {
	if be != nil && strings.EqualFold(be.Dialect, "vulkan") {
		return "Vulkan"
	}
	if be != nil && strings.EqualFold(be.Dialect, "rocm") {
		return "ROCm"
	}
	return ""
}

// nonCUDAAdmissionApplies reports whether a launch takes the mapped-row
// admission. The opt-in and the dialect are checked before the inventory is
// read. Vulkan is the native API of no vendor group, so for it the design's
// third condition (a device outside the primary group, or a backend API that
// is not the primary group's) always holds. ROCm is the AMD group's own API,
// so it applies only with an AMD card beside a GPU of another vendor: caps
// holds a GPU the inventory names AMD at its PCI address. The answer does not
// depend on which other GPUs caps holds, so the detection-time and runtime
// capabilities agree.
func nonCUDAAdmissionApplies(be *backendInfo, caps *detect.Capabilities, inventory func() []detect.PCIDisplay) bool {
	if be == nil || caps == nil || len(caps.GPUs) == 0 || inventory == nil || !detect.ExperimentalAMD() {
		return false
	}
	prefix := admissionRowPrefix(be)
	if prefix == "" {
		return false
	}
	inv := inventory()
	if inventoryGPUVendors(inv) < 2 {
		return false
	}
	return prefix != "ROCm" || capsHoldVendor(caps, inv, "amd")
}

// capsHoldVendor reports whether a GPU of caps is a card of vendor in the
// inventory, matched by PCI address.
func capsHoldVendor(caps *detect.Capabilities, inv []detect.PCIDisplay, vendor string) bool {
	for _, g := range caps.GPUs {
		address, ok := detect.CanonicalPCIAddress(g.PCIBusID)
		if !ok {
			continue
		}
		for _, d := range inv {
			if d.Address == address && d.Vendor == vendor {
				return true
			}
		}
	}
	return false
}

// inventoryGPUVendors counts the GPU vendors a backend could open: a known
// vendor with a bound driver that is not a VM passthrough. A BMC display (an
// unknown vendor) does not count.
func inventoryGPUVendors(inv []detect.PCIDisplay) int {
	vendors := map[string]bool{}
	for _, d := range inv {
		known := d.Vendor == "nvidia" || d.Vendor == "amd" || d.Vendor == "intel"
		if known && d.Driver != "" && d.Driver != "vfio-pci" {
			vendors[d.Vendor] = true
		}
	}
	return len(vendors)
}

// admissionRowsScope is the scope component of a gated launch, or "".
func admissionRowsScope(be *backendInfo, caps *detect.Capabilities) string {
	if nonCUDAAdmissionApplies(be, caps, hostPCIDisplays) {
		return admissionRowsScopeVersion
	}
	return ""
}

// verifiedScopeKeyString is the verified-config key, the same at lookup and
// save.
func verifiedScopeKeyString(model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, opts placement.Options) string {
	key := placement.NewCalibrationScopeKey(model, caps, opts, nil)
	key.AdmissionRows = admissionRowsScope(be, caps)
	return key.String() + "-plan" + planLogicVersion
}

// admissionUnknownError is a gated refusal for want of proof (an unverified
// device order, unknown free memory, a missing or failed oracle, a row set
// that does not match the map), not a measured deficit. It says nothing
// about a reused verified config, which therefore is kept.
type admissionUnknownError struct{ err error }

func (e *admissionUnknownError) Error() string { return e.err.Error() }
func (e *admissionUnknownError) Unwrap() error { return e.err }

func admissionUnknown(err error) error {
	if err == nil {
		return nil
	}
	return &admissionUnknownError{err: err}
}

// verifiedConfigStaleAfter reports whether a launch failure is evidence
// against the verified config it reused: every failure except a gated
// refusal for want of proof.
func verifiedConfigStaleAfter(err error) bool {
	var unknown *admissionUnknownError
	return err != nil && !errors.As(err, &unknown)
}

// postLaunchSystemProbesApply reports whether the MoE post-launch system
// overhead and VRAM-delta probes may run: they read nvidia-smi and CUDA log
// tags into a cache keyed by hardware only, which a gated launch must not
// write.
func postLaunchSystemProbesApply(be *backendInfo, caps *detect.Capabilities) bool {
	return !nonCUDAAdmissionApplies(be, caps, hostPCIDisplays)
}

// stagedPrefillRaiseApplies reports whether the staged-prefill microbatch
// raise may be tried: it prices its rungs with CUDA rows only, so not on the
// gated path.
func stagedPrefillRaiseApplies(be *backendInfo, runtimeCaps *detect.Capabilities) bool {
	return !nonCUDAAdmissionApplies(be, runtimeCaps, hostPCIDisplays)
}

// admissionDevice is one device of a gated launch.
type admissionDevice struct {
	Row     string // the backend's name, prefix + ordinal: "Vulkan1"
	View    int    // the launch's GPU index (runtime caps): placement, -ts, --device
	Name    string
	TotalMB int
	PCI     string // canonical; "" when detection reported none (vulkaninfo)
	Vendor  string // the inventory's vendor at PCI; "" when unknown
	FreeOK  bool
	FreeMB  int      // the admission figure: the minimum over every ok reader
	Sources []string // "reader=MiB" for each ok reading
	Reasons []string // why a reader gave no reading
}

// admissionRows maps the oracle's rows to the launch's devices for one
// admission attempt. A nil *admissionRows is today's CUDA mapping.
type admissionRows struct {
	prefix  string
	devices []admissionDevice // in ordinal order, which is view order
}

func (r *admissionRows) device(name string) (admissionDevice, bool) {
	if r == nil {
		return admissionDevice{}, false
	}
	for _, d := range r.devices {
		if d.Row == name {
			return d, true
		}
	}
	return admissionDevice{}, false
}

// index is the view index an oracle row names.
func (r *admissionRows) index(name string) (int, bool) {
	if r == nil {
		return cudaDeviceIndex(name)
	}
	d, ok := r.device(name)
	return d.View, ok
}

// freeMB is the free memory admission charges a view GPU against. Gated, it
// is 0 for a device without an ok reading; checkMemory refuses those first.
func (r *admissionRows) freeMB(g detect.GPU) int {
	if r == nil {
		return g.VRAMFreeMB()
	}
	for _, d := range r.devices {
		if d.View == g.Index && d.FreeOK {
			return d.FreeMB
		}
	}
	return 0
}

// overheadByGPU is the per-device runtime overhead reserve. A measured CUDA
// overhead is not evidence for another backend on the same card (the system
// cache is keyed by hardware only), so a gated device keeps the unmeasured
// reserve.
func (r *admissionRows) overheadByGPU(cacheDir string, gpus []detect.GPU) map[int]int {
	if r == nil {
		return placement.PlanningCUDAOverheadByGPU(cacheDir, gpus)
	}
	out := map[int]int{}
	for _, g := range gpus {
		out[g.Index] = placement.UnmeasuredCUDAOverheadMB
	}
	return out
}

// checkRows requires the oracle's rows to match the map: one row for every
// mapped device (0/0/0 is fine, absence is not: a missing row would compare
// nothing and admit, as "fits ()" did), and no nonzero row that names no
// device of the map (an empty one is an idle device and is ignored).
func (r *admissionRows) checkRows(devs []preflightDevice) error {
	if r == nil {
		return nil
	}
	for _, m := range r.devices {
		n := 0
		for _, d := range devs {
			if d.Name == m.Row {
				n++
			}
		}
		if n != 1 {
			return fmt.Errorf("the memory oracle printed %d rows for %s (GPU %d), want exactly one (map: %s); refusing", n, m.Row, m.View, r.describe())
		}
	}
	for _, d := range devs {
		if d.Name == "Host" || d.TotalMB() == 0 {
			continue
		}
		if _, ok := r.device(d.Name); !ok {
			return fmt.Errorf("the memory oracle printed device row %s that this launch's device map does not contain (map: %s); refusing", d.Name, r.describe())
		}
	}
	return nil
}

// checkMemory refuses a device row whose free memory is unknown.
func (r *admissionRows) checkMemory(devs []preflightDevice) error {
	if r == nil {
		return nil
	}
	for _, d := range devs {
		dev, ok := r.device(d.Name)
		if !ok || dev.FreeOK {
			continue
		}
		pci := dev.PCI
		if pci == "" {
			pci = "unknown"
		}
		return fmt.Errorf("free memory of %s (%s, PCI %s) is unknown: %s", dev.Row, dev.Name, pci, strings.Join(dev.Reasons, "; "))
	}
	return nil
}

func (r *admissionRows) describe() string {
	parts := make([]string, 0, len(r.devices))
	for _, d := range r.devices {
		parts = append(parts, fmt.Sprintf("%s=GPU %d", d.Row, d.View))
	}
	return strings.Join(parts, ", ")
}

// summary is one line per launch attempt: each row's device and the readings
// behind its free figure.
func (r *admissionRows) summary() string {
	parts := make([]string, 0, len(r.devices))
	for _, d := range r.devices {
		pci := d.PCI
		if pci == "" {
			pci = "PCI unknown"
		}
		free := "free unknown"
		if d.FreeOK {
			free = fmt.Sprintf("free %d MiB", d.FreeMB)
		}
		readings := append(append([]string(nil), d.Sources...), d.Reasons...)
		parts = append(parts, fmt.Sprintf("%s = GPU %d (%s, %s): %s [%s]", d.Row, d.View, d.Name, pci, free, strings.Join(readings, "; ")))
	}
	return strings.Join(parts, ", ")
}

// admissionListing is the backend's own listing under this process's
// environment, which is the launch's (the visibility export is already set),
// never served from the cache: its free figures are read.
var admissionListing = func(path string) (string, error) {
	res := devicelist.Run(devicelist.Probe{Binary: path, Timeout: devicelist.DefaultTimeout, NoCache: true})
	if res.TimedOut {
		return "", fmt.Errorf("%s --list-devices timed out", path)
	}
	if res.Err != nil {
		return "", fmt.Errorf("%s --list-devices: %w", path, res.Err)
	}
	return res.Output, nil
}

// amdgpuSysfsRoot is the file-system root of the amdgpu memory reader (a
// test seam).
var amdgpuSysfsRoot = "/"

// listedMemory is one "  <Prefix><k>: <description> (T MiB, F MiB free)"
// line of --list-devices (llama.cpp common/arg.cpp).
type listedMemory struct {
	Ordinal int
	Name    string
	TotalMB int
	FreeMB  int
}

func parseListedMemory(prefix, out string) []listedMemory {
	line := regexp.MustCompile(`^\s*` + regexp.QuoteMeta(prefix) + `(\d+):\s*(.+?)\s*\((\d+) MiB, (\d+) MiB free\)\s*$`)
	var listed []listedMemory
	for _, text := range strings.Split(out, "\n") {
		m := line.FindStringSubmatch(text)
		if m == nil {
			continue
		}
		ordinal, _ := strconv.Atoi(m[1])
		total, _ := strconv.Atoi(m[3])
		free, _ := strconv.Atoi(m[4])
		listed = append(listed, listedMemory{Ordinal: ordinal, Name: m[2], TotalMB: total, FreeMB: free})
	}
	return listed
}

// admissionTotalMatches allows the spread APIs report for one card (NVML and
// Vulkan differ by a few hundred MiB), as matchBackendDevices does.
func admissionTotalMatches(reported, detected int) bool {
	diff := reported - detected
	if diff < 0 {
		diff = -diff
	}
	return diff <= detected/10+256
}

// amdgpuVRAMFree reads whole-card VRAM from amdgpu sysfs: every process's
// use, floored to MiB free. A card that is runtime-suspended (or changing
// state) is not read: the read would wake it.
func amdgpuVRAMFree(pci string) (freeMB, totalMB int, reason string, ok bool) {
	dir := filepath.Join(amdgpuSysfsRoot, "sys", "bus", "pci", "devices", pci)
	read := func(name string) (int64, bool) {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return 0, false
		}
		v, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
		return v, err == nil
	}
	if data, err := os.ReadFile(filepath.Join(dir, "power", "runtime_status")); err == nil {
		// "unsupported" means runtime PM is off: the card never suspends.
		switch status := strings.TrimSpace(string(data)); status {
		case "suspended", "suspending", "resuming", "error":
			return 0, 0, "sysfs not read: the card is runtime-" + status, false
		}
	}
	total, okTotal := read("mem_info_vram_total")
	used, okUsed := read("mem_info_vram_used")
	if !okTotal || !okUsed {
		return 0, 0, "sysfs mem_info_vram_total/used unreadable", false
	}
	if total <= 0 || used < 0 || used > total {
		return 0, 0, fmt.Sprintf("sysfs reports %d of %d bytes used", used, total), false
	}
	return int((total - used) >> 20), int(total >> 20), "", true
}

// buildAdmissionRows lists the backend under the launch's environment and
// builds the gated map. The listing must show exactly the launch's GPUs as
// <prefix>0..n-1 in view order, with their names and totals; otherwise the
// order is unverified and the attempt is refused. Free memory per device is
// the minimum of every ok reader:
//   - nvml: an NVIDIA card's detected figure (nvidia-smi, refreshed at the
//     launch boundary, net of seated companions), only with a PCI address
//     the inventory names as NVIDIA;
//   - listing: the backend's free figure, only when it is below the total
//     (Vulkan reports the whole heap as free without VK_EXT_memory_budget);
//   - sysfs: amdgpu mem_info_vram_*, only for a card the inventory names as
//     AMD at its PCI address.
//
// baselineServing marks a calibration prescreen while the baseline still
// holds its memory: the live listing and sysfs figures would count it, so
// they are not used and a device without another reading is unknown.
func buildAdmissionRows(be *backendInfo, caps *detect.Capabilities, inventory []detect.PCIDisplay, baselineServing bool) (*admissionRows, error) {
	rows, err := buildAdmissionRowsChecked(be, caps, inventory, baselineServing)
	if err != nil {
		return nil, fmt.Errorf("%v; select the launch's devices with --gpus, or unset GGRUN_EXPERIMENTAL_AMD for the unchecked path", err)
	}
	return rows, nil
}

func buildAdmissionRowsChecked(be *backendInfo, caps *detect.Capabilities, inventory []detect.PCIDisplay, baselineServing bool) (*admissionRows, error) {
	prefix := admissionRowPrefix(be)
	gpus := append([]detect.GPU(nil), caps.GPUs...)
	sort.Slice(gpus, func(i, j int) bool { return gpus[i].Index < gpus[j].Index })
	for pos, g := range gpus {
		if g.Index != pos {
			return nil, fmt.Errorf("%s device order could not be verified: the launch's GPU indexes are not 0..%d", prefix, len(gpus)-1)
		}
	}
	out, err := admissionListing(be.Path)
	if err != nil {
		return nil, fmt.Errorf("%s device order could not be verified: %v", prefix, err)
	}
	listed := parseListedMemory(prefix, out)
	if len(listed) != len(gpus) {
		return nil, fmt.Errorf("%s device order could not be verified: the backend lists %d %s devices under this launch's environment, the plan uses %d", prefix, len(listed), prefix, len(gpus))
	}
	vendorAt := map[string]string{}
	for _, d := range inventory {
		vendorAt[d.Address] = d.Vendor
	}
	rows := &admissionRows{prefix: prefix}
	for pos, g := range gpus {
		l := listed[pos]
		if l.Ordinal != pos || !strings.EqualFold(strings.TrimSpace(l.Name), strings.TrimSpace(g.Name)) || !admissionTotalMatches(l.TotalMB, g.VRAMTotalMB) {
			return nil, fmt.Errorf("%s device order could not be verified: %s%d is %s (%d MiB), the plan expects %s (%d MiB) there",
				prefix, prefix, l.Ordinal, l.Name, l.TotalMB, g.Name, g.VRAMTotalMB)
		}
		d := admissionDevice{Row: prefix + strconv.Itoa(pos), View: g.Index, Name: g.Name, TotalMB: g.VRAMTotalMB}
		if pci, ok := detect.CanonicalPCIAddress(g.PCIBusID); ok {
			d.PCI, d.Vendor = pci, vendorAt[pci]
		}
		var readings []int
		reading := func(source string, mb int) {
			d.Sources = append(d.Sources, fmt.Sprintf("%s=%d", source, mb))
			readings = append(readings, mb)
		}
		if d.Vendor == "nvidia" {
			reading("nvml", g.VRAMFreeMB())
		}
		if baselineServing {
			d.Reasons = append(d.Reasons, "the live listing and sysfs figures would count the serving baseline")
		} else {
			if l.FreeMB < l.TotalMB {
				reading("listing", l.FreeMB)
			} else {
				d.Reasons = append(d.Reasons, "the backend listing reports the whole heap as free (no memory budget)")
			}
			if d.Vendor == "amd" {
				free, total, why, ok := amdgpuVRAMFree(d.PCI)
				switch {
				case !ok:
					d.Reasons = append(d.Reasons, why)
				case !admissionTotalMatches(total, g.VRAMTotalMB):
					d.Reasons = append(d.Reasons, fmt.Sprintf("sysfs total %d MiB does not match the card's %d MiB", total, g.VRAMTotalMB))
				default:
					reading("sysfs", free)
				}
			}
		}
		if d.PCI == "" {
			d.Reasons = append(d.Reasons, "no PCI address, so no NVML or sysfs reading")
		}
		if len(readings) > 0 {
			d.FreeOK = true
			d.FreeMB = g.VRAMFreeMB() // never above detection's figure, which carries companion reservations
			for _, mb := range readings {
				d.FreeMB = min(d.FreeMB, mb)
			}
		}
		rows.devices = append(rows.devices, d)
	}
	rows.chargeIndistinguishable()
	return rows, nil
}

// chargeIndistinguishable charges each of two or more cards that the listing
// cannot tell apart (same name, totals within the order check's tolerance) against the smallest
// free figure of the set: which ordinal is which card is not proven before
// load, so a swap must not admit a plan against the emptier card. One member
// without an ok reading leaves the whole set unknown.
func (r *admissionRows) chargeIndistinguishable() {
	// The same tolerance the order check accepts: two cards it cannot tell
	// apart there are not told apart here either.
	alike := func(a, b admissionDevice) bool {
		return strings.EqualFold(strings.TrimSpace(a.Name), strings.TrimSpace(b.Name)) &&
			(admissionTotalMatches(a.TotalMB, b.TotalMB) || admissionTotalMatches(b.TotalMB, a.TotalMB))
	}
	out := make([]admissionDevice, len(r.devices))
	copy(out, r.devices)
	for i, a := range r.devices {
		for j, b := range r.devices {
			if i == j || !alike(a, b) {
				continue
			}
			switch {
			case !b.FreeOK:
				out[i].FreeOK = false
				out[i].Reasons = append(out[i].Reasons, fmt.Sprintf("indistinguishable from %s, whose free memory is unknown", b.Row))
			case out[i].FreeOK && b.FreeMB < out[i].FreeMB:
				out[i].FreeMB = b.FreeMB
				out[i].Sources = append(out[i].Sources, fmt.Sprintf("indistinguishable-from-%s=%d", b.Row, b.FreeMB))
			}
		}
	}
	r.devices = out
}

// nonCUDAEmbeddedMTPReservation is the embedded-MTP charge for gated rows:
// today's KV and compute terms plus the three the oracle cannot price
// (placement.EmbeddedMTPExtraReservation). All terms are charged on every
// device row: which device holds the NextN block, the recurrent layers and
// the output is not proven here, so a multi-device plan over-reserves.
func nonCUDAEmbeddedMTPReservation(adm *admissionRows, model *placement.ModelProfile, strategy *placement.Strategy, target []preflightDevice, contextMB, computeFloorMB int) ([]preflightDevice, error) {
	extra, err := placement.EmbeddedMTPExtraReservation(model, strategy)
	if err != nil {
		return nil, fmt.Errorf("embedded MTP memory terms are unknown (%v); rerun with --spec off to skip this check", err)
	}
	var rows []preflightDevice
	var terms []string
	for _, d := range target {
		if _, ok := adm.device(d.Name); !ok {
			continue
		}
		computeMB := max(d.ComputeMB, computeFloorMB)
		// The recurrent snapshots and h_nextn do not scale with context, so
		// they stay out of ContextMB, which recovery prices per token.
		rows = append(rows, preflightDevice{
			Name:          d.Name,
			ModelMB:       extra.NextNWeightsMB,
			ContextMB:     contextMB,
			ComputeMB:     computeMB,
			UnaccountedMB: extra.RSSnapshotMB + extra.HNextnMB,
		})
		terms = append(terms, fmt.Sprintf("%s kv=%d compute=%d nextn=%d rs=%d h_nextn=%d", d.Name, contextMB, computeMB, extra.NextNWeightsMB, extra.RSSnapshotMB, extra.HNextnMB))
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("embedded MTP has no measured %s device rows; rerun with --spec off to skip this check", adm.prefix)
	}
	fmt.Fprintf(os.Stderr, "[launch] embedded MTP reservation: %s MiB\n", strings.Join(terms, "; "))
	return rows, nil
}
