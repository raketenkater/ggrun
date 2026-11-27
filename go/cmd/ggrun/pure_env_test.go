package main

import (
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The launch planner needs the GPU visibility and Vulkan device order as
// values. These tests pin that the pure helpers return exactly what the setters
// export, touch no process state, and that the setters keep their old return
// values, side effects and printed text. Expected values are literals read off
// the code before the split.

var pureEnvKeys = []string{"CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES"}

func sortedEnviron() []string {
	env := os.Environ()
	sort.Strings(env)
	return env
}

// environAdded is what after has that before lacks, sorted.
func environAdded(before, after []string) []string {
	had := map[string]bool{}
	for _, kv := range before {
		had[kv] = true
	}
	var added []string
	for _, kv := range after {
		if !had[kv] {
			added = append(added, kv)
		}
	}
	return added
}

func environRemoved(before, after []string) []string {
	return environAdded(after, before)
}

func assertOnlyAssignment(t *testing.T, name string, before, after []string, want string) {
	t.Helper()
	var wantAdded []string
	if want != "" {
		wantAdded = []string{want}
	}
	if added := environAdded(before, after); !reflect.DeepEqual(added, wantAdded) {
		t.Fatalf("%s: environment gained %q, want %q", name, added, wantAdded)
	}
	if removed := environRemoved(before, after); len(removed) != 0 {
		t.Fatalf("%s: environment lost %q", name, removed)
	}
}

func TestGPUVisibilityEnvIsPureAndMatchesSetter(t *testing.T) {
	cases := []struct {
		name  string
		req   *launchRequest
		tag   string
		key   string
		value string
	}{
		{"nil request", nil, "llama", "", ""},
		{"no restriction", &launchRequest{}, "llama", "", ""},
		{"cpu mode on a cuda host", &launchRequest{CPUMode: true}, "llama", "CUDA_VISIBLE_DEVICES", "-1"},
		{"cpu mode wins over --gpus", &launchRequest{CPUMode: true, GPUsFlag: "1"}, "ik_llama", "CUDA_VISIBLE_DEVICES", "-1"},
		{"cpu mode on vulkan without --gpus", &launchRequest{CPUMode: true}, "vulkan", "", ""},
		{"cpu mode on vulkan with --gpus", &launchRequest{CPUMode: true, GPUsFlag: "1"}, "vulkan", "GGML_VK_VISIBLE_DEVICES", "1"},
		{"--gpus single", &launchRequest{GPUsFlag: "1"}, "llama", "CUDA_VISIBLE_DEVICES", "1"},
		{"--gpus several", &launchRequest{GPUsFlag: "0,1,2"}, "llama", "CUDA_VISIBLE_DEVICES", "0,1,2"},
		{"--gpus not from 0", &launchRequest{GPUsFlag: "1,2"}, "llama", "CUDA_VISIBLE_DEVICES", "1,2"},
		{"--gpus out of order", &launchRequest{GPUsFlag: "2,0"}, "ik_llama", "CUDA_VISIBLE_DEVICES", "0,2"},
		{"--gpus with spaces", &launchRequest{GPUsFlag: " 2 , 1 "}, "llama", "CUDA_VISIBLE_DEVICES", "1,2"},
		{"vulkan --gpus", &launchRequest{GPUsFlag: "2,1"}, "Vulkan", "GGML_VK_VISIBLE_DEVICES", "1,2"},
		{"--gpus blank", &launchRequest{GPUsFlag: "  "}, "llama", "", ""},
		{"--gpus invalid", &launchRequest{GPUsFlag: "abc"}, "llama", "", ""},
		{"--gpus negative", &launchRequest{GPUsFlag: "-1"}, "llama", "", ""},
		{"--gpus duplicate", &launchRequest{GPUsFlag: "1,1"}, "llama", "", ""},
	}
	for _, c := range cases {
		goldenPlanUnsetenv(t, pureEnvKeys...)
		before := sortedEnviron()
		key, value := gpuVisibilityEnv(c.req, c.tag)
		if key != c.key || value != c.value {
			t.Fatalf("%s: gpuVisibilityEnv = %q, %q; want %q, %q", c.name, key, value, c.key, c.value)
		}
		if after := sortedEnviron(); !reflect.DeepEqual(before, after) {
			t.Fatalf("%s: the pure call changed the environment", c.name)
		}

		want := ""
		if c.key != "" {
			want = c.key + "=" + c.value
		}
		out := captureStdout(t)
		got := applyGPUVisibility(c.req, c.tag)
		if printed := out(); printed != "" {
			t.Fatalf("%s: applyGPUVisibility printed %q", c.name, printed)
		}
		if got != want {
			t.Fatalf("%s: applyGPUVisibility = %q, want %q", c.name, got, want)
		}
		assertOnlyAssignment(t, c.name, before, sortedEnviron(), want)
	}
}

func TestVulkanDeviceOrderIsPureAndMatchesSetter(t *testing.T) {
	cuda := &backendInfo{Path: "/x/llama-server", Dialect: "llama"}
	noGPUs := &detect.Capabilities{}
	rig := rigVulkanList
	same := `  Vulkan0: NVIDIA GeForce RTX 4070 (12528 MiB, 1 MiB free)
  Vulkan1: NVIDIA GeForce RTX 3090 Ti (24810 MiB, 1 MiB free)
  Vulkan2: NVIDIA GeForce RTX 3060 (12534 MiB, 1 MiB free)
`
	other := "  Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 1 MiB free)\n"
	cases := []struct {
		name    string
		req     *launchRequest
		be      *backendInfo
		caps    *detect.Capabilities
		list    string
		lists   bool // the setter runs the listing
		value   string
		wantErr string
	}{
		{"nil backend", &launchRequest{}, nil, rigGPUs(), rig, false, "", ""},
		{"cuda backend", &launchRequest{GPUsFlag: "1"}, cuda, rigGPUs(), rig, false, "", ""},
		{"vulkan without detected gpus", &launchRequest{GPUsFlag: "1"}, vulkanBackend(), noGPUs, rig, false, "", ""},
		{"vulkan with nil caps", &launchRequest{}, vulkanBackend(), nil, rig, false, "", ""},
		{"permutation", &launchRequest{}, vulkanBackend(), rigGPUs(), rig, true, "1,0,2", ""},
		{"nil request permutation", nil, vulkanBackend(), rigGPUs(), rig, true, "1,0,2", ""},
		{"cpu mode keeps the permutation", &launchRequest{CPUMode: true}, vulkanBackend(), rigGPUs(), rig, true, "1,0,2", ""},
		{"identity order", &launchRequest{}, vulkanBackend(), rigGPUs(), same, true, "", ""},
		{"identity order with --gpus", &launchRequest{GPUsFlag: "1,2"}, vulkanBackend(), rigGPUs(), same, true, "1,2", ""},
		{"--gpus single", &launchRequest{GPUsFlag: "1"}, vulkanBackend(), rigGPUs(), rig, true, "0", ""},
		{"--gpus not from 0", &launchRequest{GPUsFlag: "1,2"}, vulkanBackend(), rigGPUs(), rig, true, "0,2", ""},
		{"--gpus out of order", &launchRequest{GPUsFlag: "2,0"}, vulkanBackend(), rigGPUs(), rig, true, "1,2", ""},
		{"--gpus missing gpu", &launchRequest{GPUsFlag: "5"}, vulkanBackend(), rigGPUs(), rig, true, "", "--gpus index 5 is not a detected GPU"},
		{"--gpus invalid", &launchRequest{GPUsFlag: "abc"}, vulkanBackend(), rigGPUs(), rig, true, "", `"abc" is not a non-negative GPU index`},
		{"listing of other cards", &launchRequest{}, vulkanBackend(), rigGPUs(), other, true, "", "backend lists 1 devices, detection found 3"},
	}
	for _, c := range cases {
		goldenPlanUnsetenv(t, pureEnvKeys...)
		listed := parseVulkanDeviceList(c.list)
		listedCopy := append([]listedDevice(nil), listed...)
		before := sortedEnviron()
		key, value, err := vulkanDeviceOrder(c.req, c.be, c.caps, listed)
		if after := sortedEnviron(); !reflect.DeepEqual(before, after) {
			t.Fatalf("%s: the pure call changed the environment", c.name)
		}
		if !reflect.DeepEqual(listed, listedCopy) {
			t.Fatalf("%s: the pure call changed its listing", c.name)
		}
		wantKey := ""
		if c.value != "" {
			wantKey = "GGML_VK_VISIBLE_DEVICES"
		}
		if c.wantErr != "" {
			if err == nil || err.Error() != c.wantErr || key != "" || value != "" {
				t.Fatalf("%s: vulkanDeviceOrder = %q, %q, %v; want error %q", c.name, key, value, err, c.wantErr)
			}
		} else if err != nil || key != wantKey || value != c.value {
			t.Fatalf("%s: vulkanDeviceOrder = %q, %q, %v; want %q, %q", c.name, key, value, err, wantKey, c.value)
		}

		calls := 0
		var listedPath string
		lister := func(path string) ([]listedDevice, error) {
			calls++
			listedPath = path
			return parseVulkanDeviceList(c.list), nil
		}
		env, setErr := vulkanDeviceOrderEnv(c.req, c.be, c.caps, lister)
		wantCalls := 0
		if c.lists {
			wantCalls = 1
		}
		if calls != wantCalls {
			t.Fatalf("%s: setter listed devices %d times, want %d", c.name, calls, wantCalls)
		}
		if calls == 1 && listedPath != c.be.Path {
			t.Fatalf("%s: setter listed %q, want the backend %q", c.name, listedPath, c.be.Path)
		}
		want := ""
		if c.value != "" {
			want = "GGML_VK_VISIBLE_DEVICES=" + c.value
		}
		if c.wantErr != "" {
			if setErr == nil || setErr.Error() != c.wantErr || env != "" {
				t.Fatalf("%s: vulkanDeviceOrderEnv = %q, %v; want error %q", c.name, env, setErr, c.wantErr)
			}
		} else if setErr != nil || env != want {
			t.Fatalf("%s: vulkanDeviceOrderEnv = %q, %v; want %q", c.name, env, setErr, want)
		}
		assertOnlyAssignment(t, c.name, before, sortedEnviron(), want)
	}
}

// The pure function takes the listing as a value: given a backend path that
// does not exist, it still answers, so it cannot have run the backend.
func TestVulkanDeviceOrderDoesNotRunTheListing(t *testing.T) {
	goldenPlanUnsetenv(t, pureEnvKeys...)
	be := &backendInfo{Path: filepath.Join(t.TempDir(), "missing-llama-server"), Dialect: "vulkan"}
	key, value, err := vulkanDeviceOrder(&launchRequest{}, be, rigGPUs(), parseVulkanDeviceList(rigVulkanList))
	if err != nil || key != "GGML_VK_VISIBLE_DEVICES" || value != "1,0,2" {
		t.Fatalf("vulkanDeviceOrder = %q, %q, %v", key, value, err)
	}
	if _, err := listVulkanDevices(be.Path); err == nil {
		t.Fatal("listing a missing backend succeeded, so this test proves nothing")
	}
}

// The launch exports GPU visibility first and the Vulkan order second, so on a
// Vulkan backend with --gpus the permuted value is what the backend sees, and a
// failed listing leaves the generic --gpus value in place.
func TestVulkanOrderOverridesGPUVisibilityInCallOrder(t *testing.T) {
	req := &launchRequest{GPUsFlag: "0,2"}
	be := vulkanBackend()

	goldenPlanUnsetenv(t, pureEnvKeys...)
	if env := applyGPUVisibility(req, backendDialect(be)); env != "GGML_VK_VISIBLE_DEVICES=0,2" {
		t.Fatalf("applyGPUVisibility = %q", env)
	}
	if env, err := vulkanDeviceOrderEnv(req, be, rigGPUs(), listing(rigVulkanList)); err != nil || env != "GGML_VK_VISIBLE_DEVICES=1,2" {
		t.Fatalf("vulkanDeviceOrderEnv = %q, %v", env, err)
	}
	if got := os.Getenv("GGML_VK_VISIBLE_DEVICES"); got != "1,2" {
		t.Fatalf("GGML_VK_VISIBLE_DEVICES = %q, want the permuted 1,2", got)
	}
	if _, ok := os.LookupEnv("CUDA_VISIBLE_DEVICES"); ok {
		t.Fatal("a Vulkan launch exported CUDA_VISIBLE_DEVICES")
	}

	goldenPlanUnsetenv(t, pureEnvKeys...)
	applyGPUVisibility(req, backendDialect(be))
	failing := func(string) ([]listedDevice, error) { return nil, os.ErrNotExist }
	if env, err := vulkanDeviceOrderEnv(req, be, rigGPUs(), failing); err == nil || env != "" {
		t.Fatalf("vulkanDeviceOrderEnv with a failed listing = %q, %v", env, err)
	}
	if got := os.Getenv("GGML_VK_VISIBLE_DEVICES"); got != "0,2" {
		t.Fatalf("GGML_VK_VISIBLE_DEVICES = %q, want the generic 0,2 after a failed listing", got)
	}
}

// applyVulkanDeviceOrder hard-wires the real listing, so only the paths that
// do not need a working Vulkan backend are reachable here: silent when the
// order does not apply, and the warning when the listing fails.
func TestApplyVulkanDeviceOrderPrintedText(t *testing.T) {
	cuda := &backendInfo{Path: "/x/llama-server", Dialect: "llama"}
	for _, c := range []struct {
		name string
		be   *backendInfo
		caps *detect.Capabilities
	}{
		{"cuda backend", cuda, rigGPUs()},
		{"vulkan without gpus", vulkanBackend(), &detect.Capabilities{}},
	} {
		goldenPlanUnsetenv(t, pureEnvKeys...)
		before := sortedEnviron()
		out := captureStdout(t)
		applyVulkanDeviceOrder(&launchRequest{}, c.be, c.caps)
		if printed := out(); printed != "" {
			t.Fatalf("%s: printed %q", c.name, printed)
		}
		assertOnlyAssignment(t, c.name, before, sortedEnviron(), "")
	}

	goldenPlanUnsetenv(t, pureEnvKeys...)
	before := sortedEnviron()
	be := &backendInfo{Path: filepath.Join(t.TempDir(), "missing-llama-server"), Dialect: "vulkan"}
	errOut := captureStderrForPureEnv(t)
	out := captureStdout(t)
	applyVulkanDeviceOrder(&launchRequest{}, be, rigGPUs())
	printed := out()
	warned := errOut()
	if printed != "" {
		t.Fatalf("failed listing printed %q on stdout", printed)
	}
	const prefix = "[launch] warning: could not align Vulkan device order with detected GPUs: "
	if !strings.HasPrefix(warned, prefix) || !strings.HasSuffix(warned, "\n") || strings.Count(warned, "\n") != 1 {
		t.Fatalf("failed listing warned %q, want one line starting %q", warned, prefix)
	}
	assertOnlyAssignment(t, "failed listing", before, sortedEnviron(), "")
}

func captureStderrForPureEnv(t *testing.T) func() string {
	t.Helper()
	orig := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stderr = w
	var buf strings.Builder
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = io.Copy(&buf, r)
	}()
	t.Cleanup(func() { os.Stderr = orig })
	return func() string {
		os.Stderr = orig
		_ = w.Close()
		<-done
		return buf.String()
	}
}
