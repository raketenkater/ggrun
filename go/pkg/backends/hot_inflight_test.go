package backends

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func extractedCPPBlock(src, marker string) (string, error) {
	start := strings.Index(src, marker)
	if start < 0 {
		return "", fmt.Errorf("missing %q", marker)
	}
	open := strings.Index(src[start:], "{")
	if open < 0 {
		return "", fmt.Errorf("missing body for %q", marker)
	}
	open += start
	depth := 0
	for i := open; i < len(src); i++ {
		switch src[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return src[start : i+1], nil
			}
		}
	}
	return "", fmt.Errorf("unterminated body for %q", marker)
}

func cacheSourceFromFeaturePatch(t *testing.T) string {
	t.Helper()
	parts := strings.Split(string(hotExpertsFeaturePatch), "diff --git a/src/llama-moecache.cpp b/src/llama-moecache.cpp\n")
	if len(parts) != 2 {
		t.Fatal("embedded MoE cache source patch missing")
	}
	section := strings.SplitN(parts[1], "\ndiff --git ", 2)[0]
	var src strings.Builder
	for _, line := range strings.Split(section, "\n") {
		if strings.HasPrefix(line, "+") && !strings.HasPrefix(line, "+++") {
			src.WriteString(line[1:])
			src.WriteByte('\n')
		}
	}
	return src.String()
}

func runInflightHarness(t *testing.T, source string) (int, error) {
	t.Helper()
	layer, err := extractedCPPBlock(source, "struct layer_state")
	if err != nil {
		return -1, err
	}
	job, err := extractedCPPBlock(source, "struct upload_job")
	if err != nil {
		return -1, err
	}
	cache, err := extractedCPPBlock(source, "struct moe_cache")
	if err != nil {
		return -1, err
	}
	step, err := extractedCPPBlock(source, "void llama_moe_cache_step()")
	if err != nil {
		return -1, err
	}
	cpp := `
#include <cstdint>
#include <thread>
#include <cinttypes>
#include <condition_variable>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <vector>
struct ggml_tensor {};
struct ggml_context {};
using ggml_backend_buffer_t = void *;
struct llama_moe_cache_layer { std::map<int32_t,int32_t> table; };
` + layer + `;
` + job + `;
` + cache + `;
moe_cache * g_cache = nullptr;
void set_table_entry(llama_moe_cache_layer & pub, int32_t id, int32_t slot) {pub.table[id]=slot;}
#define LLAMA_LOG_DEBUG(...) do {} while (0)
` + step + `
int main() {
    moe_cache mc;
    mc.n_slots = 2;
    mc.max_inserts = 2;
    layer_state ls;
    ls.slot_expert.assign(2, -1);
    ls.expert_slot.assign(4, -1);
    ls.slot_last_use.assign(2, 0);
    ls.slot_in_flight.assign(2, false);
    INFLIGHT_INIT
    ls.pending.push_back(1);
    mc.layers.push_back(std::move(ls));
    g_cache = &mc;
    llama_moe_cache_step();
    if (mc.todo.size() != 1 || mc.layers[0].expert_slot[1] >= 0) return 10;
    mc.layers[0].pending.push_back(1);
    llama_moe_cache_step();
    if (mc.todo.size() != 1) return 11;
    if (!mc.layers[0].pub.table.empty()) return 14;
    mc.done.push_back(mc.todo.front());
    mc.todo.pop_front();
    llama_moe_cache_step();
    if (mc.layers[0].expert_slot[1] != 0) return 12;
    mc.layers[0].pending.push_back(2);
    llama_moe_cache_step();
    mc.done.push_back(mc.todo.front());
    mc.todo.pop_front();
    llama_moe_cache_step();
    mc.layers[0].pending.push_back(3);
    llama_moe_cache_step();
    if (mc.layers[0].expert_slot[2] != 1 || mc.layers[0].expert_slot[1] != -1) return 13;
    if (mc.layers[0].pub.table[1] != mc.n_slots || mc.layers[0].pub.table[2] != 1) return 15;
    return 0;
}
`
	init := ""
	if strings.Contains(source, "std::vector<bool>     expert_in_flight;") {
		init = "ls.expert_in_flight.assign(4, false);"
	}
	cpp = strings.Replace(cpp, "INFLIGHT_INIT", init, 1)
	dir := t.TempDir()
	cppPath := filepath.Join(dir, "harness.cpp")
	binPath := filepath.Join(dir, "harness")
	if err := os.WriteFile(cppPath, []byte(cpp), 0600); err != nil {
		return -1, err
	}
	cmd := exec.Command("g++", "-std=c++17", "-O0", cppPath, "-o", binPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		return -1, fmt.Errorf("compile: %w: %s", err, out)
	}
	if out, err := exec.Command(binPath).CombinedOutput(); err != nil {
		if failure, ok := err.(*exec.ExitError); ok {
			return failure.ExitCode(), nil
		}
		return -1, fmt.Errorf("harness: %w: %s", err, out)
	}
	return 0, nil
}

func TestHotExpertInflightDedupUsesPatchedSchedulerSource(t *testing.T) {
	if _, err := exec.LookPath("g++"); err != nil {
		t.Skip("g++ unavailable")
	}
	original := cacheSourceFromFeaturePatch(t)
	if code, err := runInflightHarness(t, original); err != nil || code != 11 {
		t.Fatalf("unpatched source must compile and reproduce duplicate job (11): code=%d err=%v", code, err)
	}
	patch, err := os.ReadFile(filepath.Join("patches", "features", "hot-experts", "0003-deduplicate-inflight-experts.patch"))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "src"), 0700); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "src", "llama-moecache.cpp")
	if err := os.WriteFile(file, []byte(original), 0600); err != nil {
		t.Fatal(err)
	}
	apply := exec.Command("git", "apply", "--unsafe-paths", "-")
	apply.Dir = dir
	apply.Stdin = strings.NewReader(string(patch))
	if out, err := apply.CombinedOutput(); err != nil {
		t.Fatalf("apply dedup patch: %v: %s", err, out)
	}
	patched, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if code, err := runInflightHarness(t, string(patched)); err != nil || code != 0 {
		t.Fatalf("patched scheduler: code=%d err=%v", code, err)
	}
}
