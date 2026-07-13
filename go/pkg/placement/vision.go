package placement

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/gguf"
)

// findOrDownloadMMProj finds a vision projector locally, or downloads from HuggingFace.
// Validates GGUF metadata for compatibility with the text model.
// textName and textBasename are the text model's GGUF metadata name and basename fields.
// quantizedBy is the GGUF metadata quantizer field (e.g. "unsloth") used to build candidate repos.
func findOrDownloadMMProj(modelPath, cacheDir, textName, textBasename, quantizedBy string) (string, error) {
	modelDir := filepath.Dir(modelPath)
	baseName := strings.TrimSuffix(filepath.Base(modelPath), ".gguf")
	// Strip common quant suffixes (Q4_K_M, Q5_K_M, Q8_0, F16, etc.)
	baseNoQuant := baseName
	for _, suffix := range []string{"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q6_K", "Q8_0", "F16", "F32", "BF16", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL"} {
		if strings.HasSuffix(baseNoQuant, "-"+suffix) {
			baseNoQuant = strings.TrimSuffix(baseNoQuant, "-"+suffix)
			break
		}
	}

	// 1. Check model-specific mmproj first
	for _, ftype := range []string{"F16", "BF16", "F32"} {
		for _, name := range []string{baseName, baseNoQuant} {
			c := filepath.Join(modelDir, fmt.Sprintf("mmproj-%s-%s.gguf", name, ftype))
			if _, err := os.Stat(c); err == nil {
				if err := validateMMProj(c, textName, textBasename); err == nil {
					return c, nil
				}
			}
		}
	}

	// 2. Check generic mmproj files
	candidates := []string{
		filepath.Join(modelDir, "mmproj-F16.gguf"),
		filepath.Join(modelDir, "mmproj-BF16.gguf"),
		filepath.Join(modelDir, "mmproj-F32.gguf"),
		filepath.Join(modelDir, "mmproj.gguf"),
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			if err := validateMMProj(c, textName, textBasename); err != nil {
				fmt.Fprintf(os.Stderr, "[vision] rejecting %s: %v\n", filepath.Base(c), err)
				continue
			}
			return c, nil
		}
	}

	// 3. Glob for any mmproj/projector files
	entries, err := os.ReadDir(modelDir)
	if err == nil {
		for _, e := range entries {
			name := strings.ToLower(e.Name())
			if !e.IsDir() && strings.HasSuffix(name, ".gguf") &&
				(strings.Contains(name, "mmproj") || strings.Contains(name, "projector")) {
				c := filepath.Join(modelDir, e.Name())
				if err := validateMMProj(c, textName, textBasename); err != nil {
					fmt.Fprintf(os.Stderr, "[vision] rejecting %s: %v\n", e.Name(), err)
					continue
				}
				return c, nil
			}
		}
	}

	// 4. Try native Go download from HuggingFace
	if path, err := downloadMMProj(modelDir, textName, textBasename, quantizedBy); err == nil {
		return path, nil
	}

	return "", fmt.Errorf("no mmproj found — place mmproj-F16.gguf in %s or use --mmproj <path>", modelDir)
}

// normalize strips non-alphanumeric characters and lowercases for comparison.
var normalizeRE = regexp.MustCompile(`[^a-z0-9]+`)

func normalize(s string) string {
	return normalizeRE.ReplaceAllString(strings.ToLower(s), "")
}

// validateMMProj checks that an mmproj GGUF file is valid and compatible with the text model.
// Validates the projector: arch == "clip", file completeness, and name/basename overlap.
func validateMMProj(path, textName, textBasename string) error {
	// Allow bypass for advanced users
	if os.Getenv("LLM_SERVER_SKIP_MMPROJ_CHECK") == "1" {
		fmt.Fprintf(os.Stderr, "[vision] Warning: skipping mmproj compatibility check for %s\n", path)
		return nil
	}

	info, err := gguf.Parse(path)
	if err != nil {
		return fmt.Errorf("parse failed: %w", err)
	}
	if info == nil {
		return fmt.Errorf("empty metadata")
	}

	// Check 1: architecture must be "clip"
	if info.Architecture != "clip" {
		return fmt.Errorf("architecture is %q, expected \"clip\"", info.Architecture)
	}

	// Check 2: file completeness (actual size >= declared tensor bytes)
	expectedBytes := info.NonExpertBytes + info.ExpertBytes
	if expectedBytes > 0 {
		fi, err := os.Stat(path)
		if err != nil {
			return fmt.Errorf("stat failed: %w", err)
		}
		if fi.Size() < int64(expectedBytes) {
			return fmt.Errorf("incomplete file: %d bytes, expected at least %d", fi.Size(), expectedBytes)
		}
	}

	// Check 3: name/basename overlap between text model and projector
	textIDs := make(map[string]bool)
	if n := normalize(textName); n != "" {
		textIDs[n] = true
	}
	if n := normalize(textBasename); n != "" {
		textIDs[n] = true
	}

	projIDs := make(map[string]bool)
	if n := normalize(info.Name); n != "" {
		projIDs[n] = true
	}
	if n := normalize(info.Basename); n != "" {
		projIDs[n] = true
	}

	if len(textIDs) == 0 || len(projIDs) == 0 {
		return fmt.Errorf("cannot verify compatibility: missing name/basename metadata")
	}

	// Check for overlap
	found := false
	for id := range textIDs {
		if projIDs[id] {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("mmproj metadata does not match the selected model (text: %v, proj: %v)", textIDs, projIDs)
	}

	return nil
}

// downloadMMProj attempts to download a compatible mmproj from HuggingFace.
// It builds candidate repos from the text model's basename and quantized_by metadata,
// queries the HF tree API for mmproj files, downloads and validates them.
func downloadMMProj(modelDir, textName, textBasename, quantizedBy string) (string, error) {
	basename := textBasename
	if basename == "" {
		basename = textName
	}
	if basename == "" {
		return "", fmt.Errorf("no model basename for mmproj lookup")
	}

	// Build candidate HuggingFace repos
	var repoCandidates []string
	if quantizedBy != "" {
		repoCandidates = append(repoCandidates, quantizedBy+"/"+basename+"-GGUF")
		if textName != "" && textName != basename {
			repoCandidates = append(repoCandidates, quantizedBy+"/"+textName+"-GGUF")
		}
	}
	for _, q := range []string{"unsloth", "bartowski", "lmstudio-community"} {
		if q == quantizedBy {
			continue
		}
		repoCandidates = append(repoCandidates, q+"/"+basename+"-GGUF")
	}

	safeBasename := sanitizeFilename(basename)
	if safeBasename == "" {
		safeBasename = "model"
	}

	client := &http.Client{Timeout: 30 * time.Second}

	for _, repo := range repoCandidates {
		paths := listRepoMMProjCandidates(client, repo)
		seen := make(map[string]bool)

		for _, remotePath := range paths {
			if remotePath == "" || seen[remotePath] {
				continue
			}
			seen[remotePath] = true

			remoteBase := filepath.Base(remotePath)
			suffix := remoteBase
			suffix = strings.TrimPrefix(suffix, "mmproj-")
			suffix = strings.TrimPrefix(suffix, "mmproj_")
			if suffix == remoteBase {
				suffix = remoteBase
			}
			safeSuffix := sanitizeFilename(suffix)
			if safeSuffix == "" {
				safeSuffix = "mmproj.gguf"
			}
			dest := filepath.Join(modelDir, "mmproj-"+safeBasename+"-"+safeSuffix)

			// Check if already downloaded and valid
			if _, err := os.Stat(dest); err == nil {
				if err := validateMMProj(dest, textName, textBasename); err == nil {
					fmt.Fprintf(os.Stderr, "[vision] Found compatible mmproj: %s\n", dest)
					return dest, nil
				}
			}

			dlURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s",
				repo, url.PathEscape(remotePath))

			// HEAD check before downloading
			headResp, err := client.Head(dlURL)
			if err != nil || headResp.StatusCode != http.StatusOK {
				continue
			}
			headResp.Body.Close()

			fmt.Fprintf(os.Stderr, "[vision] Downloading mmproj from %s: %s\n", repo, remotePath)

			tmpDest := dest + ".tmp"
			if err := downloadFile(client, dlURL, tmpDest); err != nil {
				fmt.Fprintf(os.Stderr, "[vision] download interrupted; partial file kept for resume: %s\n", tmpDest)
				continue
			}

			// Verify GGUF magic
			if !isGGUF(tmpDest) {
				fmt.Fprintf(os.Stderr, "[vision] Downloaded file is not a valid GGUF, removing\n")
				os.Remove(tmpDest)
				continue
			}

			// Validate metadata compatibility
			if err := validateMMProj(tmpDest, textName, textBasename); err != nil {
				fmt.Fprintf(os.Stderr, "[vision] Downloaded mmproj does not match: %v, removing\n", err)
				os.Remove(tmpDest)
				continue
			}

			if err := os.Rename(tmpDest, dest); err != nil {
				os.Remove(tmpDest)
				return "", fmt.Errorf("rename: %w", err)
			}
			fmt.Fprintf(os.Stderr, "[vision] Downloaded: %s\n", dest)
			return dest, nil
		}
	}

	return "", fmt.Errorf("no compatible mmproj found on HuggingFace")
}

// listRepoMMProjCandidates queries the HuggingFace tree API for mmproj files in a repo.
// Returns paths sorted by precision preference (F16 > BF16 > F32 > other).
func listRepoMMProjCandidates(client *http.Client, repo string) []string {
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s/tree/main?recursive=1", repo)
	resp, err := client.Get(apiURL)
	if err != nil || resp.StatusCode != http.StatusOK {
		return fallbackMMProjPaths()
	}
	defer resp.Body.Close()

	var items []struct {
		Path string `json:"path"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&items); err != nil {
		return fallbackMMProjPaths()
	}

	var paths []string
	for _, item := range items {
		name := strings.ToLower(filepath.Base(item.Path))
		if strings.HasSuffix(name, ".gguf") && strings.Contains(name, "mmproj") {
			paths = append(paths, item.Path)
		}
	}

	// Deduplicate
	seen := make(map[string]bool)
	var unique []string
	for _, p := range paths {
		if !seen[p] {
			seen[p] = true
			unique = append(unique, p)
		}
	}

	sort.Slice(unique, func(i, j int) bool {
		return mmprojRank(unique[i]) < mmprojRank(unique[j])
	})

	// Append fallback paths at the end
	unique = append(unique, fallbackMMProjPaths()...)
	return unique
}

func mmprojRank(path string) int {
	name := strings.ToLower(filepath.Base(path))
	switch {
	case name == "mmproj-f16.gguf":
		return 0
	case name == "mmproj-bf16.gguf":
		return 1
	case name == "mmproj-f32.gguf":
		return 2
	case strings.Contains(name, "f16"):
		return 3
	case strings.Contains(name, "bf16"):
		return 4
	case strings.Contains(name, "f32"):
		return 5
	case name == "mmproj.gguf":
		return 6
	default:
		return 7
	}
}

func fallbackMMProjPaths() []string {
	return []string{"mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-F32.gguf", "mmproj.gguf"}
}

func downloadFile(client *http.Client, url, dest string) error {
	offset := int64(0)
	if fi, err := os.Stat(dest); err == nil && !fi.IsDir() {
		offset = fi.Size()
	}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	flags := os.O_CREATE | os.O_WRONLY | os.O_TRUNC
	if resp.StatusCode == http.StatusPartialContent && offset > 0 {
		wantPrefix := fmt.Sprintf("bytes %d-", offset)
		if !strings.HasPrefix(resp.Header.Get("Content-Range"), wantPrefix) {
			return fmt.Errorf("invalid resume Content-Range %q, expected %q", resp.Header.Get("Content-Range"), wantPrefix+"...")
		}
		flags = os.O_CREATE | os.O_WRONLY | os.O_APPEND
	} else if resp.StatusCode == http.StatusRequestedRangeNotSatisfiable && offset > 0 {
		if strings.TrimSpace(resp.Header.Get("Content-Range")) == fmt.Sprintf("bytes */%d", offset) {
			return nil
		}
		return fmt.Errorf("HTTP %d resuming at byte %d", resp.StatusCode, offset)
	} else if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	// A server that ignores Range returns 200. Truncate and restart rather than
	// appending a second copy and silently corrupting the GGUF.
	f, err := os.OpenFile(dest, flags, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func isGGUF(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil {
		return false
	}
	return string(magic) == "GGUF"
}

var sanitizeRE = regexp.MustCompile(`[^A-Za-z0-9._+\-]`)

func sanitizeFilename(s string) string {
	s = sanitizeRE.ReplaceAllString(s, "_")
	s = strings.Trim(s, "_")
	return s
}
