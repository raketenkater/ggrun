package main

import (
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/gguf"
	"github.com/raketenkater/ggrun/pkg/models"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

// localModelListTimeout bounds the model-directory walk the recommender waits
// for. A slow disk then only means no exact profiles this run; the walk
// itself is bounded by models.List.
const localModelListTimeout = 2 * time.Second

// enableLocalQuantProfiles lets the recommender read the exact precision of
// catalog files that are already downloaded: a local model whose total bytes
// equal the catalog file's size and whose name carries the quant name is that
// file. Profiles go through the GGUF header cache, keyed by file identity.
func enableLocalQuantProfiles() {
	cfg, err := config.Load()
	if err != nil || strings.TrimSpace(cfg.ModelDir) == "" {
		recommend.SetLocalQuantProfile(nil)
		return
	}
	root := cfg.ModelDir
	var (
		once   sync.Once
		listed []models.Model
		mu     sync.Mutex
		cache  *gguf.HeaderCache
	)
	recommend.SetLocalQuantProfile(func(c recommend.Candidate, q recommend.QuantOption) (gguf.QuantProfile, bool) {
		if q.SizeBytes <= 0 {
			return gguf.QuantProfile{}, false
		}
		once.Do(func() {
			done := make(chan []models.Model, 1)
			go func() {
				list, _ := models.List(root)
				done <- list
			}()
			select {
			case listed = <-done:
			case <-time.After(localModelListTimeout):
			}
		})
		path, ok := matchLocalModel(listed, root, q)
		if !ok {
			return gguf.QuantProfile{}, false
		}
		mu.Lock()
		defer mu.Unlock()
		if cache == nil {
			cache = gguf.OpenHeaderCache(cfg.CacheDir)
		}
		p, err := cache.QuantProfile(path)
		_ = cache.Save()
		return p, err == nil
	})
}

// matchLocalModel finds the downloaded copy of a catalog file: exactly its
// bytes (summed over shards), and its quant name in the model's name.
func matchLocalModel(list []models.Model, root string, q recommend.QuantOption) (string, bool) {
	quant := strings.ToUpper(strings.TrimSpace(q.Name))
	if q.SizeBytes <= 0 || quant == "" {
		return "", false
	}
	for _, m := range list {
		if m.Bytes != q.SizeBytes || len(m.Files) == 0 || !strings.Contains(strings.ToUpper(m.Name), quant) {
			continue
		}
		return filepath.Join(root, m.Files[0]), true
	}
	return "", false
}
