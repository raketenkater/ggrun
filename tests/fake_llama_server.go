// Command fake_llama_server is a cross-platform llama-server stand-in used by
// release installer smoke tests. It proves the installed ggrun can discover,
// launch, health-check, and stop a packaged backend without loading a real LLM.
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
)

func main() {
	host, port := "127.0.0.1", 8080
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--help", "-h":
			fmt.Println("fake llama-server --host HOST --port PORT --model MODEL --n-gpu-layers N --threads N")
			return
		case "--version":
			fmt.Println("fake llama-server 1.0")
			return
		case "--host":
			if i+1 < len(args) {
				i++
				host = args[i]
			}
		case "--port":
			if i+1 < len(args) {
				i++
				port, _ = strconv.Atoi(args[i])
			}
		default:
			if value, ok := strings.CutPrefix(args[i], "--host="); ok {
				host = value
			}
			if value, ok := strings.CutPrefix(args[i], "--port="); ok {
				port, _ = strconv.Atoi(value)
			}
		}
	}

	http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"status": "ok"})
	})
	http.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"object": "list", "data": []any{}})
	})
	http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{
			"id": "smoke", "object": "chat.completion",
			"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}}},
			"usage":   map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		})
	})

	addr := fmt.Sprintf("%s:%d", host, port)
	fmt.Printf("fake llama-server listening on %s\n", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}
