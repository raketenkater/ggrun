# Flag and environment reference

Generated from the code by `go test ./cmd/ggrun -run TestReferenceIsCurrent -args -update-reference`;
do not edit by hand. See [usage.md](usage.md) for what the options do.

## Launch flags

`--flag value` and `--flag=value` mean the same. A switch takes `=true`/`=false`.
Flags ggrun does not know are passed to the backend.

| Flag | Takes a value |
|---|---|
| `--ai-tune` | no (switch) |
| `--allow-live-memory-probe` | no (switch) |
| `--backend` | yes |
| `--batch-size` | yes |
| `--benchmark` | no (switch) |
| `--cache-ram` | yes |
| `--cache-type-k` | yes |
| `--cache-type-v` | yes |
| `--calibrate` | yes |
| `--cgroup-headroom` | yes |
| `--chat-template` | yes |
| `--claude-code` | no (switch) |
| `--claude-max-active` | yes |
| `--claude-penalties` | yes |
| `--claude-profile` | yes |
| `--claude-resume` | yes |
| `--claude-resume-force` | no (switch) |
| `--claude-reviewer` | yes |
| `--cpu` | no (switch) |
| `--ctx` | yes |
| `--ctx-checkpoints` | yes |
| `--ctx-size` | yes |
| `--download` | no (switch) |
| `--dry-run` | no (switch) |
| `--emit-server-argv-json` | no (switch) |
| `--force-spec-moe` | no (switch) |
| `--gateway-host` | yes |
| `--gateway-port` | yes |
| `--gateway-token-file` | yes |
| `--gateway-url` | yes |
| `--gpus` | yes |
| `--headless` | no (switch) |
| `--host` | yes |
| `--install-claude` | no (switch) |
| `--inventory` | yes |
| `--keep-alive` | no (switch) |
| `--kv` | yes |
| `--kv-placement` | yes |
| `--kv-quality` | yes |
| `--mmap` | no (switch) |
| `--mmproj` | yes |
| `--model` | yes |
| `--no-cached-config` | no (switch) |
| `--no-mmap` | no (switch) |
| `--no-support-online` | no (switch) |
| `--no-swa-full` | no (switch) |
| `--parallel` | yes |
| `--port` | yes |
| `--ram-budget` | yes |
| `--ram-headroom` | yes |
| `--ram-limit-percent` | yes |
| `--retune` | no (switch) |
| `--rounds` | yes |
| `--server-bin` | yes |
| `--show-configs` | no (switch) |
| `--spec` | yes |
| `--support-expert` | yes |
| `--support-online` | no (switch) |
| `--swa-full` | no (switch) |
| `--threads` | yes |
| `--tune-cache` | yes |
| `--ubatch-size` | yes |
| `--vision` | no (switch) |
| `--vram-headroom` | yes |
| `--worker-benchmark` | no (switch) |
| `-b` | yes |
| `-c` | yes |
| `-cpu` | no (switch) |
| `-cram` | yes |
| `-ctk` | yes |
| `-ctv` | yes |
| `-ctx` | yes |
| `-ctxcp` | yes |
| `-gpus` | yes |
| `-kv` | yes |
| `-kv-quality` | yes |
| `-m` | yes |
| `-port` | yes |
| `-t` | yes |
| `-ub` | yes |
| `-vision` | no (switch) |

## Configuration variables

Each sets a config key; the `GGRUN_` spelling wins over the `LLM_` one when both are set.

| Variable | Alias |
|---|---|
| `LLM_PORT` | `GGRUN_PORT` |
| `LLM_CTX_SIZE` | `GGRUN_CTX_SIZE` |
| `LLM_MAX_RESTARTS` | `GGRUN_MAX_RESTARTS` |
| `LLM_KEEP_ALIVE` | `GGRUN_KEEP_ALIVE` |
| `LLM_HEALTH_TIMEOUT` | `GGRUN_HEALTH_TIMEOUT` |
| `LLM_MODEL_DIR` | `GGRUN_MODEL_DIR` |
| `LLM_CACHE_DIR` | `GGRUN_CACHE_DIR` |
| `LLM_LOG_DIR` | `GGRUN_LOG_DIR` |
| `LLM_RAM_BUDGET` | `GGRUN_RAM_BUDGET` |
| `LLM_RAM_LIMIT_PERCENT` | `GGRUN_RAM_LIMIT_PERCENT` |
| `LLM_VRAM_HEADROOM` | `GGRUN_VRAM_HEADROOM` |
| `LLM_RAM_HEADROOM` | `GGRUN_RAM_HEADROOM` |
| `LLM_CGROUP_HEADROOM_MB` | `GGRUN_CGROUP_HEADROOM_MB` |
| `LLM_KV_PLACEMENT` | `GGRUN_KV_PLACEMENT` |
| `LLM_KV_QUALITY` | `GGRUN_KV_QUALITY` |
| `LLM_SWA_FULL` | `GGRUN_SWA_FULL` |
| `LLM_ASSUME_YES` | `GGRUN_ASSUME_YES` |
| `LLM_ALLOW_LIVE_MEMORY_PROBE` | `GGRUN_ALLOW_LIVE_MEMORY_PROBE` |
| `LLM_BACKEND` | `GGRUN_BACKEND` |
| `LLAMA_SERVER` | `GGRUN_LLAMA_SERVER` |
| `LLM_APP_HOME` | `GGRUN_APP_HOME` |
| `LLM_TUNE_ROUNDS` | `GGRUN_TUNE_ROUNDS` |
| `LLM_VISION` | `GGRUN_VISION` |
| `LLM_PARALLEL` | `GGRUN_PARALLEL` |
| `LLM_HOST` | `GGRUN_HOST` |
| `LLM_SPEC` | `GGRUN_SPEC` |
| `LLM_SUPPORT_EXPERT` | `GGRUN_SUPPORT_EXPERT` |
| `LLM_SUPPORT_ONLINE` | `GGRUN_SUPPORT_ONLINE` |
| `LLM_SUPPORT_MODEL` | `GGRUN_SUPPORT_MODEL` |
| `LLM_CLAUDE_PENALTIES` | `GGRUN_CLAUDE_PENALTIES` |

## Environment variables read by the Go code

- `CLAUDE_CONFIG_DIR`
- `COLUMNS`
- `CUDACXX`
- `EDITOR`
- `FREETOKEN_BIN`
- `GGML_OP_OFFLOAD_MIN_BATCH`
- `GGRUN_ADVISOR_CONSENT`
- `GGRUN_ALLOW_LIVE_MEMORY_PROBE`
- `GGRUN_APP_HOME`
- `GGRUN_ASSUME_YES`
- `GGRUN_BACKEND`
- `GGRUN_BACKEND_FORMAT_RETRY`
- `GGRUN_BACKEND_VERIFY_MODEL`
- `GGRUN_CACHE_DIR`
- `GGRUN_CGROUP_HEADROOM_MB`
- `GGRUN_CLAUDE_AUTO_REVIEWER`
- `GGRUN_CLAUDE_AUTO_SELF_REVIEW`
- `GGRUN_CLAUDE_PENALTIES`
- `GGRUN_CLAUDE_PERMISSION_MODE`
- `GGRUN_CLAUDE_PROGRESS`
- `GGRUN_CLAUDE_REVIEWER_BIN`
- `GGRUN_CLAUDE_REVIEWER_MODEL`
- `GGRUN_CTX_SIZE`
- `GGRUN_DOWNLOADER`
- `GGRUN_HEALTH_TIMEOUT`
- `GGRUN_HOST`
- `GGRUN_INSTALL_CLAUDE`
- `GGRUN_KEEP_ALIVE`
- `GGRUN_KV_PLACEMENT`
- `GGRUN_KV_QUALITY`
- `GGRUN_LEARNED_REPLAN`
- `GGRUN_LLAMA_SERVER`
- `GGRUN_LOG_DIR`
- `GGRUN_MAX_RESTARTS`
- `GGRUN_MEMGUARD_LIBRARY`
- `GGRUN_MODEL_DIR`
- `GGRUN_NO_WHATS_NEW`
- `GGRUN_PARALLEL`
- `GGRUN_PORT`
- `GGRUN_QUEUE_WAIT_SECONDS`
- `GGRUN_RAM_BUDGET`
- `GGRUN_RAM_HEADROOM`
- `GGRUN_RAM_LIMIT_PERCENT`
- `GGRUN_SCOPE_MODE`
- `GGRUN_SPEC`
- `GGRUN_SUPPORT_EXPERT`
- `GGRUN_SUPPORT_MODEL`
- `GGRUN_SUPPORT_ONLINE`
- `GGRUN_SWA_FULL`
- `GGRUN_TRACE_PLACEMENT`
- `GGRUN_TUNE_ROUNDS`
- `GGRUN_VERIFY_OOM_REPLAN`
- `GGRUN_VISION`
- `GGRUN_VRAM_HEADROOM`
- `HOME`
- `LD_PRELOAD`
- `LLAMA_API_KEY`
- `LLAMA_ARG_LOG_VERBOSITY`
- `LLAMA_SERVER`
- `LLM_ALLOW_LIVE_MEMORY_PROBE`
- `LLM_APP_HOME`
- `LLM_ASSUME_YES`
- `LLM_BACKEND`
- `LLM_BANDWIDTH_PROFILE`
- `LLM_CACHE_DIR`
- `LLM_CGROUP_HEADROOM_MB`
- `LLM_CLAUDE_PENALTIES`
- `LLM_CONFIG`
- `LLM_CTX_SIZE`
- `LLM_HEALTH_TIMEOUT`
- `LLM_HOST`
- `LLM_INSTALL_GO_ROOT`
- `LLM_KEEP_ALIVE`
- `LLM_KV_PLACEMENT`
- `LLM_KV_QUALITY`
- `LLM_LOG_DIR`
- `LLM_MAX_RESTARTS`
- `LLM_MODEL_DIR`
- `LLM_PARALLEL`
- `LLM_PORT`
- `LLM_RAM_BUDGET`
- `LLM_RAM_HEADROOM`
- `LLM_RAM_LIMIT_PERCENT`
- `LLM_SCRIPT_DIR`
- `LLM_SERVER_HOME`
- `LLM_SERVER_LIB_HUB`
- `LLM_SERVER_NO_UPDATE_CHECK`
- `LLM_SERVER_REPO`
- `LLM_SERVER_SKIP_DRAFT_DOWNLOAD`
- `LLM_SERVER_SKIP_MMPROJ_CHECK`
- `LLM_SERVER_UPDATE_CHECKED`
- `LLM_SERVER_VERSION`
- `LLM_SPEC`
- `LLM_SUPPORT_EXPERT`
- `LLM_SUPPORT_MODEL`
- `LLM_SUPPORT_ONLINE`
- `LLM_SWA_FULL`
- `LLM_TUNE_ROUNDS`
- `LLM_VISION`
- `LLM_VRAM_HEADROOM`
- `MSYSTEM`
- `PATH`
- `USERPROFILE`
- `VISUAL`
