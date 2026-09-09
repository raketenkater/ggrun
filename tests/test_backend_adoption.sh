#!/usr/bin/env bash
# A real dynamic-loader failure must reject adoption; a repaired quiet ELF must work.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_dir="$(mktemp -d -t ggrun-adoption.XXXXXX)"
trap 'rm -rf "$fixture_dir"' EXIT
[[ "$(uname -s)" == Linux ]] || { echo "SKIP: ELF loader regression requires Linux"; exit 0; }
command -v cc >/dev/null || { echo "FAIL: cc required for loader regression" >&2; exit 1; }

# Source production guards without running network/install stages.
eval "$(awk '
 /^is_native_binary\(\)|^is_real_llama_server\(\)|^backend_actually_runs\(\)/ {keep=1}
 keep {print}
 keep && /^}/ {keep=0}
' "$repo_dir/install.sh")"
cat > "$fixture_dir/dependency.c" <<'C'
int ggrun_adoption_fixture(void) { return 0; }
C
cat > "$fixture_dir/server.c" <<'C'
extern int ggrun_adoption_fixture(void);
int main(void) { return ggrun_adoption_fixture(); }
C
cc -shared -fPIC "$fixture_dir/dependency.c" -Wl,-soname,libggrun_adoption_fixture.so -o "$fixture_dir/libggrun_adoption_fixture.so"
cc "$fixture_dir/server.c" -L"$fixture_dir" -lggrun_adoption_fixture '-Wl,-rpath,$ORIGIN' -o "$fixture_dir/llama-server"
is_real_llama_server "$fixture_dir/llama-server" || { echo "FAIL: working quiet backend rejected" >&2; exit 1; }
mv "$fixture_dir/libggrun_adoption_fixture.so" "$fixture_dir/dependency.saved"
if is_real_llama_server "$fixture_dir/llama-server"; then
 echo "FAIL: adopted backend with a missing shared library" >&2; exit 1
fi
[[ "${BACKEND_RUN_ERROR:-}" == *libggrun_adoption_fixture.so* ]] || { echo "FAIL: loader diagnostic lost" >&2; exit 1; }
mv "$fixture_dir/dependency.saved" "$fixture_dir/libggrun_adoption_fixture.so"
is_real_llama_server "$fixture_dir/llama-server" || { echo "FAIL: repaired backend rejected" >&2; exit 1; }
echo "PASS: native backend adoption rejects a real missing library and accepts its repair"
