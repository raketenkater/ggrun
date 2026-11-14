# ggrun client contract

Small reusable Go module for local and remote Claude client policy. No model
runtime, hardware detector or network-provider dependency. `Profile.Validate`
enforces protocol/policy v1; `Environment` constructs a scoped child environment.
The Workflow helper runs on the machine owning the workflow files.

`contract.schema.json` and `testdata/profile-v1.json` are the wire contract.
Clients pin a reviewed module version or commit. Version incompatible changes;
do not send executable commands or arbitrary environment settings in metadata.

Private files use owner-only Unix permissions or Windows user-bound DPAPI.
Windows protected files must be re-created when changing user/computer; do not
copy a plaintext credential file into the Windows configuration directory.

Run `go test -race -count=1 ./...`. Windows runtime behavior additionally needs
native Windows CI; a cross-build alone is not validation of DPAPI or terminals.

## Utility capacity and Workflow concurrency

Optional `utility_route` metadata describes context per slot, slot count,
whether a separate backend is available and whether routing may fall back to
main. Its context is conservative across both destinations. Missing metadata
means unknown capacity; it does not imply a suitable judge, schema support or
deadline compliance. Native classifier routing can override the requested alias.

`Environment` caps Workflow concurrency to executor capacity and preserves a
valid lower inherited limit. Reviewer slots do not increase main capacity.
Explicit utility subagents use the smaller applicable capacity, conservatively
one on old profiles. This cap does not measure another session's queue pressure.
