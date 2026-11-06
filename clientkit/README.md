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
