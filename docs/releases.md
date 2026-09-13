# Release verification

Each tag publishes platform archives, `install.sh`, `install.ps1`, and
`SHA256SUMS`. The installer checks the checksum before unpacking. `ggrun --update`
refuses an installer that is not listed in `SHA256SUMS`.

The current latest tag (`v3.2.8`) includes Linux CPU, Linux Vulkan, Linux
CUDA (ik_llama.cpp), macOS Metal, and Windows CPU. Setup downloads the CUDA
bundle when it is on the release; it only compiles if that file is missing.

The release workflow also publishes SHA256SUMS.bundle, a keyless Sigstore
signature bundle for SHA256SUMS. To verify it manually with cosign:

~~~bash
cosign verify-blob \
  --bundle SHA256SUMS.bundle \
  --certificate-identity-regexp 'https://github.com/raketenkater/ggrun/.github/workflows/release.yml@refs/tags/.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  SHA256SUMS
~~~

Then verify the archive:

~~~bash
sha256sum -c SHA256SUMS
~~~

The release pipeline pins the ik_llama.cpp revision used to build the CUDA
bundle. Its workflow run and the signed checksum bundle are the source of truth
for a published artifact.

For optional Linux multi-GPU acceptance with a large model, see
[large-model hardening](large-model-hardening.md).
