# Release verification

Each tagged release publishes platform bundles and a SHA256SUMS file. The
installers verify the checksum before unpacking a release bundle.

Linux NVIDIA releases include ggrun-linux-x86_64-cuda.tar.gz. It contains the
pinned ik_llama.cpp CUDA backend so a normal x86_64 installation does not need a
local CUDA toolkit or compiler.

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
