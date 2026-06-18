# Brand & naming reminder

Read once when working on naming, the README, or any public post. Not meant to be
loaded every turn — just check it before writing user-facing copy.

Audience = r/LocalLLaMA / HN / self-hosters. **Looking un-marketed is the marketing.**

- No hype words: blazingly fast, supercharge, seamless, powerful, revolutionary, effortless.
- No AI-hype framing or emoji. It's a GGUF launcher, not an "AI app".
- Lead with a command + real output. Numbers need a reproducible command.
- State limitations out loud (it builds trust).
- Name: plain lowercase, `/usr/bin`-style (like fzf, mosh, caddy). Not a coined
  brand (no Revv/.ai). Check GitHub/Homebrew/PyPI/npm before locking one in.
- Rejected names (don't re-propose): Crank, yoke, manifold, stoke, hotrod, clutch, rev/Revv.

## Decision (leaning — revisit before launch, no rush)

- **Name = `ggrun` (provisional, favored).** Clean on PyPI/npm/Homebrew; no competing
  CLI; self-documenting to GGUF users (`ggrun = gguf run`). (`span` rejected: two
  existing `span` CLIs + taken username/registries.)
- Open gut-check: `gg` can read as gamer slang / slightly string-like. Mitigation if
  kept: README line 1 spells out "ggrun = gguf run". Not final — fine to reconsider
  until the rename actually lands.
- **Rename `llm-server` → `ggrun` BEFORE the big launch post.** Don't spend the
  one-shot launch on the old name and rename after.
- **Bridge, don't hard-cut** (there are prior llm-server Reddit posts to preserve):
  GitHub repo rename auto-redirects old links; keep a `llm-server` binary alias and
  a "formerly llm-server" line in README + launch post for one release cycle, then drop.
- Reserve `ggrun` on PyPI + npm now so nobody squats during the transition.
