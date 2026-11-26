# Remote agent availability corrections — October 30

Addresses the reproducible blockers in #113 and a memory-sizing invariant
identified while reviewing #117. No model identity, context, KV, GPU placement,
companion selection or user permission override is changed.

## Permission deadline

Claude 2.1.307 cancels a mock Auto classifier delayed 75 seconds after about
60 seconds despite the general API timeout of 2147483647 ms. With the main
model also classifying, ordinary tool use can therefore stop before execution.

The local launcher now defaults to normal prompts when the companion is
explicitly disabled. The remote client does the same when no separate worker
is advertised. Protocol 1 has a worker bit, not a distinct reviewer bit, so
review-only companions conservatively receive normal prompts on the client.
Explicit CLI/environment Auto choices and inheritance remain authoritative.
No failed review is turned into an approval.

In an isolated fixture, the corrected client with the real qualified native CLI
executed the harmless, explicitly allowed Python marker with zero classifier
requests. Without the allow rule the same tool returned `requires approval`,
again with no classifier request. These are mock-provider integration results,
not evidence of a completed real-model goal or of the owner's laptop upgrade.

Existing alpha.2 users can resume with:

```sh
ggrun-client claude --connection server --resume -- --permission-mode default
```

## Future host-memory capacity

Previously, when the measured backend exceeded the estimated host footprint,
the post-canary floor could discard the future prompt-cache allowance. When
the planned floor won, it also consumed the separate cgroup growth headroom.

The corrected floor is the larger estimated/measured base plus the planned
cache capacity. The scope adds growth headroom beside that floor, then clamps
to the original resource ceiling. This deliberately retains the whole cache
allowance because the first canary is not proof that the cache is populated.
Explicit RAM budgets still bypass automatic scope resizing. Missing footprint,
zero headroom, non-Linux, mmap reclaim and sparse lookup paths retain their
existing eligibility and absolute ceiling rules.

Coverage includes CPU-only, dense offload and MoE sizing, future growth,
explicit overrides and ceiling clamps. The uncached core gate passed. This is
an availability/accounting correction, not a performance promotion.

The prior OOM is not fully attributed or closed by these tests. `memory.peak`
is a lifetime peak and may include loading before the scope was tightened; it
does not establish the kill-time usage or hard limit. Keep #117 open for a
representative contained long-session/cache-churn run. The current no-companion
run had no new OOM events at inspection; do not kill it just to reload a patch.

After the gate passed, the same conservative sizing rule was applied to the
single owner's live backend scope, without restarting or cancelling requests.
The hard limit increased from 69551 to 81313 MiB: measured non-reclaimable
62369 MiB + the entire 14848 MiB cache allowance + 4096 MiB growth headroom.
This remains below the original 205823 MiB admission ceiling; 146267 MiB was
available before the adjustment. The scope reported no OOM events afterward.
This is an operator-scoped mitigation, not proof of universal long-run safety.
The patched executable must still be used on the next launch for persistence.

Compaction and goal-continuation issues remain separately tracked: controlled
compaction works, while the owner's failing Workflow/goal trace is still needed
before assigning an additional code defect.
