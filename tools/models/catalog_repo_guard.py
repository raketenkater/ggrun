#!/usr/bin/env python3
"""Fail when a catalog refresh changes the repository of an existing row.

A scheduled refresh may resolve a different Hugging Face repository for a model
that is already offered. That changes what users download without review, so
the refresh must stop and a person must look at it.

usage: catalog_repo_guard.py OLD_CATALOG NEW_CATALOG
"""
import json
import sys


def changed_repos(old, new):
    """(name, old_repo, new_repo) for rows present in both with a different repo."""
    before = {r.get("name"): r.get("repo", "") for r in old.get("candidates", []) if r.get("name")}
    out = []
    for row in new.get("candidates", []):
        name = row.get("name")
        if name in before and before[name] and row.get("repo", "") != before[name]:
            out.append((name, before[name], row.get("repo", "")))
    return out


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    with open(argv[1]) as fh:
        old = json.load(fh)
    with open(argv[2]) as fh:
        new = json.load(fh)
    changes = changed_repos(old, new)
    for name, was, now in changes:
        print(f"repository changed for {name}: {was} -> {now}", file=sys.stderr)
    if changes:
        print(f"{len(changes)} existing row(s) changed repository; review before publishing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
