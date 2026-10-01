#!/usr/bin/env bash
# Make a repository's labels match .github/labels.yml.
#
#   scripts/sync_labels.sh <owner/repo>            create or update every label
#   scripts/sync_labels.sh <owner/repo> --prune    …and delete labels not in the file
#   scripts/sync_labels.sh <owner/repo> --dry-run  print the gh commands only
#
# Creation uses `gh label create --force` (create or update in place), so it is
# idempotent and safe to re-run. --prune deletes GitHub's defaults we do not
# use (duplicate, invalid, wontfix, enhancement, documentation) — a label that
# is removed from the file stays deleted, so prune only on purpose.
# Needs the gh CLI (logged in, admin on the repo) and PyYAML.
set -euo pipefail
repo="${1:?usage: sync_labels.sh <owner/repo> [--prune] [--dry-run]}"
shift
prune=0; dry=0
for a in "$@"; do
  case "$a" in --prune) prune=1 ;; --dry-run) dry=1 ;; *) echo "unknown flag $a" >&2; exit 2 ;; esac
done
cd "$(dirname "$0")/.."

# name<TAB>color<TAB>description, one per line, from the YAML file.
labels=$(python3 -c '
import sys, yaml
for l in yaml.safe_load(open(".github/labels.yml")):
    print("\t".join(str(l[k]) for k in ("name", "color", "description")))
')

run() { if [ "$dry" = 1 ]; then printf '%q ' "$@"; echo; else "$@"; fi; }

while IFS=$'\t' read -r name color desc; do
  run gh label create "$name" --repo "$repo" --color "$color" --description "$desc" --force
done <<< "$labels"

if [ "$prune" = 1 ]; then
  wanted=$(cut -f1 <<< "$labels")
  gh label list --repo "$repo" --limit 500 --json name --jq '.[].name' | while IFS= read -r have; do
    grep -Fxq "$have" <<< "$wanted" || run gh label delete "$have" --repo "$repo" --yes
  done
fi
echo "labels synced to $repo ($(wc -l <<< "$labels" | tr -d ' ') in the file)"
