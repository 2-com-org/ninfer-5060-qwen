#!/usr/bin/env bash
# Publish wiki/ to the GitHub wiki of the repository.
#
#   scripts/qwen38/publish-wiki.sh [--repo OWNER/NAME] [--branch BRANCH] [--dry-run DIR]
#
# Links between pages (Serving.md) become wiki links (Serving); links into the repository
# (../docs/x.md) become blob URLs on BRANCH. The wiki must be enabled and have a first page (create
# any page once on GitHub), and git must be able to push to OWNER/NAME.wiki.git. --dry-run writes
# the converted pages to DIR instead of pushing.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
repo=2-com-org/ninfer-5060-qwen
branch=master
dry=''
while (( $# )); do
  case "$1" in
    --repo) repo=${2:?}; shift 2 ;;
    --branch) branch=${2:?}; shift 2 ;;
    --dry-run) dry=${2:?}; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) printf 'publish-wiki.sh: unknown option %s\n' "$1" >&2; exit 2 ;;
  esac
done

work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
if [[ -n $dry ]]; then target=$dry; mkdir -p "$target"; else
  git clone --quiet "https://github.com/$repo.wiki.git" "$work/wiki"
  target=$work/wiki
fi
blob="https://github.com/$repo/blob/$branch"
for page in "$root"/wiki/*.md; do
  sed -E \
    -e 's#\]\(([A-Za-z_][A-Za-z0-9_-]*)\.md(\#[^)]*)?\)#](\1\2)#g' \
    -e "s#\\]\\(\\.\\./([^)]*)\\)#](${blob//#/\\#}/\\1)#g" \
    "$page" > "$target/$(basename -- "$page")"
done
[[ -z $dry ]] || exit 0
git -C "$target" add -A
if git -C "$target" diff --cached --quiet; then
  printf 'The wiki is up to date.\n'
  exit 0
fi
git -C "$target" commit --quiet -m "docs: update from $branch $(git -C "$root" rev-parse --short HEAD)"
git -C "$target" push --quiet
printf 'Published https://github.com/%s/wiki\n' "$repo"
