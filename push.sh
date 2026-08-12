#!/usr/bin/env bash
# Push gh-public/ to GitHub as julianmb/nemotron-3.5-30b-a3b-rocmfp4.
#
# Pre-requisites (once):
#   gh auth login -h github.com          # interactive; fixes the invalid token
#   gh repo create nemotron-3.5-30b-a3b-rocmfp4 --public --source . --remote origin --push
#
# Or run step-by-step from this directory:
#   gh auth status
#   gh repo create nemotron-3.5-30b-a3b-rocmfp4 --public --source . --remote origin --push
#   git push -u origin main
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="nemotron-3.5-30b-a3b-rocmfp4"

echo "==> Checking gh auth..."
gh auth status || { echo "Run: gh auth login -h github.com"; exit 1; }

echo "==> Creating + pushing ${REPO} ..."
cd "${ROOT}"
gh repo create "${REPO}" --public --source . --remote origin --push
git push -u origin main

echo "==> Done. View at: https://github.com/julianmb/${REPO}"
