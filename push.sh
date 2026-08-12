#!/usr/bin/env bash
# Push gh-public/ to GitHub as julianmb/nemotron-3.5-30b-a3b-rocmfp4.
#
# Pre-requisites (once):
#   gh auth login -h github.com          # interactive re-auth
#
# Usage:
#   ./push.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="nemotron-3.5-30b-a3b-rocmfp4"

echo "==> Checking gh auth..."
if ! gh auth status >/dev/null 2>&1; then
    echo "Error: gh authentication failed. Please run:"
    echo "  gh auth login -h github.com"
    exit 1
fi

echo "==> Creating & pushing GitHub repo julianmb/${REPO}..."
cd "${ROOT}"

if gh repo view "julianmb/${REPO}" >/dev/null 2>&1; then
    echo "Repo already exists on GitHub, pushing main branch..."
    git remote add origin "https://github.com/julianmb/${REPO}.git" 2>/dev/null || true
    git push -u origin main
else
    gh repo create "${REPO}" --public --source . --remote origin --push
fi

echo "==> Success! View repository at:"
echo "    https://github.com/julianmb/${REPO}"
