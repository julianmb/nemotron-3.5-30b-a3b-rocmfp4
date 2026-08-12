#!/usr/bin/env bash
# test_patches.sh
#
# Validates that both patches apply cleanly using `git am` or `git apply`
# against a user-specified clone of charlie12345/ROCmFPX.
#
# Usage:
#   ./test_patches.sh /path/to/ROCmFPX
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-..}"

if [[ ! -f "${TARGET}/convert_hf_to_gguf.py" ]]; then
    echo "Error: ${TARGET} does not appear to be a valid ROCmFPX repository."
    echo "Usage: $0 /path/to/ROCmFPX"
    exit 1
fi

echo "==> Testing patch 1: 0001-converter-detect-w4a16-nvfp4.patch"
if git -C "${TARGET}" am --check "${ROOT}/patches/0001-converter-detect-w4a16-nvfp4.patch" >/dev/null 2>&1; then
    echo "  [OK] Patch 1 applies cleanly via git am."
else
    git -C "${TARGET}" apply --check "${ROOT}/patches/0001-converter-detect-w4a16-nvfp4.patch"
    echo "  [OK] Patch 1 applies cleanly via git apply."
fi

echo "==> Testing patch 2: 0002-converter-dequant-output-to-f16.patch"
if git -C "${TARGET}" am --check "${ROOT}/patches/0002-converter-dequant-output-to-f16.patch" >/dev/null 2>&1; then
    echo "  [OK] Patch 2 applies cleanly via git am."
else
    git -C "${TARGET}" apply --check "${ROOT}/patches/0002-converter-dequant-output-to-f16.patch"
    echo "  [OK] Patch 2 applies cleanly via git apply."
fi

echo "==> All patches validated successfully!"
