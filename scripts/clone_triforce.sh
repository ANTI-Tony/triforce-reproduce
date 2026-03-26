#!/usr/bin/env bash
# Clone the official TriForce repository into vendor/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENDOR_DIR="$PROJECT_DIR/vendor"

REPO_URL="https://github.com/Infini-AI-Lab/TriForce.git"
TARGET_DIR="$VENDOR_DIR/TriForce"

if [ -d "$TARGET_DIR" ]; then
    echo "[INFO] TriForce already cloned at $TARGET_DIR"
    echo "[INFO] To re-clone, remove $TARGET_DIR first"
    exit 0
fi

mkdir -p "$VENDOR_DIR"
echo "[INFO] Cloning TriForce from $REPO_URL ..."
git clone "$REPO_URL" "$TARGET_DIR"
echo "[INFO] TriForce cloned to $TARGET_DIR"
echo "[INFO] Commit: $(cd "$TARGET_DIR" && git rev-parse HEAD)"
