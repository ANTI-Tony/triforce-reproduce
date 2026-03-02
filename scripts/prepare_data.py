#!/usr/bin/env python3
"""Copy PG-19 test data to the location TriForce expects."""
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

SRC_FILE = os.path.join(PROJECT_DIR, "data", "pg19_test.jsonl")
TRIFORCE_DATA_DIR = os.path.join(PROJECT_DIR, "vendor", "TriForce", "data", "pg19")
DST_FILE = os.path.join(TRIFORCE_DATA_DIR, "pg19_test.jsonl")


def validate_jsonl(path: str) -> int:
    """Validate JSONL file and return line count."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" not in obj:
                    print(f"[WARN] Line {i}: missing 'text' field")
                count += 1
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {i}: invalid JSON - {e}")
                sys.exit(1)
    return count


def main():
    # Check source exists
    if not os.path.exists(SRC_FILE):
        print(f"[ERROR] Source data not found: {SRC_FILE}")
        print(f"[INFO] Please place PG-19 test JSONL at: {SRC_FILE}")
        sys.exit(1)

    # Check TriForce is cloned
    triforce_dir = os.path.join(PROJECT_DIR, "vendor", "TriForce")
    if not os.path.isdir(triforce_dir):
        print(f"[ERROR] TriForce not found at {triforce_dir}")
        print("[INFO] Run scripts/clone_triforce.sh first")
        sys.exit(1)

    # Validate source
    print(f"[INFO] Validating {SRC_FILE} ...")
    n = validate_jsonl(SRC_FILE)
    print(f"[INFO] Found {n} valid samples")

    # Copy to TriForce data dir
    os.makedirs(TRIFORCE_DATA_DIR, exist_ok=True)
    shutil.copy2(SRC_FILE, DST_FILE)
    print(f"[INFO] Copied data to {DST_FILE}")

    # Verify copy
    src_size = os.path.getsize(SRC_FILE)
    dst_size = os.path.getsize(DST_FILE)
    if src_size == dst_size:
        print(f"[INFO] Verified: {src_size} bytes")
    else:
        print(f"[ERROR] Size mismatch: src={src_size}, dst={dst_size}")
        sys.exit(1)

    print("[INFO] Data preparation complete!")


if __name__ == "__main__":
    main()
