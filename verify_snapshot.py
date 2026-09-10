"""Verify published source files against the snapshot manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
for entry in manifest["files"]:
    path = root / entry["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"SHA-256 mismatch: {entry['path']}")
print(f"Verified {len(manifest['files'])} source files from {manifest['source_commit']}")
