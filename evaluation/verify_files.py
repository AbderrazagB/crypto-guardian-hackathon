"""Verify the exact published model/data/notebook snapshot before replay."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / "evaluation/PUBLICATION-MANIFEST.json").read_text())
for name, info in manifest["files"].items():
    data = (root / name).read_bytes()
    assert len(data) == info["bytes"] and hashlib.sha256(data).hexdigest() == info["sha256"], name
print("Verified", len(manifest["files"]), "published source/input/artifact files.")
