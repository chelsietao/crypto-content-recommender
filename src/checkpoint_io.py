"""Verify checkpoint bytes against an adjacent curated manifest."""
import hashlib
import json
from pathlib import Path


def verified_checkpoint_hash(path):
    path = Path(path)
    manifest = json.loads(path.with_name('manifest.json').read_text())
    expected = manifest.get(path.name)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not isinstance(expected, str) or actual != expected:
        raise ValueError('Checkpoint hash does not match manifest')
    return actual
