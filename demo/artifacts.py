"""Resolve local artifacts or download the exact checksum-verified demo bundle."""
import hashlib
import json
import os
import shutil
import tempfile
import threading
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

MANIFEST = Path(__file__).with_name("artifact_manifest.json")
_RESOLVE_LOCK = threading.Lock()
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "artifacts/cse440-results"


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify(root):
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or incorrect model artifact: {name}")


@lru_cache(maxsize=1)
def resolve_artifacts():
    with _RESOLVE_LOCK:
        return _resolve_artifacts()


def _resolve_artifacts():
    root = Path(os.environ.get("DEMO_ARTIFACT_DIR", str(DEFAULT_ROOT))).expanduser()
    if root.is_dir():
        verify(root)
        return root

    url = os.environ.get("DEMO_ARTIFACT_URL", "")
    if not url.startswith("https://"):
        raise FileNotFoundError("Configure local artifacts or an HTTPS demo artifact URL before deployment.")
    manifest = json.loads(MANIFEST.read_text())
    root.parent.mkdir(parents=True, exist_ok=True)
    # Temporary extraction avoids publishing a partially downloaded directory.
    with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / "bundle.zip"
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        if digest(archive) != manifest["bundle_sha256"]:
            raise ValueError("The downloaded artifact bundle checksum does not match.")
        staging = temporary / "cse440-results"
        with zipfile.ZipFile(archive) as bundle:
            for name in manifest["files"]:
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open("cse440-results/" + name) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        verify(staging)
        # Existing valid roots are normally caught above; never overwrite one.
        shutil.move(str(staging), str(root))
    return root
