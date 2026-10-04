"""Install the author's original ZIP into the local ignored artifact directory."""
import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.artifacts import DEFAULT_ROOT, MANIFEST, digest, verify
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path, help="scientific_paper_demo_artifacts.zip")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if digest(args.zip) != manifest["bundle_sha256"]:
        parser.error("This ZIP does not match the original verified artifact bundle.")
    DEFAULT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_ROOT.exists():
        verify(DEFAULT_ROOT)
        print("The verified artifacts are already installed.")
        return
    with tempfile.TemporaryDirectory(dir=DEFAULT_ROOT.parent) as directory:
        staging = Path(directory) / "cse440-results"
        with zipfile.ZipFile(args.zip) as archive:
            for name in manifest["files"]:
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open("cse440-results/" + name) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        verify(staging)
        shutil.move(str(staging), str(DEFAULT_ROOT))
    print("Verified artifacts installed. Run: python -m streamlit run demo/app.py")


if __name__ == "__main__":
    main()
