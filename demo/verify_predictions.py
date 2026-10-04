"""Check the app's inference code against the original held-out predictions."""
import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from demo.artifacts import resolve_artifacts
from demo.inference import BertClassifier, LogisticClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abstracts", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 checks all 1,796 test abstracts")
    parser.add_argument("--model", choices=["both", "bert", "lr"], default="both")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    abstracts = args.abstracts.read_text(encoding="utf-8-sig").splitlines()
    summary = []
    with zipfile.ZipFile(args.bundle) as bundle:
        with bundle.open("cse440-results/split_manifest.csv") as stream:
            manifest = pd.read_csv(stream).set_index("document_id")
        assert len(abstracts) == len(manifest) == 11967
        options = [("lr", "logistic_regression", LogisticClassifier), ("bert", "bert_base", BertClassifier)]
        for key, filename, constructor in options:
            if args.model not in ["both", key]:
                continue
            with bundle.open(f"cse440-results/predictions/{filename}.csv") as stream:
                reference = pd.read_csv(stream)
            if args.limit:
                reference = reference.head(args.limit)
            rows = manifest.loc[reference["document_id"]]
            assert rows["split"].eq("test").all()
            assert rows["label"].tolist() == reference["true_label"].tolist()
            model = constructor(resolve_artifacts())
            predictions = []
            for index, row in enumerate(rows["source_row"]):
                predictions.append(model.predict(abstracts[int(row)]).label_id)
                if (index + 1) % 100 == 0:
                    print(f"{key}: {index + 1}/{len(reference)}", flush=True)
            mismatches = sum(a != b for a, b in zip(predictions, reference["predicted_label"]))
            summary.append({"model": key, "checked": len(reference), "mismatches": mismatches})
    print(json.dumps(summary, indent=2))
    if any(row["mismatches"] for row in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
