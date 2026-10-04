"""Load the original selected checkpoints without fitting or retraining."""
import sys
import json
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

LABELS = {
    0: "Computer Science", 1: "Electrical Engineering", 2: "Psychology",
    3: "Mechanical Engineering", 4: "Civil Engineering",
    5: "Medical Science", 6: "Biochemistry",
}
MAX_CHARACTERS = 20000
MAX_TOKENS = 384


def normalize_text(text):
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized.lower()).strip()


def validate_text(text):
    if not text.strip():
        raise ValueError("Paste an abstract or choose an example first.")
    if len(text) > MAX_CHARACTERS:
        raise ValueError(f"Please keep your abstract under {MAX_CHARACTERS:,} characters.")
    return text


@dataclass
class Prediction:
    model: str
    label: str
    label_id: int
    scores: dict
    seconds: float
    token_count: int | None = None
    truncated: bool = False


class LogisticClassifier:
    def __init__(self, root):
        # The original notebook pickled its preprocessor as __main__.normalize_text.
        # Register the identical callable before loading the trusted artifact.
        sys.modules["__main__"].normalize_text = normalize_text
        self.vectorizer = joblib.load(Path(root) / "tfidf_vectorizer.joblib")
        self.model = joblib.load(Path(root) / "checkpoints/LR_3.joblib")
        if set(self.model.classes_) != set(LABELS):
            raise ValueError("The baseline checkpoint has an unexpected class mapping.")
        if len(self.vectorizer.vocabulary_) != self.model.n_features_in_:
            raise ValueError("The baseline model and vectorizer do not match.")
        self.lock = threading.Lock()

    def predict(self, text):
        validate_text(text)
        with self.lock:
            start = time.perf_counter()
            features = self.vectorizer.transform([text])
            if features.nnz == 0:
                raise ValueError("This text has no terms recognized by the baseline. Try a fuller scientific abstract or BERT.")
            scores = self.model.predict_proba(features)[0]
            label = int(self.model.classes_[int(np.argmax(scores))])
            elapsed = time.perf_counter() - start
        return Prediction("Logistic Regression", LABELS[label], label,
                          {LABELS[int(k)]: float(v) for k, v in zip(self.model.classes_, scores)}, elapsed)


class BertClassifier:
    def __init__(self, root):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        path = Path(root) / "checkpoints/BERT_3/best_model"
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        if self.model.config.num_labels != len(LABELS):
            raise ValueError("The BERT checkpoint has an unexpected number of labels.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.lock = threading.Lock()

    def predict(self, text):
        validate_text(text)
        with self.lock, self.torch.inference_mode():
            start = time.perf_counter()
            count = len(self.tokenizer(text, truncation=False)["input_ids"])
            inputs = self.tokenizer(text, padding="max_length", truncation=True,
                                    max_length=MAX_TOKENS, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            scores = self.model(**inputs).logits.softmax(dim=-1)[0].cpu().numpy()
            label = int(np.argmax(scores))
            elapsed = time.perf_counter() - start
        return Prediction("BERT", LABELS[label], label,
                          {LABELS[k]: float(v) for k, v in enumerate(scores)}, elapsed,
                          token_count=count, truncated=count > MAX_TOKENS)
