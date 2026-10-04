# Run and deploy the interactive classifier

The Streamlit app uses the original selected BERT_3 and LR_3 models. It offers editable synthetic examples for seven domains, ranked scores, side-by-side comparison, inference timing, and downloadable predictions. No model training occurs at startup or during prediction. Text is retained only in the visitor's active Streamlit session; the application does not write or log submissions.

## Run locally

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r demo/requirements.txt
python demo/prepare_artifacts.py /path/to/scientific_paper_demo_artifacts.zip
python -m streamlit run demo/app.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The installation script verifies the author's original ZIP checksum and extracts only the selected model and tokenizer files. The local `artifacts/` directory is ignored by Git. For an existing extracted bundle, set `DEMO_ARTIFACT_DIR` to the absolute `cse440-results` directory; its files must match the checked-in integrity manifest.

The ZIP is an external runtime dependency. A source checkout alone cannot classify abstracts; the application never substitutes an untrained model.

## Deploy on Streamlit Community Cloud

1. Merge the app changes into the deployment branch.
2. Upload the original `scientific_paper_demo_artifacts.zip` to a durable artifact host that exposes a direct HTTPS download URL. Keep the exact uploaded bytes; the downloader checks the ZIP's SHA-256 and each selected file's hash. Use a versioned, immutable URL when possible. An HTML sharing page is not a download URL.
3. Create a Streamlit Community Cloud app connected to this repository. Select the deployment branch and **`demo/app.py`** as the entrypoint. Set Python to **3.12**. Its adjacent `demo/requirements.txt` supplies the inference dependencies, avoiding the training environment in the root requirements file.
4. In the app's advanced settings/secrets, add:

```toml
DEMO_ARTIFACT_URL = "https://YOUR-DIRECT-DOWNLOAD-URL/scientific_paper_demo_artifacts.zip"
```

5. Deploy and classify an example using each model and Compare both. Measure cold startup, warm requests, and memory use on this host before calling it production-ready. BERT's checkpoint alone is approximately 438 MB before loading; free hosting capacity must be verified experimentally.
6. Add the successful public URL to the repository About panel and README. The app source is implemented; public deployment is still pending until the artifact URL and hosting account are configured.

Top-level Streamlit secrets are exposed as environment variables. Never commit private hosting URLs or access tokens to source control. The app downloads artifacts only when its first prediction loads a model, then caches the verified files and model objects. The first request can therefore take much longer than later requests.

## Verification

On October 4, 2026, the author reported **zero mismatches for both selected models over all 1,796 original test abstracts on Kaggle**. The uploaded saved predictions were independently checked against the full-precision result tables. The app implementation was also checked locally on CPU: all 1,796 Logistic Regression predictions and a 28-abstract BERT sample matched, and all four integration tests passed. The local BERT sample is separate from the author's full Kaggle verification. The app preserves Unicode NFKC/lowercase/whitespace preprocessing for TF-IDF, raw abstracts for BERT, the original 384-token maximum, and the split manifest's numeric-to-domain mapping. BERT's saved config contains generic LABEL_0…LABEL_6 names; the app uses the verified domain mapping instead.

To verify the app implementation against the original predictions:

```bash
python demo/verify_predictions.py \
  --abstracts /path/to/X.txt \
  --bundle /path/to/scientific_paper_demo_artifacts.zip
```

`--model lr` or `--model bert` selects one model. `--limit 28` checks a sample; omitting it checks the full test split. The command uses saved checkpoints without fitting the vectorizer or retraining.

Run the integration checks after installing artifacts:

```bash
python -m unittest discover -s tests -v
```

The checks exercise actual model inference, example selection, empty-input handling, comparison, clearing stale results, out-of-vocabulary baseline input, and BERT truncation. Submitted abstracts are not globally cached. A per-model lock serializes access to globally cached inference resources.

## Interpretation

Scores are uncalibrated probabilities and are not guarantees of correctness. The models force a single label among seven known domains, including for irrelevant or interdisciplinary input. Results describe one dataset and split; unrelated-domain performance and repeated-run uncertainty were not measured. The synthetic examples are illustrations and are not evaluation evidence.
