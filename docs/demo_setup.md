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

The security update was checked on Linux CPU with Python 3.12: all nine tests passed and `pip check` reported no conflicts. Advisory scanning covered all 67 installed runtime packages with no known vulnerabilities reported, checking the CPU PyTorch build against upstream version 2.13.0. Both models produced identical labels across 28 synthetic example variants before and after the model-library upgrades (Torch 2.10.0/Transformers 5.0.0/tokenizers 0.22.2 versus Torch 2.13.0/Transformers 5.18.0/tokenizers 0.23.2). This is a compatibility smoke check; the full 1,796-abstract benchmark was not rerun for this update.

The checks exercise actual model inference, example selection, empty-input handling, comparison, clearing stale results, out-of-vocabulary baseline input, BERT truncation, cooldowns, shared request limits, and competing requests. Submitted abstracts are not globally cached. A per-model lock serializes access to globally cached inference resources.

## Public demo request controls

One process-wide gate, shared across Streamlit sessions, admits at most one classification request at a time. Admission happens before artifact resolution, model loading, or inference. Compare both occupies one slot until both models finish. Busy requests receive an immediate retry message rather than waiting in an inference queue. Accepted requests, including failed model loads or predictions, count toward a rolling limit of 12 submissions per minute across all sessions. Each session also has a 10-second cooldown after its accepted request finishes. Rejected requests do not extend that cooldown or allocate per-visitor tracking entries.

These limits bound model work; they do not stop connection floods or guarantee fair access. Opening a new session bypasses the session cooldown but still encounters the shared concurrency and rate caps. State resets on process restart and applies separately to each server process. Stronger per-user or per-IP enforcement needs authentication or a trusted gateway in front of the hosting service. Never trust a browser-supplied IP header for this purpose.

Codespaces installs the same demo dependencies on Python 3.12 and runs Streamlit with its CORS and XSRF protections enabled by default. Keep port 8501 private while developing. If the embedded preview fails, open the forwarded port in a browser tab; do not disable these protections.

## Demo dependency updates

`demo/requirements.in` defines the model-compatible environment and explicit security upgrades. `demo/requirements.txt` pins its full resolved dependency set, including packages the hosting environment may already have installed. PyTorch uses official CPU wheel URLs for Python 3.12 on Linux x86_64/aarch64, Windows AMD64/ARM64, and Apple silicon macOS 14+. Other packages come only from PyPI. Use Python 3.12 for these wheels. The original training requirements in the repository root describe the experiment environment and are separate from public demo deployment.

To refresh the pins with uv and audit the resulting environment:

```bash
uv pip compile demo/requirements.in --python-version 3.12 --universal --emit-index-url -o demo/requirements.txt
python -m pip install -r demo/requirements.txt
python -m pip check
pip-audit
```

Install `uv` and `pip-audit` as development tools separately. Check the audit output for skipped packages: the CPU build's `torch==2.13.0+cpu` version may require a separate advisory lookup for its upstream `torch==2.13.0` release. Review audit findings for applicability and rerun the integration and saved-prediction checks after changing model libraries. An audit is a check against known advisories, not a guarantee that no vulnerabilities exist. Merging demo dependency changes into the deployed branch triggers a Community Cloud rebuild; reboot the app if it retains its previous environment. Rebuild an existing Codespace to apply its updated setup.

## Interpretation

Scores are uncalibrated probabilities and are not guarantees of correctness. The models force a single label among seven known domains, including for irrelevant or interdisciplinary input. Results describe one dataset and split; unrelated-domain performance and repeated-run uncertainty were not measured. The synthetic examples are illustrations and are not evaluation evidence.
