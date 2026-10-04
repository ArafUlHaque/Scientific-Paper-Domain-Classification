"""Interactive scientific abstract classification demo."""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from demo.artifacts import resolve_artifacts
from demo.inference import BertClassifier, LogisticClassifier, validate_text
from demo.request_limits import RequestGate, RequestRejected

st.set_page_config(page_title="Scientific Paper Domain Classifier", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def load_classifier(name):
    root = resolve_artifacts()
    return BertClassifier(root) if name == "BERT" else LogisticClassifier(root)


@st.cache_resource(show_spinner=False)
def request_gate():
    return RequestGate()


examples = json.loads((Path(__file__).parent / "examples.json").read_text())

st.caption("NLP PROJECT · ARAF UL HAQUE")
st.title("Where does your research belong?")
st.write("Paste an English scientific abstract to classify it into one of seven research domains.")

classify_tab, study_tab = st.tabs(["Try the classifier", "About the study"])

with classify_tab:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        sample = st.selectbox("Start with an example", ["Write your own", *examples])
        if st.button("Use example", disabled=sample == "Write your own"):
            st.session_state["abstract"] = examples[sample]
        text = st.text_area("Scientific abstract", key="abstract", height=260,
                            placeholder="Paste your abstract here…", max_chars=20000)
        choice = st.radio("Model", ["BERT", "Logistic Regression", "Compare both"], horizontal=True)
        st.caption("BERT achieved the strongest test result. Logistic Regression offers a lightweight baseline.")
        submitted = st.button("Classify abstract", type="primary", width="stretch")
        st.caption("Please allow 10 seconds between submissions. When the demo is busy, try again shortly.")
        st.caption("Examples are synthetic illustrations, not benchmark samples. Submitted text is not written to files or logged by this app.")

    signature = (text, choice)
    if st.session_state.get("result_signature") != signature:
        st.session_state.pop("results", None)
        st.session_state.pop("errors", None)
    if submitted:
        st.session_state["result_signature"] = signature
        results, errors = [], []
        try:
            validate_text(text)
            names = ["BERT", "Logistic Regression"] if choice == "Compare both" else [choice]
            with request_gate().admit(st.session_state.get("last_classification_finished")):
                try:
                    for name in names:
                        try:
                            with st.spinner(f"Classifying with {name}…"):
                                results.append(asdict(load_classifier(name).predict(text)))
                        except ValueError as error:
                            errors.append(f"{name}: {error}")
                        except Exception:
                            # Do not expose submitted text, local paths, or server details.
                            errors.append(f"{name} is currently unavailable. Please try again later.")
                finally:
                    st.session_state["last_classification_finished"] = time.monotonic()
        except RequestRejected as error:
            errors.append(str(error))
        except ValueError as error:
            errors.append(str(error))
        st.session_state["results"] = results
        st.session_state["errors"] = errors

    with right:
        st.subheader("Prediction")
        for error in st.session_state.get("errors", []):
            st.error(error)
        results = st.session_state.get("results", [])
        if not results and not st.session_state.get("errors"):
            st.info("Your predicted domain and ranked scores will appear here.")
        for result in results:
            with st.container(border=True):
                st.caption(result["model"])
                st.subheader(result["label"])
                st.caption(f"Prediction time: {result['seconds']:.2f} seconds · excludes model loading")
                scores = pd.DataFrame(result["scores"].items(), columns=["Domain", "Score"])
                scores = scores.sort_values("Score", ascending=False)
                st.bar_chart(scores.set_index("Domain"), horizontal=True, color="#167D7F")
                st.dataframe(scores, hide_index=True, width="stretch",
                             column_config={"Score": st.column_config.NumberColumn(format="%.4f")})
                if result["truncated"]:
                    st.warning("This abstract exceeds BERT's 384-token limit. Its prediction uses only the beginning of the abstract.")
        if len(results) == 2:
            if results[0]["label"] == results[1]["label"]:
                st.success("Both models predict the same domain.")
            else:
                st.info("The models disagree. Interdisciplinary abstracts can span several domains.")
        if results:
            st.caption("Scores are uncalibrated model probabilities. They do not establish that a prediction is correct.")
            st.download_button("Download prediction", json.dumps(results, indent=2),
                               file_name="domain_prediction.json", mime="application/json")
    st.divider()
    st.caption("The models always choose a known domain. They cannot reliably identify unrelated text or represent multiple fields in one prediction.")

with study_tab:
    st.subheader("From lexical features to contextual representations")
    st.write("The project compares ten classifiers on WOS-11967: 11,967 abstracts across seven parent domains. Configurations were selected on validation macro-F1 and evaluated on 1,796 held-out test abstracts.")
    a, b, c = st.columns(3)
    a.metric("BERT test macro-F1", "95.06%")
    b.metric("Baseline test macro-F1", "90.27%")
    c.metric("Research domains", "7")
    st.write("The demo uses the original BERT_3 and LR_3 checkpoints. The author reported zero prediction mismatches for both models across the 1,796 original test abstracts when verifying the uploaded artifacts on Kaggle.")
    st.write("Results cover one dataset and split. Performance on other sources, repeated-run uncertainty, and calibrated confidence were not measured.")
    st.markdown("[Source code and results](https://github.com/ArafUlHaque/Scientific-Paper-Domain-Classification) · [Project report](https://github.com/ArafUlHaque/Scientific-Paper-Domain-Classification/blob/main/reports/project_report.pdf) · [Kaggle notebook](https://www.kaggle.com/code/arafulhaque/scientific-paper-domain-classification)")
    st.caption("Dataset: Kowsari et al., Web of Science Dataset v6, CC BY 4.0. End-to-end implementation and experiments: Araf Ul Haque.")
