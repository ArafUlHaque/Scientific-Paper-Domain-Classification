# Scientific Paper Domain Classification

Classifying scientific abstracts into seven research domains using TF-IDF, GloVe recurrent networks, and BERT.

**11,967 abstracts · 10 models · 30 tuning configurations · 95.06% best test macro-F1**

[Explore the notebook](notebooks/scientific_paper_domain_classification.ipynb) · [Open on Kaggle](https://www.kaggle.com/code/arafulhaque/scientific-paper-domain-classification) · [Read the report](reports/project_report.pdf) · [Reproduction notes](docs/reproducibility.md)

## Overview

Scientific abstracts often share vocabulary across disciplines, making automatic organization challenging. This project compares three representation families on **WOS-11967** to predict one of seven parent domains: Computer Science, Electrical Engineering, Psychology, Mechanical Engineering, Civil Engineering, Medical Science, or Biochemistry.

The study covers dataset auditing, exploratory analysis, representation-specific preprocessing, hyperparameter tuning, held-out evaluation, and error analysis. It was completed for CSE440: Natural Language Processing II at BRAC University, Summer 2026.

## Results

Configurations were selected using validation macro-F1, with validation accuracy as a tie-breaker. The selected models were evaluated on **1,796 held-out test abstracts**.

| Model | Representation | Test accuracy | Test macro-F1 |
|---|---|---:|---:|
| **BERT Base** | WordPiece + fine-tuned BERT | **95.16%** | **95.06%** |
| Bidirectional GRU | GloVe 6B 100d | 90.59% | 90.56% |
| Logistic Regression | TF-IDF | 90.59% | 90.27% |
| Random Forest | TF-IDF | 89.64% | 89.58% |
| GRU | GloVe 6B 100d | 87.64% | 87.56% |
| Bidirectional LSTM | GloVe 6B 100d | 85.24% | 85.14% |
| LSTM | GloVe 6B 100d | 84.86% | 84.76% |
| Multinomial Naive Bayes | TF-IDF | 84.24% | 83.58% |
| Bidirectional SimpleRNN | GloVe 6B 100d | 73.00% | 71.36% |
| SimpleRNN | GloVe 6B 100d | 48.44% | 44.38% |

![Test macro-F1 comparison across ten models](results/figures/final_model_comparison.png)

**What the comparison shows:** BERT achieved the highest test macro-F1, improving on Bidirectional GRU by **4.50 percentage points**. Logistic Regression reached **90.27% macro-F1**, providing a strong lexical baseline with a much shorter recorded tuning run. The time measurements describe the recorded experiment environment; inference latency and deployment cost were not benchmarked.

These values come from the completed notebook's saved outputs. The [downloadable tables](results/) retain displayed precision; they are not the original full-precision Kaggle CSV files. Minor report rounding differences are documented in [reproducibility notes](docs/reproducibility.md).

## Experimental design

| Component | Approach |
|---|---|
| Dataset | WOS-11967, version 6; 11,967 English abstracts |
| Inputs and target | `X.txt` abstracts and `YL1.txt` parent labels |
| Split | 8,376 train / 1,795 validation / 1,796 test; seed 42 |
| Data checks | Missing/empty abstracts, exact duplicates, normalized duplicates, conflicting labels |
| TF-IDF | Train-fitted vocabulary; unigrams and bigrams; up to 50,000 features |
| Recurrent models | Train-fitted tokenizer; 30,000 vocabulary entries; 320-token sequences; 100d GloVe |
| BERT | `bert-base-uncased`; 384 WordPiece tokens; 3 epochs; 3 learning rates |
| Selection | Three configurations per model; validation macro-F1, then accuracy |
| Evaluation | Accuracy, macro-F1, weighted-F1, per-class reports, confusion matrices |

The saved audit reports no exact or normalized duplicate abstracts. Representation fitting uses training data. Test performance is excluded from hyperparameter selection. The exported notebook's final-stage session **reloads saved experiments**, so its displayed outputs are evidence of the completed study, not a fresh retraining performed for this repository update.

## Error analysis

![BERT raw and normalized confusion matrices](results/figures/bert_base_confusion_matrices.png)

The most frequent BERT confusion was **Biochemistry → Medical Science (11 cases)**, followed by **Mechanical Engineering → Civil Engineering (8 cases)**. Shared terminology and interdisciplinary subject matter are plausible explanations. The dataset forces each abstract into one parent category.

[Top confusion pairs](results/top_confusion_pairs.csv) · [Per-class results for all models](results/classification_reports/)

## Explore or reproduce

1. **Inspect the study:** open the notebook above; saved results are visible without running code. Sections 10–12 contain tuning, selection, evaluation, and interpretation.
2. **Understand the data:** follow [dataset instructions](data/README.md).
3. **Run on Kaggle:** use the linked notebook with its input dataset and complete saved artifact bundle. Read [reproduction notes](docs/reproducibility.md) first: the submitted notebook depends on those artifacts and is not currently a one-click training run from an empty environment.
4. **Re-export the displayed evidence locally:** install `pandas` and `lxml`, then run `python scripts/extract_notebook_outputs.py`. This reads saved notebook outputs and does not train models.

## Repository guide

| Path | Contents |
|---|---|
| `notebooks/` | Completed notebook with original code and saved outputs, plus reading notes |
| `reports/` | Submitted project report |
| `results/` | Exported result tables, per-class reports, figures, and provenance |
| `data/` | Acquisition instructions and separately supplied split artifacts |
| `docs/` | Reproduction details, contribution statement, and demo design |
| `scripts/` | Reproducible extraction of saved notebook outputs |

## Limitations and next step

- Results use one dataset, split, and seed; repeated-run uncertainty and external-domain generalization were not measured.
- Single-label predictions cannot fully represent interdisciplinary abstracts.
- Fixed sequence lengths can discard information from longer abstracts.
- General-purpose GloVe vectors do not cover all scientific vocabulary.
- Recorded tuning times are specific to the experiment sessions.

**Interactive demo implemented:** paste an abstract, try examples from seven domains, and compare the original BERT and Logistic Regression classifiers with ranked scores. [App source](demo/app.py) · [Run locally or deploy](docs/demo_setup.md). Public hosting is pending; the original model bundle is an external runtime dependency. The author verified both selected checkpoints on Kaggle with zero mismatches across all 1,796 saved test predictions.

## Author and credits

**End-to-end implementation, experiments, evaluation, and report preparation: Araf Ul Haque.** The submitted report lists Araf Ul Haque, Taufiq Khan Shuvo, and Ahanaf Rafi Bhuiyan as authors. [Contribution details](docs/contributions.md).

Dataset: Kowsari et al., *Web of Science Dataset*, version 6, [Mendeley Data](https://data.mendeley.com/datasets/9rw3vkcfy4/6), DOI `10.17632/9rw3vkcfy4.6`. Dataset license: CC BY 4.0. GloVe and BERT references are included in the report. The dataset license does not establish a license for this repository's code.
