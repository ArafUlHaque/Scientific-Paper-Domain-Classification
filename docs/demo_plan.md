# Interactive abstract-classification demo

Status: app implemented; public deployment pending.

The app uses the selected BERT_3 and LR_3 checkpoints, matching TF-IDF vectorizer, original preprocessing, and verified seven-domain mapping. Visitors can paste an abstract, load a synthetic example, select either model or compare both, view ranked scores, and download predictions.

See [setup, deployment, and verification instructions](demo_setup.md). Model weights are an external runtime dependency with checked-in SHA-256 hashes. They are not committed to source control.

Remaining deployment tasks: provide a direct HTTPS URL for the original artifact ZIP, configure the Streamlit hosting account, measure memory and response time on the actual host, and add the successful public URL and screenshot to the README.
