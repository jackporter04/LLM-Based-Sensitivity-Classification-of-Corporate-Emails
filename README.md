# Code Submission — Jack Porter

## Overview

This notebook contains the code used to run the classification and retrieval experiments for my dissertation. It is written in Python and designed to run in Google Colab.

The notebook is structured sequentially and divided into the following sections:

**Data Preparation** — loads the SARA dataset via ir-datasets, extracts the body text from each email stripping headers and metadata, and removes any emails above 14,000 tokens to stay within the Gemma API rate limits. The remaining emails are split into three processing buckets based on token size, with larger emails assigned longer sleep times between API calls.

**Emotion Signal Analysis** — maps the UC Berkeley emotion annotations onto the development set to identify which emotions appear more frequently in sensitive versus non-sensitive emails. The eight emotions most associated with sensitive emails were selected as the signals for the emotion and tone method.

**LLM Querying** — sends each email to the Gemma-3-4b-it model via Google AI Studio, once per signal question per method. Each method has its own set of questions stored in a dictionary. A shared context block is prepended to every prompt to ensure consistency across methods. The querying function includes checkpointing so it resumes from the last saved row if interrupted, and saves progress every 10 rows. Pre-computed prediction files are loaded further down the notebook to avoid re-running this section unnecessarily.

**Classification Evaluation** — applies an OR-gate strategy to combine the individual signal predictions into a final binary sensitivity label for each email. Performance is then evaluated using balanced accuracy, precision, recall, F1, and MCC. Confusion matrices and McNemar's significance tests are also produced here.

**Threshold Analysis** — tests alternative classification thresholds by requiring two or more, three or more, etc. signal detections before labelling an email as sensitive. Results are plotted across all methods for each metric.

**Category and Length Analysis** — evaluates classification performance broken down by UC Berkeley email category and by token length range.

**Ablation Study** — removes one signal at a time per method and recalculates performance to identify which signals contribute most.

**Cross-Method Aggregation** — identifies the top six signals by true positives and by precision across all three methods, then evaluates two hybrid aggregation approaches using the OR-gate strategy.

**Prompt Decomposition Comparison** — compares the decomposed expert-derived method against a single-prompt version using the same signals, to isolate the effect of prompt decomposition on classification performance.

**Retrieval Pipeline** — builds a BM25 retrieval pipeline using PyTerrier, then applies each method's sensitivity predictions as a post-retrieval filter. Performance is measured using MAP, nDCG@10, and a custom Sens@10 metric tracking the average number of sensitive documents in the top 10 results. An unfiltered BM25 baseline and a perfect oracle filter are included as reference points.

## Files Not Included

The following files were too large to submit but are required to run the notebook in full:

- `100_new_prompt.csv` — the fixed development set of 100 emails used for prompt design
- `enron_with_categories/` — UC Berkeley annotated Enron emails used for category and emotion analysis
- `dev_full_updated.csv`, `tax_full.csv`, `emotions_full_updated.csv`, `baseline_updated.csv` — pre-computed LLM predictions for each method
