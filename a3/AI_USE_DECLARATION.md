# AI Use Declaration — Assignment 3 (draft for the student to review and complete)

Course: AI Security and Risk Analysis (AISRA1301) · Student: Nursultan Yessenbek

This draft lists the substantive AI assistance that was actually used. Please check each item, edit it, and complete the
bracketed parts before submission.

## Tools used

- **Claude (Anthropic)**, used through the Claude app on 6 November 2026.
- **ChatGPT (OpenAI)** for the Midterm prototype, website, journal and presentation (21–23 October 2026), which Assignment 3 builds on.

## What the AI assistant did

1. Read the assignment brief, the Midterm report, the Midterm repository and the Midterm Colab notebook, and extracted the 72 frozen
   Midterm cases (verified by SHA-256 against the original notebook asset).
2. Proposed the research questions, the three mechanisms (injection detector, source validation, output verification) and the
   experimental design; wrote the code in `a3/` (test-suite builders, detector wrapper, RAG pipeline, verifier, Colab runner,
   analysis and figure scripts) and the Colab notebook.
3. Wrote the synthetic test content: benign access requests, Russian/Kazakh translations of six Midterm attacks, the RAG v2 corpus,
   questions and poisoned/tampered documents. Obfuscations were generated with PyRIT converters; Tensor Trust attacks were sampled
   from the published benchmark.
4. Operated Google Colab in the student's account to run the experiment, retrieved the raw results and checked their SHA-256.
5. Proposed the manual labels for RAG answers (`results/rag_manual_labels.csv`) using the rubric in `README.md`.
6. Computed all metrics, produced the figures and drafted the technical report text.

## What the student did

- [Describe your own decisions, e.g. choice of topic and system, approval of the design and of publishing to GitHub.]
- [Describe what you independently verified, e.g. re-ran the notebook, checked the labels, checked numbers in the report.]

## Verification

All reported numbers are recomputed by `a3/analyze.py` from the raw Colab records. Post-hoc analyses are labelled as such
in the report and README. [Confirm here which parts you personally reviewed.]
