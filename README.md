# TopiCK (EMNLP 2025) – Reproduction with Stable Inference

This repository is a forked and modified version of the original **TopiCK (EMNLP 2025)** implementation.
The goal of this fork is to provide a **robust and reproducible experimental setup**
by addressing practical issues in inference, evaluation, and model configuration.

---

## Key Modifications

This fork introduces the following changes on top of the original implementation:

- **Stabilized inference pipeline**
  - Revised prediction handling to correctly support label-based outputs
  - Fixed evaluation logic to ensure consistent and reliable accuracy computation

- **Controlled model configuration**
  - Adapted the codebase to support lightweight model variants
  - Simplified experimental settings for reproducible local runs

- **Improved retrieval stability**
  - Fixed FAISS import and runtime issues observed during retrieval
  - Reduced environment-specific failure cases

These changes focus on improving **experimental reliability and reproducibility**
rather than maximizing peak model performance.

---

## Acknowledgement

This repository is **forked from the original implementation of TopiCK (EMNLP 2025)**.
All credit for the original method and experimental design belongs to the original authors.

This fork contains only practical modifications for stable experimentation
and reproducible evaluation.
