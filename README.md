Cross-Lingual Word Sense Disambiguation (CL-WSD) Benchmark Generator

This repository contains a Python-based pipeline for generating a Cross-Lingual Word Sense Disambiguation (CL-WSD) benchmark. The tool is designed to create high-quality evaluation data for testing the contextual reasoning capabilities of multilingual language models.

Overview
Effective evaluation of multilingual language models requires testing for contextual understanding, not just translation recall. This project provides a framework to automatically generate a CL-WSD benchmark to directly address this.

The generated task presents a model with a source-language sentence as context and asks it to select the correct translation from five target-language options. The distractors are other valid translations of the source word for different senses, making the task a true test of contextual disambiguation. The output is a JSONL file compatible with standard evaluation frameworks like lm-evaluation-harness.

Features
Offline First: Uses the BabelNet Python API in offline mode (local index) or via RPC, requiring no active internet connection during generation.

Tiered Language Support: Easily generate benchmarks for high, medium, or low-resource languages using a simple configuration file.

High-Quality Data: Includes a multi-stage quality control pipeline for text normalization, data cleaning, and script validation for non-Latin languages.

Reproducible: Uses a fixed random seed for deterministic shuffling and selection, ensuring that the same benchmark can be generated every time.

Configurable: Highly configurable via command-line arguments to control the number of choices, items, senses per word, and generation modes (exhaustive vs. capped).

Rich Output: Produces standardized JSONL records with rich metadata, including synset IDs, provenance, and generator version for full auditability.

