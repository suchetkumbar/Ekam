# Ekam — one business, one record

**Ekam** (Sanskrit: *one*) works out which records across messy Indian business sources —
"Sharma Traders Pvt Ltd", "SHARMA TRADERS PRIVATE LIMITED", "M/s Sharma Trdrs",
"शर्मा ट्रेडर्स प्रा. लि." — are the same company, and merges them into one golden record.

> **Status: Milestone 2 of 5 (blocking + baselines) — see [docs/reports/M2.md](docs/reports/M2.md).**
> The full README (demo, results tables, limitations) arrives with Milestone 5.

## What exists today

* **Data:** 278,030 real companies from the MCA company master (data.gov.in, GODL-India),
  plus six public matching benchmarks — see [`docs/DATA.md`](docs/DATA.md).
* **Synthetic multi-source generator** with exact ground truth: 4 source systems (ERP, GST
  portal, OCR'd CRM, Hindi vendor register) × 3 difficulty levels, India-specific noise
  (legal-suffix variants, abbreviations, Devanagari and romanisation variants, city aliases,
  wrong PINs, missing/garbled CIN and clearly-synthetic GSTINs) — see
  [`docs/NOISE.md`](docs/NOISE.md) and the [audit](docs/reports/M1_audit.md).
* **Normalisation:** legal-suffix canonicalisation, abbreviation expansion, Devanagari →
  Latin, romanisation folding (Shree/Shri/Sri, Lakshmi/Laxmi), address parsing (PIN, state,
  city aliases), phonetic keys.
* **Entity-disjoint splits** with leakage tests; 467 tests.

## Quick start

```bash
uv sync            # Python 3.11, pinned by uv.lock
make data          # download, ingest, generate sources, split, audit  (Windows: winget install ezwinports.make)
make test
```

## Prior work

Ekam builds on — and will be compared against — **Splink** (probabilistic Fellegi–Sunter
linkage), **Magellan / py_entitymatching** and **DeepMatcher** (benchmarks and deep EM),
and **Ditto** (pre-trained-LM entity matching). What it adds is the India-specific noise
and normalisation, an end-to-end pipeline measured stage by stage, label-efficiency via
active learning, field-level explanations, and a laptop-scale run.

Motivated by the Amazon ML Challenge 2026 (business entity resolution) and the IndiaAI
Innovation Challenge 2026 MSME vendor-mapping track; neither dataset is used (see DATA.md).
