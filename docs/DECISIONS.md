# Decision log

Choices that shape results, with the reason for each. Newest last.

| Date | Area | Decision | Reason |
|---|---|---|---|
| 2026-11-25 | Data | Do not use the Amazon ML Challenge 2026 data | Only unofficial mirrors exist; no public licence. Cited as motivation only. |
| 2026-11-25 | Data | MCA company master (data.gov.in, 2015 snapshot) as the clean source; 5-state subset for M1, full country in M4 | Public, keyless, GODL-India; one CIN = one real entity gives exact ground truth. |
| 2026-11-25 | Data | Benchmarks: Abt-Buy, Amazon-Google, DBLP-ACM, Walmart-Amazon, Company (DeepMatcher) + WDC Products 80% cc; local only | Comparability with prior work; no redistribution licence. |
| 2026-11-25 | Privacy | Column allow-list at ingest (+ refuse personal-looking names) | A future snapshot with director/e-mail/phone columns cannot leak them. |
| 2026-11-25 | Synth | 4 sources (erp, gst_portal, crm_ocr, hindi_vendor) × 3 levels; cluster structure independent of level | Difficulty is the only variable between levels; one split serves all. |
| 2026-11-25 | Synth | Held-out abbreviations, city aliases and Hindi loanwords (generator-only) | The normaliser shares vocabulary with the generator; held-out variants keep evaluation honest. |
| 2026-11-25 | Synth | Synthetic GSTIN: valid format/checksum, reserved `ZZZ` PAN prefix, bijective per-entity serial | Random digits produced 10k cross-entity collisions; real GSTINs are unique; never real values. |
| 2026-11-25 | Eval | Entity-disjoint 70/15/15 split, stratified by cluster size × state | No entity crosses splits. |
| 2026-11-25 | Eval | Every M2+ metric reported with and without identifier features | Exact CIN/GSTIN alone gives recall 0.71 on easy — a shortcut that would inflate results. |
| 2026-11-25 | Eval | WDC loaders drop `cluster_id_*` | It is the label. |
| 2026-11-25 | Tooling | uv + `uv.lock`; GNU make wrapping `uv run python -m ekam.…`; Hydra configs; local MLflow (SQLite) | Reproducible one-command stages. |
| 2026-11-25 | Tooling | Heavy ML dependencies added in the milestone that first needs them | Small install for early milestones; the lock file stays the single pin. |
| 2026-11-26 | Blocking | Sentence encoder for nearest-neighbour blocking: `intfloat/multilingual-e5-small` (118M, 384-d) | Fits 6 GB VRAM, handles Devanagari; MiniLM-multilingual kept as an alternative config. |
| 2026-11-26 | Blocking | Rules = OR of 4 conjunctions (PIN+first token, phonetic key, first token+city, exact name key), blocks > 200 records skipped; plus FAISS kNN (k=10) on name+city embeddings; union is the default candidate set | Rules alone keep 94.5% / 84.5% / 67.5% of true pairs (easy/medium/hard); kNN is needed for the noisy levels. |
| 2026-11-26 | Matching | All matchers score the *same* candidate pairs; thresholds chosen on the valid split; report candidate-level and end-to-end (blocking-inclusive) P/R/F1 | Separates blocking loss from matcher quality; fair comparison. |
| 2026-11-26 | Splink | Splink 5 (DuckDB), unsupervised: λ from exact name+PIN, u by random sampling, m by EM on (PIN, first token) and (phonetic key, city); scores our blocked pairs via `register_blocked_pairs_for_predict` | EM on PIN alone = 1.5e8 pairs (one Bengaluru PIN has 5.5k companies) and ran the laptop out of memory. |
| 2026-11-26 | Tooling | torch: CUDA 12.8 build from the PyTorch index on Windows, PyPI elsewhere; matplotlib added for the blocking curve | GPU for embeddings (M2) and fine-tuning (M3); one plot dependency. |
