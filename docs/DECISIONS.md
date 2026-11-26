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
| 2026-11-26 | Generator | New noise type `addr_other`: a record gets another company's real registered address (same city, else same state) at 0.15 / 0.05 / 0.30 / 0.20 (erp / gst_portal / crm_ocr / hindi_vendor, medium); data regenerated and M2 re-run | First M2 run: the street line acted as a near-identifier (hard: LightGBM ~80% gain on street features). After the fix the street adds +0.007 F1 at hard (was +0.032). |
| 2026-11-26 | M3 encoder | Cross-encoder = `intfloat/multilingual-e5-small` fine-tuned with a 1-logit head (Ditto-style `COL … VAL …` serialisation); MiniLM-multilingual as an ablation | Already cached, fits 6 GB, multilingual (Devanagari). MuRIL-base not used (larger download, ~2× slower). |
| 2026-11-26 | M3 cascade | LightGBM (M2 `no_ids`) pre-filter keeps 99.5% of valid true candidates; the cross-encoder scores only survivors; pre-filtered pairs count as non-matches in every metric | The laptop GPU scores ~1k pairs/s; all 0.9M candidates per split would take ~15 min per model and split. The pre-filter removes ~80% of candidates. |
| 2026-11-26 | M3 training data | Cross-encoder trains on train-split survivors chosen with out-of-fold LightGBM scores | Training pairs then look like test pairs; no self-scoring. |
| 2026-11-26 | M3 augmentation | Ditto's generic operators (span delete/shuffle, attribute delete/shuffle, side swap), not the generator's noise functions | Augmenting with the generator's own noise would teach the generator, not matching. |
| 2026-11-26 | M3 active learning | Pool/eval = blocked candidates (not cascade survivors); uncertainty vs random; F1 at a fixed 0.5 threshold + average precision | The pre-filter itself needs many labels; a team starting from zero has no tuned threshold. |
| 2026-11-26 | M3 LLM comparison | Skipped | Costs money / needs a key; not approved. |
