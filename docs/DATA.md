# Data sources and decisions

Every dataset Ekam uses, where it came from, under what terms, and what we decided.
Raw data lives under `data/` (gitignored) and is fetched by `make data`; exact URLs,
byte sizes and SHA-256 checksums are in `data/raw/*/manifest.json` after a run.
Checked on **2026-11-25**.

## Decision log

| # | Source | Decision | Why |
|---|---|---|---|
| 1 | MCA company master data (data.gov.in) | **Used — primary clean source** | Public, keyless download, GODL-India licence, one row per real company with a unique CIN → exact entity ground truth. |
| 2 | Amazon ML Challenge 2026 (business entity resolution) | **Not used** | Only unofficial Kaggle/Hugging Face mirrors exist (one says it is a personal copy whose content belongs to Amazon); no public licence. Cited as motivation only. |
| 3 | IndiaAI Innovation Challenge 2026 — MSME TEAMS vendor mapping (AIKosh) | **Not used** | No dataset is published for this track (applications closed March 2026). Cited as motivation only. |
| 4 | AIKosh "Company Master Data" | Not needed | Mirrors the MCA data; kept as a fallback if data.gov.in stops serving the files. |
| 5 | DeepMatcher/Magellan benchmarks: Abt-Buy, Amazon-Google, DBLP-ACM, Walmart-Amazon, Company | **Used — local only** | Standard EM benchmarks for comparability. No explicit redistribution licence → never committed or bundled. |
| 6 | WDC Products (80% corner cases) | **Used — local only** | Modern, hard product-matching benchmark with unseen-entity test sets. Licence not stated on the download page → never committed or bundled. |
| 7 | Synthetic multi-source data generated from MCA | **Used — main training/eval data** | Real Indian records have no labelled duplicates; the generator gives exact ground truth with controlled India-specific noise (docs/NOISE.md). |

## 1. MCA company master data

* **Publisher:** Ministry of Corporate Affairs via the Open Government Data Platform India.
* **Catalog:** "Company Master Data", catalog UUID `ec58dab7-d891-4abb-936e-d5d274a6ce9b`,
  resource "Registrars of Companies (RoC)-wise Company Master Data".
* **Files:** one CSV per state,
  `https://data.gov.in/sites/default/files/dataurl15092015/company_master_data_upto_Mar_2015_<State>.csv`
  (state names with underscores, e.g. `Uttar_Pradesh`). Located through the public OGD
  backend API (`/backend/dmspublic/v1/resources?filters[catalog_title]=Company Master Data`);
  **no API key was needed.**
* **Snapshot:** "up to March 2015". This is old (companies registered later are absent and
  statuses are stale), which does not matter for entity resolution: we need realistic
  names/addresses, not current status.
* **Licence:** Government Open Data License – India (GODL-India): free use, adaptation and
  redistribution with attribution. Attribution: *"Company master data, Ministry of Corporate
  Affairs, Government of India, via data.gov.in, GODL-India."*
* **Subset for Milestone 1:** Goa, Karnataka, Uttar Pradesh, Gujarat, Rajasthan — a mix of a
  Hindi-belt state, a southern state, a western state and a small one.

  | State | Rows | Bytes |
  |---|---:|---:|
  | Goa | 7,353 | 2.2 MB |
  | Gujarat | 81,184 | 25.9 MB |
  | Karnataka | 78,524 | 25.7 MB |
  | Rajasthan | 45,804 | 14.1 MB |
  | Uttar Pradesh | 65,165 | 20.5 MB |
  | **Total** | **278,030** | **88 MB** |

  The full country (all state files, expected ~1.5 M+ companies, well under 1 GB) is
  fetched in Milestone 4 for the ≥1 M-record scale test by adding states to
  `conf/data/mca.yaml`.
* **Columns (13):** CIN, date of registration, name, status, class, category, authorised
  capital, paid-up capital, registered state, RoC, principal business activity,
  registered-office address, sub-category.
* **Privacy:** this snapshot has **no personal columns** (no director names, DIN, e-mail or
  phone). Ingest still enforces an **allow-list** (`conf/data/mca.yaml: allow_columns`), so a
  future snapshot that adds such columns cannot leak them; a second guard refuses any
  output column whose name looks personal. Capital columns are dropped as unnecessary.
* **Quality observed (raw):** 0 duplicate CINs; 19 malformed CINs (kept, flagged
  `cin_valid=false`); 167 exact-name collisions between different CINs (genuine
  homonyms — useful hard negatives); 23% of addresses have no PIN; suffix variants
  already present (PRIVATE LIMITED 233,646 · PVT LTD 13,543 · PVT. LTD. 1,734 …);
  inconsistent city spellings (AHAMEDABAD, BANGALORE) and glued words
  (HOUSEMITHAKHALI). We parse but do not "fix" these: they are the starting point.
* **After ingest** (`data/interim/mca_entities.parquet`): 278,030 entities; city parsed for
  91%, PIN for 78%, CIN valid for 99.99%.

## 2. Public entity-matching benchmarks

Downloaded as zips, extracted with a zip-slip-safe extractor. Counts are pairs (positives).

| Dataset | Source | Table A / B | Attributes | Train | Valid | Test |
|---|---|---:|---|---:|---:|---:|
| Abt-Buy | DeepMatcher (Textual) | 1,081 / 1,092 | name, description, price | 5,743 (616) | 1,916 (206) | 1,916 (206) |
| Amazon-Google | DeepMatcher (Structured) | 1,363 / 3,226 | title, manufacturer, price | 6,874 (699) | 2,293 (234) | 2,293 (234) |
| DBLP-ACM | DeepMatcher (Structured) | 2,616 / 2,294 | title, authors, venue, year | 7,417 (1,332) | 2,473 (444) | 2,473 (444) |
| Walmart-Amazon | DeepMatcher (Structured) | 2,554 / 22,074 | title, category, brand, modelno, price | 6,144 (576) | 2,049 (193) | 2,049 (193) |
| Company | DeepMatcher (Textual) | 28,200 / 28,200 | content (long text) | 67,596 (16,859) | 22,533 (5,701) | 22,503 (5,640) |
| WDC Products 80% cc | Univ. Mannheim | pair files | title, brand, description, price, priceCurrency | 19,835 (8,471) | 4,500 (500) | 4,500 (500) + unseen-50% / unseen-100% test sets |

* URLs: `https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/...` (per-dataset `*_exp_data.zip`)
  and `https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip`.
* **Licences:** neither page states a redistribution licence; the DeepMatcher sets derive
  from earlier research benchmarks (Köpcke et al. 2010 and others) used widely for research.
  **Decision:** local use only, always cited, never committed or shipped in the demo.
* **Known caveats (handled in M2):**
  * **WDC pair files contain `cluster_id_left/right` — that *is* the label.** Loaders must
    drop it before feature extraction; a test will enforce this.
  * DeepMatcher splits are pair-level, not entity-disjoint (the same record can appear in
    train and test pairs). We report on the published splits for comparability and say so.
  * "Company" is long free text (Wikipedia-style), not structured name/address; it is the
    closest public company-matching set, not an Indian one.

## 3. Synthetic multi-source data (the main dataset)

Built by `ekam.synth.generator` from the 278,030 MCA entities; full description of every
noise type in [`docs/NOISE.md`](NOISE.md), measured statistics in
[`docs/reports/M1_audit.md`](reports/M1_audit.md).

* **4 source systems** with their own schemas and noise profiles: `erp`, `gst_portal`,
  `crm_ocr`, `hindi_vendor` (configs in `conf/sources/`).
* **3 difficulty levels** (easy ×0.4, medium ×1.0, hard ×1.8 on every noise rate), sharing the
  same cluster structure.
* **669,893 records per level**, 645,547 true matching pairs; cluster sizes 1–9 (18% of
  entities are singletons, 35% pairs, 30% triples).
* **Ground truth** in a separate file (`record_id → entity_id`); `record_id` is a random UUID.
* **Synthetic GSTINs:** valid format and checksum, but PAN block always starts with the
  reserved prefix `ZZZ`, and values are a scrambled per-entity serial — never real GSTINs.
* **Entity split:** 70/15/15 by entity, stratified by cluster size × state
  (`data/processed/entity_split.parquet`): 194,623 / 41,705 / 41,702 entities.
* **Leakage checks** (all pass, enforced by tests): entity-disjoint split; every record
  inherits its entity's split; no entity id in any field; file order random; identifier
  shortcut measured (an exact CIN/GSTIN matcher reaches recall 0.71 / 0.29 / 0.08 on
  easy / medium / hard at precision 0.9999 — so M2+ reports everything with and without
  identifier features).

## Reproducing

```bash
make setup
make data          # ~3 min after downloads on the reference laptop
make test
```

Same seed → byte-identical `records.parquet` / `ground_truth.parquet` regardless of worker
count or chunk size (verified; checksums in `data/synth/<level>/manifest.json`).
