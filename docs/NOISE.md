# Noise model of the synthetic sources

`ekam.synth.generator` turns each clean MCA company into 0–N records in four "source
systems". This page documents every kind of noise, how it is configured, and where the
generator is knowingly **easier or harder than reality**. Measured rates are in
[`reports/M1_audit.md`](reports/M1_audit.md).

## How a record is made

1. **Structure (same at every difficulty level).** For each (entity, source) a seeded RNG
   decides whether the entity is present (`coverage`) and how many in-source duplicates it
   has (each extra copy with probability `dup_rate`, at most 6).
2. **Noise.** Each noise type fires independently with probability
   `min(base_rate × level_multiplier, 0.95)`, in a fixed order (name, then address, city,
   state, PIN, identifiers). Every decision is logged in the ground-truth file:
   `noise_applied` (it changed the value) or `noise_ineligible` (drawn, but there was
   nothing to change, e.g. no PIN to corrupt).
3. **Schema.** The record is written in the source's own column names; fields a source does
   not hold (e.g. CIN in the GST portal) are absent, not empty. `crm_ocr` writes one
   free-text address line containing street, city, state and PIN.

Levels: **easy ×0.4**, **medium ×1.0**, **hard ×1.8** (hard also allows 2 typo edits).
Configs: `conf/sources/*.yaml`, `conf/noise/*.yaml`.

## Sources

| source | coverage | in-source dup rate | address format | schema (canonical → native) |
|---|---:|---:|---|---|
| `erp` — in-house ERP vendor master | 0.75 | 0.05 | split | name→vendor_name, address→address_line1, city→city, state→state, pin→pincode, cin→cin, gstin→gst_no |
| `gst_portal` — GST registration export | 0.60 | 0.02 | split, forced UPPER case | name→legal_name, address→principal_place, city→district, state→state_name, pin→pin_code, gstin→gstin |
| `crm_ocr` — CRM fed by OCR'd invoices | 0.50 | 0.15 | single line | name→company, address→address, cin→cin_no, gstin→gstin_no |
| `hindi_vendor` — regional Hindi register | 0.40 | 0.05 | split | name→firm_name, address→pata, city→shahar, state→rajya, pin→pin, gstin→gstin |

## Noise types

Examples are real generator outputs.

| noise type | what it does | example |
|---|---|---|
| `name_suffix_variant` | re-renders the legal form with another variant of the *same* form (drawn only if `name_suffix_drop` did not fire) | SHARMA TRADERS PRIVATE LIMITED → SHARMA TRADERS (P) Ltd |
| `name_suffix_drop` | removes the legal form | → SHARMA TRADERS |
| `name_abbrev` | abbreviates business words using variants **the normaliser knows** | YASHASWI HOUSING PROJECT AND INVESTMENTS → … & INVTS |
| `name_abbrev_heldout` | abbreviations **the normaliser has never seen** (`heldout` in abbreviations.yaml) | TRADERS → TRDR, INDUSTRIES → INDSTR |
| `name_case` | title / lower / mixed case | praia da goa real estates private limited |
| `name_typo` | 1 (hard: up to 2) keyboard edits: neighbour substitution, deletion, insertion, transposition; never the first letter | ESTATES → ESTAFTES |
| `name_ocr` | 1–3 OCR confusions (O↔0, I↔1↔L, S↔5, B↔8, RN↔M, CL↔D, VV↔W, E→F …) | PRAIA DA GOA REAL → PRAIA DA 6Oa RFAL |
| `name_token_drop` | drops a non-first word | PRAIA DA GOA REAL ESTATES → PRAIA DA GOA ESTATES |
| `name_token_swap` | swaps two adjacent words | |
| `name_spacing` | glues two words or splits one | DA GOA REAL → DA GOAREAL |
| `name_translit_variant` | re-spells 1–2 Indian words with romanisation variants (sh↔s, ee↔i, aa↔a, w↔v, ksh↔x, ph→f, doubled consonants, Shri/Shree/Sri, final -a) — English business words untouched | SHARMA → SARMA, LAXMI → LAKSHMI, AGARWAL → AGGARWAL |
| `name_devanagari` | renders the whole name in Devanagari: business loanwords from a lexicon, everything else via heuristic Latin→ITRANS→Devanagari | YASHASWI HOUSING PROJECT … PRIVATE LIMITED → यशस्वी हाउसिंग प्रोजेक्ट … प्रायवेट लिमिटेड |
| `name_prefix_ms` | adds a trade prefix | → M/s. SHARMA TRADERS |
| `addr_abbrev` | ROAD→RD, NEAR→NR, OPPOSITE→OPP, SECTOR→SEC, INDUSTRIAL→INDL … (70% of eligible words) | OPPOSITE TIMBLO BHAVAN → OPP. TIMBLO BHAVAN |
| `addr_reorder` | moves a block of the street line to the other end ("A B C D" → "C D, A B") | |
| `addr_truncate` | keeps the first 40–75% of the street tokens | |
| `addr_typo`, `addr_ocr` | as for names, on the street line | CHAMBERS → CNAM8ERS |
| `city_alias` | old name / spelling / abbreviation the normaliser knows (Bombay, Bangalore, Baroda, Gurgaon, Allahabad, Panjim …) | Margao → Madgaon |
| `city_alias_heldout` | city spellings the normaliser has never seen | Bengaluru → Bangaloor |
| `city_missing` | city dropped | |
| `state_variant` | code or variant (UP, U.P., KA, KAR, Gujrat, Goa State, upper/lower case) | Karnataka → KAR |
| `pin_missing` | PIN dropped | |
| `pin_wrong` | realistic wrong PIN: last-digit slip (45%), adjacent transposition (30%), another PIN from the same postal circle (25%) | 560070 → 560071 |
| `cin_missing`, `gstin_missing` | identifier dropped | |
| `cin_typo`, `gstin_typo` | one keying error (substitution or transposition); usually breaks the GSTIN checksum | |

### Base rates per source (difficulty medium; × 0.4 easy, × 1.8 hard, capped at 0.95)

"—" = the source does not use this noise type (or does not hold the field).

| noise type | erp | gst_portal | crm_ocr | hindi_vendor |
|---|---:|---:|---:|---:|
| `addr_abbrev` | 0.40 | 0.30 | 0.40 | 0.20 |
| `addr_ocr` | — | — | 0.30 | — |
| `addr_reorder` | — | 0.40 | — | — |
| `addr_truncate` | — | 0.10 | 0.30 | — |
| `addr_typo` | 0.05 | — | 0.20 | 0.05 |
| `cin_missing` | 0.30 | — | 0.85 | — |
| `cin_typo` | 0.02 | — | 0.10 | — |
| `city_alias` | 0.15 | 0.10 | 0.20 | 0.35 |
| `city_alias_heldout` | — | — | — | 0.10 |
| `city_missing` | — | — | 0.20 | — |
| `gstin_missing` | 0.40 | 0.05 | 0.80 | 0.50 |
| `gstin_typo` | 0.02 | 0.03 | 0.10 | 0.05 |
| `name_abbrev` | 0.25 | 0.05 | 0.30 | — |
| `name_abbrev_heldout` | 0.03 | — | 0.10 | — |
| `name_case` | 0.60 | — (always upper) | 0.40 | 0.40 |
| `name_devanagari` | — | — | — | 0.40 |
| `name_ocr` | — | — | 0.35 | — |
| `name_prefix_ms` | 0.05 | — | — | 0.30 |
| `name_spacing` | — | — | 0.15 | — |
| `name_suffix_drop` | 0.05 | 0.02 | 0.25 | 0.10 |
| `name_suffix_variant` | 0.50 | 0.30 | 0.40 | 0.40 |
| `name_token_drop` | — | — | 0.15 | — |
| `name_token_swap` | — | — | — | 0.10 |
| `name_translit_variant` | — | — | — | 0.40 |
| `name_typo` | 0.05 | 0.03 | 0.20 | 0.05 |
| `pin_missing` | 0.10 | 0.02 | 0.30 | 0.20 |
| `pin_wrong` | 0.02 | 0.03 | 0.08 | 0.10 |
| `state_variant` | 0.30 | 0.05 | 0.30 | 0.40 |

Note that the *base* MCA records are already noisy (23% without PIN, ~9% without a
parseable city, suffix and spelling variation), so realised field completeness is lower
than these rates alone imply — see the audit.

## Identifiers

* **CIN** is the real, public company identifier from MCA (not personal data). It is what
  defines an entity, so an intact CIN is a perfect match signal. That is realistic, but it
  is also the main shortcut: **every M2+ result is reported with and without identifier
  features.** Audit: an exact CIN/GSTIN matcher alone reaches recall 0.71 / 0.29 / 0.08
  (easy / medium / hard) at precision 0.9999.
* **GSTIN** is **always synthetic**: valid 15-character format and mod-36 checksum; state
  code from the company's state; PAN block `ZZZ` + `C` + name initial + 4 digits + letter;
  the digits, letter and entity-number character encode a scrambled per-entity serial
  (bijective, so two companies never share a GSTIN, as in reality). `ZZZ` marks it as
  synthetic; `is_synthetic_gstin()` is asserted in tests. No real GSTIN is used or derived.

## Known limitations (stated in the README)

* **The normaliser shares vocabulary with the generator.** Legal-suffix forms, the known
  abbreviations, the known city aliases and the core Hindi loanword lexicon appear in both,
  so on those noise types normalisation is an *upper bound* on real-world benefit.
  Mitigation: **held-out** variants (`heldout` abbreviations and city aliases,
  `hindi_lexicon_heldout.yaml`) are emitted only by the generator, and evaluation can be
  sliced by the `noise_applied` log to report them separately.
* **Devanagari rendering is heuristic.** Proper names go through simple rules plus ITRANS
  (a hand check of 25 names: Indian names mostly plausible — शर्मा, राम, लक्ष्मी, कुमार; English
  words outside the lexicon are rendered letter-by-letter, e.g. CAKEZONE → ककेज़ोने, which a
  Hindi writer would spell केकज़ोन). Measured on 3,000 names, the normaliser recovers the
  exact name key after Devanagari noise **78.6%** of the time and after romanisation
  variants **97.0%** of the time. A learned transliterator (e.g. AI4Bharat IndicXlit) would
  be more realistic; it is a candidate M3 ablation, not a dependency today.
* **Street lines are too stable (found in M2).** Every record of an entity starts from the
  same clean MCA street; noise only abbreviates, re-orders, truncates or mistypes it. At the
  hard level 64% of true pairs still share an identical normalised street, so a supervised
  model can treat the street as a near-identifier (LightGBM: ~80% of gain on street features;
  removing them drops hard-level F1 from 0.975 to 0.943). Real vendor masters often hold a
  *different* address for the same company (works, branch, billing office). M2 therefore also
  reports a `no_street` diagnostic.
* **Noise is independent per type.** Real systems have correlated errors (a sloppy
  data-entry clerk makes many). Cluster structure is also simpler than reality (no mergers,
  renames, branch offices or parent/subsidiary links).
* **Hard negatives come only from the register itself**: genuinely similar companies
  (207 groups of distinct CINs with the same normalised name and PIN) and near-duplicates.
  The generator does not invent confusable non-matches.
* **Snapshot age:** the MCA data is from March 2015.
