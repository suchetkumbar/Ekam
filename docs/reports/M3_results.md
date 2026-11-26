# Milestone 3 — cross-encoder, ablations, active learning (generated)

Cascade = blocked candidates → LightGBM pre-filter (keeps 99.5% of valid true candidates) → Ditto-style cross-encoder (`intfloat/multilingual-e5-small`). Metrics over **all** blocked test candidates (pre-filtered pairs count as non-matches); `f1_e2e` also counts true pairs lost in blocking. Thresholds chosen on valid.

## Main results

| level | run | precision | recall | f1 | f1_e2e | best_m2_model | best_m2_f1 | best_m2_f1_e2e | prefilter_kept | prefilter_recall | train_pairs | best_epoch | train_seconds | score_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy | no_ids | 0.9924 | 0.9908 | 0.9916 | 0.9896 | lightgbm | 0.9907 | 0.9886 | 0.1089 | 0.9951 | 120000 | 2 | 2266 | 431 |
| hard | no_ids | 0.9831 | 0.9706 | 0.9768 | 0.9177 | lightgbm | 0.9484 | 0.8906 | 0.1914 | 0.9947 | 120000 | 2 | 2392 | 694 |
| hard | with_ids | 0.9821 | 0.9747 | 0.9784 | 0.9194 | lightgbm | 0.9515 | 0.8935 | 0.1914 | 0.9947 | 120000 | 2 | 2802 | 802 |
| medium | no_ids | 0.9908 | 0.9806 | 0.9857 | 0.9710 | lightgbm | 0.9746 | 0.9600 | 0.1395 | 0.9952 | 120000 | 2 | 2248 | 448 |

## Where the gain comes from (hard, no_ids)

Recall of true test candidate pairs: LightGBM alone (M2 protocol) vs the cascade, sliced by noise type on either record and by source pair. False positives: LightGBM 3609, cascade 1436. Leakage check: test records whose normalised name occurs in the train split: 0.32% (entities are disjoint, so memorising names cannot help).

| slice | true_pairs | lgbm_recall | cascade_recall | gain |
|---|---|---|---|---|
| name_token_drop | 5298 | 0.6019 | 0.8464 | 0.2444 |
| name_abbrev_heldout | 1676 | 0.7983 | 0.9505 | 0.1521 |
| src:crm_ocr+crm_ocr | 2885 | 0.7567 | 0.8835 | 0.1269 |
| name_ocr | 22070 | 0.8169 | 0.9365 | 0.1197 |
| src:crm_ocr+hindi_vendor | 7233 | 0.8280 | 0.9389 | 0.1109 |
| name_spacing | 9624 | 0.8303 | 0.9252 | 0.0949 |
| addr_ocr | 20590 | 0.8636 | 0.9441 | 0.0805 |
| name_typo | 21743 | 0.8623 | 0.9323 | 0.0700 |
| name_abbrev | 15689 | 0.8959 | 0.9635 | 0.0677 |
| name_token_swap | 5516 | 0.9012 | 0.9648 | 0.0636 |
| src:crm_ocr+erp | 16223 | 0.8929 | 0.9530 | 0.0600 |
| addr_truncate | 27425 | 0.8955 | 0.9544 | 0.0589 |
| addr_typo | 20487 | 0.8942 | 0.9529 | 0.0587 |
| src:crm_ocr+gst_portal | 12475 | 0.8996 | 0.9566 | 0.0570 |
| addr_other | 44268 | 0.8948 | 0.9493 | 0.0545 |
| city_missing | 11830 | 0.9015 | 0.9527 | 0.0511 |
| cin_missing | 56896 | 0.9129 | 0.9618 | 0.0489 |
| pin_missing | 31536 | 0.9165 | 0.9631 | 0.0466 |
| cin_typo | 1347 | 0.9213 | 0.9629 | 0.0416 |
| pin_wrong | 7878 | 0.9273 | 0.9653 | 0.0381 |
| gstin_missing | 78134 | 0.9314 | 0.9686 | 0.0373 |
| state_variant | 62311 | 0.9336 | 0.9693 | 0.0357 |
| city_alias | 37433 | 0.9343 | 0.9699 | 0.0356 |
| addr_abbrev | 47599 | 0.9323 | 0.9678 | 0.0354 |
| name_case | 74759 | 0.9357 | 0.9706 | 0.0349 |
| name_suffix_variant | 72223 | 0.9382 | 0.9726 | 0.0345 |
| name_devanagari | 23021 | 0.9450 | 0.9761 | 0.0311 |
| name_translit_variant | 22288 | 0.9460 | 0.9737 | 0.0277 |
| name_prefix_ms | 20933 | 0.9560 | 0.9825 | 0.0265 |
| name_suffix_drop | 27384 | 0.9312 | 0.9574 | 0.0262 |
| gstin_typo | 3263 | 0.9534 | 0.9733 | 0.0199 |
| city_alias_heldout | 3595 | 0.9574 | 0.9772 | 0.0197 |
| addr_reorder | 28964 | 0.9644 | 0.9822 | 0.0178 |
| src:hindi_vendor+hindi_vendor | 918 | 0.9826 | 0.9902 | 0.0076 |
| src:erp+hindi_vendor | 13468 | 0.9792 | 0.9846 | 0.0053 |
| src:gst_portal+hindi_vendor | 10496 | 0.9843 | 0.9869 | 0.0027 |
| src:erp+erp | 1731 | 0.9965 | 0.9971 | 0.0006 |
| src:erp+gst_portal | 20113 | 0.9961 | 0.9955 | -0.0006 |
| src:gst_portal+gst_portal | 527 | 1.0000 | 0.9981 | -0.0019 |
