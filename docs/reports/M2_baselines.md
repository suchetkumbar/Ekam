# Milestone 2 — baselines

## Synthetic sources (test split, entity-disjoint)

Matchers score the same blocked candidates (rules ∪ kNN). `precision/recall/f1` are over candidate pairs; `recall_e2e`/`f1_e2e` also count true pairs lost in blocking. Thresholds chosen on the valid split. `no_ids` removes CIN/GSTIN features; `no_street` (diagnostic) also removes street-line features.

Blocking pair completeness on test: easy 0.996, medium 0.971, hard 0.887.

| level | variant | model | precision | recall | f1 | recall_e2e | f1_e2e | test_pairs | seconds |
|---|---|---|---|---|---|---|---|---|---|
| easy | no_ids | lightgbm | 0.994 | 0.988 | 0.991 | 0.984 | 0.989 | 908646 | 38.329 |
| easy | no_ids | logreg | 0.982 | 0.969 | 0.976 | 0.965 | 0.973 | 908646 | 2.529 |
| easy | no_ids | splink | 0.933 | 0.964 | 0.948 | 0.960 | 0.946 | 908646 | 10.698 |
| easy | no_street | lightgbm | 0.991 | 0.982 | 0.987 | 0.978 | 0.984 | 908646 | 38.476 |
| easy | no_street | logreg | 0.978 | 0.955 | 0.966 | 0.951 | 0.964 | 908646 | 2.368 |
| easy | with_ids | lightgbm | 0.998 | 0.996 | 0.997 | 0.992 | 0.995 | 908646 | 29.432 |
| easy | with_ids | logreg | 0.995 | 0.990 | 0.992 | 0.986 | 0.990 | 908646 | 2.397 |
| easy | with_ids | splink | 0.940 | 0.986 | 0.963 | 0.982 | 0.961 | 908646 | 13.327 |
| hard | no_ids | lightgbm | 0.962 | 0.935 | 0.948 | 0.829 | 0.891 | 891398 | 41.379 |
| hard | no_ids | logreg | 0.910 | 0.850 | 0.879 | 0.753 | 0.824 | 891398 | 2.444 |
| hard | no_ids | splink | 0.806 | 0.865 | 0.834 | 0.767 | 0.786 | 891398 | 20.780 |
| hard | no_street | lightgbm | 0.958 | 0.924 | 0.941 | 0.819 | 0.883 | 891398 | 61.361 |
| hard | no_street | logreg | 0.903 | 0.833 | 0.867 | 0.739 | 0.813 | 891398 | 5.111 |
| hard | with_ids | lightgbm | 0.965 | 0.938 | 0.951 | 0.832 | 0.894 | 891398 | 40.844 |
| hard | with_ids | logreg | 0.921 | 0.856 | 0.887 | 0.759 | 0.832 | 891398 | 3.115 |
| hard | with_ids | splink | 0.807 | 0.868 | 0.837 | 0.770 | 0.788 | 891398 | 11.520 |
| medium | no_ids | lightgbm | 0.982 | 0.967 | 0.975 | 0.939 | 0.960 | 902216 | 38.772 |
| medium | no_ids | logreg | 0.957 | 0.920 | 0.938 | 0.893 | 0.924 | 902216 | 2.480 |
| medium | no_ids | splink | 0.949 | 0.854 | 0.899 | 0.829 | 0.885 | 902216 | 10.428 |
| medium | no_street | lightgbm | 0.976 | 0.956 | 0.966 | 0.928 | 0.951 | 902216 | 21.578 |
| medium | no_street | logreg | 0.949 | 0.899 | 0.923 | 0.872 | 0.909 | 902216 | 2.167 |
| medium | with_ids | lightgbm | 0.985 | 0.975 | 0.980 | 0.946 | 0.965 | 902216 | 38.556 |
| medium | with_ids | logreg | 0.969 | 0.936 | 0.952 | 0.908 | 0.938 | 902216 | 2.686 |
| medium | with_ids | splink | 0.953 | 0.878 | 0.914 | 0.853 | 0.900 | 902216 | 11.359 |

LightGBM top features (gain share):

- easy / no_ids: name_token_set:0.47, name_jw:0.25, name_partial:0.13, street_token_set:0.06, street_jw:0.02
- easy / no_street: name_token_set:0.49, name_jw:0.37, name_partial:0.05, name_lev:0.02, name_token_sort:0.02
- easy / with_ids: name_jw:0.53, name_token_set:0.32, street_token_set:0.04, gstin_eq:0.04, gstin_both:0.02
- hard / no_ids: name_jw:0.52, name_token_sort:0.14, name_token_set:0.10, name_partial:0.05, name_lev:0.05
- hard / no_street: name_jw:0.67, name_token_set:0.07, name_token_sort:0.07, name_partial:0.05, name_lev:0.04
- hard / with_ids: name_jw:0.52, name_token_set:0.20, name_partial:0.05, name_token_sort:0.04, name_lev:0.04
- medium / no_ids: name_jw:0.55, name_partial:0.13, name_token_set:0.12, street_token_set:0.06, name_token_sort:0.03
- medium / no_street: name_jw:0.71, name_token_set:0.10, name_partial:0.05, name_lev:0.03, name_token_sort:0.03
- medium / with_ids: name_jw:0.54, name_token_set:0.22, street_token_set:0.06, name_partial:0.03, name_token_sort:0.02

## Public benchmarks (published train/valid/test pair splits)

Attribute-wise Jaro-Winkler, token-set/sort ratios, char TF-IDF cosine and numeric differences; same LR / LightGBM protocol. DeepMatcher splits are pair-level (not entity-disjoint); WDC `test_unseen*` contain products unseen in training.

| dataset | level | model | precision | recall | f1 | test_pairs |
|---|---|---|---|---|---|---|
| abt_buy | test | lightgbm | 0.699 | 0.801 | 0.747 | 1916 |
| abt_buy | test | logreg | 0.795 | 0.660 | 0.721 | 1916 |
| amazon_google | test | lightgbm | 0.571 | 0.667 | 0.615 | 2293 |
| amazon_google | test | logreg | 0.517 | 0.731 | 0.605 | 2293 |
| company | test | lightgbm | 0.770 | 0.766 | 0.768 | 22503 |
| company | test | logreg | 0.776 | 0.761 | 0.768 | 22503 |
| dblp_acm | test | lightgbm | 0.961 | 1.000 | 0.980 | 2473 |
| dblp_acm | test | logreg | 0.986 | 0.971 | 0.978 | 2473 |
| walmart_amazon | test | lightgbm | 0.890 | 0.715 | 0.793 | 2049 |
| walmart_amazon | test | logreg | 0.723 | 0.689 | 0.706 | 2049 |
| wdc_products_80pair | test | lightgbm | 0.354 | 0.548 | 0.430 | 4500 |
| wdc_products_80pair | test | logreg | 0.336 | 0.392 | 0.362 | 4500 |
| wdc_products_80pair | test_unseen100 | lightgbm | 0.343 | 0.610 | 0.439 | 4500 |
| wdc_products_80pair | test_unseen100 | logreg | 0.305 | 0.466 | 0.369 | 4500 |
| wdc_products_80pair | test_unseen50 | lightgbm | 0.351 | 0.570 | 0.434 | 4500 |
| wdc_products_80pair | test_unseen50 | logreg | 0.332 | 0.460 | 0.386 | 4500 |
