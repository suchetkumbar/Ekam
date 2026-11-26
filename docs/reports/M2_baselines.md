# Milestone 2 — baselines

## Synthetic sources (test split, entity-disjoint)

Matchers score the same blocked candidates (rules ∪ kNN). `precision/recall/f1` are over candidate pairs; `recall_e2e`/`f1_e2e` also count true pairs lost in blocking. Thresholds chosen on the valid split. `no_ids` removes CIN/GSTIN features; `no_street` (diagnostic) also removes street-line features.

Blocking pair completeness on test: easy 0.996, medium 0.972, hard 0.890.

| level | variant | model | precision | recall | f1 | recall_e2e | f1_e2e | test_pairs | seconds |
|---|---|---|---|---|---|---|---|---|---|
| easy | no_ids | lightgbm | 0.995 | 0.994 | 0.995 | 0.990 | 0.993 | 907373 | 28.334 |
| easy | no_ids | logreg | 0.980 | 0.989 | 0.985 | 0.985 | 0.983 | 907373 | 2.018 |
| easy | no_ids | splink | 0.966 | 0.967 | 0.966 | 0.963 | 0.965 | 907373 | 10.331 |
| easy | no_street | lightgbm | 0.994 | 0.980 | 0.987 | 0.977 | 0.985 | 907373 | 21.554 |
| easy | no_street | logreg | 0.976 | 0.959 | 0.967 | 0.955 | 0.966 | 907373 | 2.197 |
| easy | with_ids | lightgbm | 0.998 | 0.998 | 0.998 | 0.994 | 0.996 | 907373 | 15.280 |
| easy | with_ids | logreg | 0.994 | 0.996 | 0.995 | 0.992 | 0.993 | 907373 | 2.417 |
| easy | with_ids | splink | 0.976 | 0.985 | 0.981 | 0.981 | 0.979 | 907373 | 12.876 |
| hard | no_ids | lightgbm | 0.978 | 0.972 | 0.975 | 0.866 | 0.919 | 891646 | 27.276 |
| hard | no_ids | logreg | 0.953 | 0.940 | 0.947 | 0.837 | 0.891 | 891646 | 2.367 |
| hard | no_ids | splink | 0.943 | 0.870 | 0.905 | 0.774 | 0.850 | 891646 | 9.563 |
| hard | no_street | lightgbm | 0.964 | 0.924 | 0.943 | 0.823 | 0.888 | 891646 | 36.669 |
| hard | no_street | logreg | 0.911 | 0.843 | 0.876 | 0.751 | 0.823 | 891646 | 2.120 |
| hard | with_ids | lightgbm | 0.982 | 0.972 | 0.977 | 0.866 | 0.920 | 891646 | 40.142 |
| hard | with_ids | logreg | 0.952 | 0.951 | 0.952 | 0.847 | 0.897 | 891646 | 2.498 |
| hard | with_ids | splink | 0.916 | 0.878 | 0.897 | 0.781 | 0.844 | 891646 | 10.427 |
| medium | no_ids | lightgbm | 0.985 | 0.989 | 0.987 | 0.961 | 0.973 | 902380 | 36.475 |
| medium | no_ids | logreg | 0.967 | 0.975 | 0.971 | 0.948 | 0.958 | 902380 | 2.104 |
| medium | no_ids | splink | 0.945 | 0.957 | 0.951 | 0.930 | 0.937 | 902380 | 9.765 |
| medium | no_street | lightgbm | 0.978 | 0.962 | 0.970 | 0.935 | 0.956 | 902380 | 35.687 |
| medium | no_street | logreg | 0.948 | 0.910 | 0.928 | 0.884 | 0.915 | 902380 | 2.121 |
| medium | with_ids | lightgbm | 0.989 | 0.990 | 0.990 | 0.963 | 0.976 | 902380 | 38.994 |
| medium | with_ids | logreg | 0.977 | 0.981 | 0.979 | 0.954 | 0.965 | 902380 | 2.369 |
| medium | with_ids | splink | 0.946 | 0.963 | 0.955 | 0.936 | 0.941 | 902380 | 11.031 |

LightGBM top features (gain share):

- easy / no_ids: street_token_set:0.65, street_jw:0.24, name_partial:0.03, name_jw:0.02, name_token_sort:0.02
- easy / no_street: name_token_set:0.57, name_jw:0.29, name_partial:0.05, name_lev:0.02, pin_eq:0.02
- easy / with_ids: street_token_set:0.67, street_jw:0.22, name_partial:0.03, name_jw:0.02, gstin_eq:0.01
- hard / no_ids: street_token_set:0.58, street_jw:0.21, name_jw:0.08, name_partial:0.04, name_lev:0.03
- hard / no_street: name_jw:0.66, name_token_set:0.08, name_token_sort:0.06, name_partial:0.04, name_lev:0.04
- hard / with_ids: street_token_set:0.58, street_jw:0.19, name_jw:0.07, name_partial:0.04, name_token_set:0.03
- medium / no_ids: street_token_set:0.62, street_jw:0.23, name_jw:0.05, name_partial:0.03, name_lev:0.02
- medium / no_street: name_jw:0.69, name_token_set:0.10, name_partial:0.05, name_lev:0.03, name_token_sort:0.03
- medium / with_ids: street_token_set:0.63, street_jw:0.21, name_jw:0.04, name_partial:0.03, name_token_set:0.02

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
