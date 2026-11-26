# Milestone 2 — blocking

Encoder `intfloat/multilingual-e5-small` (text: name_clean, city_norm); rules {'pin_first': ['pin_norm', 'first_token'], 'phonetic': ['name_phonetic'], 'first_city': ['first_token', 'city_norm'], 'key_exact': ['name_key']} with max block size 200; kNN k=10. Evaluated on the **test** split (entity-disjoint).

![blocking trade-off](M2_blocking.png)

| level | method | param | pairs | pairs_per_record | pair_completeness | reduction_ratio | pair_quality |
|---|---|---|---|---|---|---|---|
| easy | rule:pin_first | 200 | 70252 | 0.6986 | 0.5549 | 0.999986 | 0.7668 |
| easy | rule:phonetic | 200 | 106566 | 1.0597 | 0.8885 | 0.999979 | 0.8094 |
| easy | rule:first_city | 200 | 260288 | 2.5884 | 0.8787 | 0.999949 | 0.3277 |
| easy | rule:key_exact | 200 | 78140 | 0.7771 | 0.804 | 0.999985 | 0.9989 |
| easy | rules | 200 | 290581 | 2.8897 | 0.9443 | 0.999943 | 0.3155 |
| easy | knn | 10 | 757650 | 7.5345 | 0.9933 | 0.999850 | 0.1273 |
| easy | union | rules+knn10 | 908646 | 9.036 | 0.9959 | 0.999820 | 0.1064 |
| medium | rule:pin_first | 200 | 43740 | 0.435 | 0.3304 | 0.999991 | 0.7333 |
| medium | rule:phonetic | 200 | 88010 | 0.8752 | 0.73 | 0.999983 | 0.8052 |
| medium | rule:first_city | 200 | 226622 | 2.2536 | 0.7366 | 0.999955 | 0.3155 |
| medium | rule:key_exact | 200 | 57428 | 0.5711 | 0.5907 | 0.999989 | 0.9985 |
| medium | rules | 200 | 256590 | 2.5517 | 0.8409 | 0.999949 | 0.3182 |
| medium | knn | 10 | 764968 | 7.6072 | 0.9555 | 0.999849 | 0.1213 |
| medium | union | rules+knn10 | 902216 | 8.9721 | 0.9707 | 0.999822 | 0.1045 |
| hard | rule:pin_first | 200 | 23456 | 0.2333 | 0.1634 | 0.999995 | 0.6762 |
| hard | rule:phonetic | 200 | 66533 | 0.6616 | 0.5417 | 0.999987 | 0.7904 |
| hard | rule:first_city | 200 | 184438 | 1.8341 | 0.5632 | 0.999964 | 0.2965 |
| hard | rule:key_exact | 200 | 40162 | 0.3994 | 0.4132 | 0.999992 | 0.9988 |
| hard | rules | 200 | 210543 | 2.0937 | 0.6674 | 0.999958 | 0.3077 |
| hard | knn | 10 | 773660 | 7.6937 | 0.8567 | 0.999847 | 0.1075 |
| hard | union | rules+knn10 | 891398 | 8.8645 | 0.8866 | 0.999824 | 0.0966 |
