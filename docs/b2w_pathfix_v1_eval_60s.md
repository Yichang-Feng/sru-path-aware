# Phase 0 基线重评

- checkpoint: `model_3999.pt`
- horizon: 60 s,  episodes: 200,  actor: 1

| bucket | n | SR | SR(old rule) | timeout share | SPL | median d_path | p90 d_path | median start dist | median planned len | median travelled | in-goal steps | in-goal speed p50 | longest dwell run p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 200 | 0.105 | 0.540 | 0.710 | 0.043 | 0.36 | 0.66 | 15.5 | 20.7 | 27.8 | 3 | 1.10 | 3 / 7 |
| maze | 60 | 0.033 | 0.517 | 0.717 | 0.012 | 0.37 | 0.65 | 16.9 | 26.6 | 33.1 | 2 | 1.11 | 2 / 7 |
| non_maze | 10 | 0.000 | 0.600 | 0.800 | 0.000 | 0.40 | 0.69 | 11.3 | 13.7 | 45.0 | 114 | 1.06 | 6 / 7 |
| stairs | 49 | 0.122 | 0.306 | 0.735 | 0.059 | 0.38 | 0.60 | 14.8 | 21.7 | 19.3 | 0 | 1.09 | 0 / 7 |
| pits | 81 | 0.160 | 0.691 | 0.679 | 0.062 | 0.34 | 0.69 | 13.8 | 18.1 | 45.0 | 38 | 1.10 | 6 / 8 |
