# Phase 0 基线重评

- checkpoint: `model_3999.pt`
- horizon: 60 s,  episodes: 200,  actor: 1

| bucket | n | SR | SR(old rule) | timeout share | SPL | median d_path | p90 d_path | median start dist | median planned len | median travelled | in-goal steps | in-goal speed p50 | longest dwell run p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 200 | 0.010 | 0.550 | 0.870 | 0.005 | 0.38 | 0.80 | 16.1 | 19.6 | 33.2 | 12 | 1.26 | 4 / 5 |
| maze | 32 | 0.031 | 0.812 | 0.812 | 0.015 | 0.38 | 0.88 | 15.7 | 18.3 | 39.8 | 32 | 1.17 | 5 / 6 |
| non_maze | 18 | 0.000 | 0.833 | 1.000 | 0.000 | 0.46 | 1.12 | 13.5 | 15.9 | 43.7 | 43 | 1.43 | 4 / 5 |
| stairs | 65 | 0.000 | 0.169 | 0.954 | 0.000 | 0.33 | 0.40 | 17.3 | 24.2 | 13.6 | 0 | 1.15 | 0 / 4 |
| pits | 85 | 0.012 | 0.682 | 0.800 | 0.007 | 0.39 | 0.98 | 15.6 | 19.2 | 38.2 | 37 | 1.27 | 4 / 6 |
