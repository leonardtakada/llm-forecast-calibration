# LLM Forecast Calibration - zai:glm-4.6

- Merged forecasts: 723 | train: 542 | holdout: 181
- Holdout window: 2026-11-17 to 2026-11-17
- Base rate (PYES in holdout): 0.392

## Holdout results

| method                      |   n |   brier |   log_loss |    ece |   auroc |
|:----------------------------|----:|--------:|-----------:|-------:|--------:|
| identity (raw LLM)          | 181 |  0.2449 |     0.712  | 0.187  |  0.6878 |
| temperature (T=3.1738)      | 181 |  0.2377 |     0.6689 | 0.1474 |  0.6878 |
| platt scaling               | 181 |  0.2324 |     0.6582 | 0.107  |  0.6878 |
| isotonic regression         | 181 |  0.2361 |     0.6623 | 0.15   |  0.6713 |
| baseline: market T-24h      |  67 |  0.2121 |     0.6131 | 0.2466 |  0.8212 |
| baseline: market last price | 181 |  0.0006 |     0.0154 | 0.0151 |  1      |
| baseline: constant 0.5      | 181 |  0.25   |     0.6931 | 0.1077 |  0.5    |

## Findings

- Raw LLM Brier 0.2449 (ECE 0.1870)
- Best post-hoc method: **platt scaling**, Brier 0.2324 (delta +0.0125)
- Market close baseline Brier 0.0006 (market still ahead)
- Agreement: corr(LLM, market) = 0.912

See reliability.png and brier_by_category.png.