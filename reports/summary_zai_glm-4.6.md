# LLM Forecast Calibration - zai:glm-4.6

- Merged forecasts: 1028 | train: 770 | holdout: 258
- Holdout window: 2026-10-13 to 2026-11-22
- Base rate (PYES in holdout): 0.326

## Holdout results

| method                      |   n |   brier |   log_loss |    ece |   auroc |
|:----------------------------|----:|--------:|-----------:|-------:|--------:|
| identity (raw LLM)          | 258 |  0.2657 |     0.8791 | 0.196  |  0.5386 |
| temperature (T=2.7533)      | 258 |  0.2395 |     0.6798 | 0.1389 |  0.5386 |
| platt scaling               | 258 |  0.2266 |     0.6514 | 0.0819 |  0.5386 |
| isotonic regression         | 258 |  0.2259 |     0.7487 | 0.0478 |  0.5687 |
| baseline: market T-24h      | 258 |  0.1208 |     0.3566 | 0.0555 |  0.8999 |
| baseline: market last price | 258 |  0.01   |     0.0445 | 0.0171 |  0.9992 |
| baseline: constant 0.5      | 258 |  0.25   |     0.6931 | 0.1744 |  0.5    |

## Findings

- Raw LLM Brier 0.2657 (ECE 0.1960)
- Best post-hoc method: **isotonic regression**, Brier 0.2259 (delta +0.0398)
- Market close baseline Brier 0.0100 (market still ahead)
- Agreement: corr(LLM, market) = 0.265

See reliability.png and brier_by_category.png.