# LLM Forecast Calibration - zai:glm-5.3

- Merged forecasts: 1028 | train: 770 | holdout: 258
- Holdout window: 2026-10-13 to 2026-11-22
- Base rate (PYES in holdout): 0.326

## Holdout results

| method                      |   n |   brier |   log_loss |    ece |   auroc |
|:----------------------------|----:|--------:|-----------:|-------:|--------:|
| identity (raw LLM)          | 258 |  0.2644 |     0.8713 | 0.1983 |  0.5507 |
| temperature (T=3.1805)      | 258 |  0.2376 |     0.6714 | 0.1462 |  0.5507 |
| platt scaling               | 258 |  0.224  |     0.6428 | 0.0668 |  0.5507 |
| isotonic regression         | 258 |  0.2253 |     0.6447 | 0.0515 |  0.5759 |
| baseline: market T-24h      | 258 |  0.1208 |     0.3566 | 0.0555 |  0.8999 |
| baseline: market last price | 258 |  0.01   |     0.0445 | 0.0171 |  0.9992 |
| baseline: constant 0.5      | 258 |  0.25   |     0.6931 | 0.1744 |  0.5    |

## Findings

- Raw LLM Brier 0.2644 (ECE 0.1983)
- Best post-hoc method: **platt scaling**, Brier 0.2240 (delta +0.0404)
- Market close baseline Brier 0.0100 (market still ahead)
- Agreement: corr(LLM, market) = 0.272

See reliability.png and brier_by_category.png.