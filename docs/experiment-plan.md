# Experiment plan

1. Fix and record a random seed and simulation parameters.
2. Run noise-only trials to estimate empirical Pfa; use enough independent trials for a meaningful estimate.
3. Run target-present trials across an SNR sweep and estimate Pd.
4. Repeat for CA-CFAR and each learned detector using identical test examples.
5. Calibrate learned-detector thresholds on validation data to match a chosen Pfa.
6. Report median and percentile latency, hardware/software versions, and confidence intervals.
7. Keep train/validation/test seeds disjoint to avoid leakage.

The included 1-D demo is a starting point. Extend it with complex I/Q generation, matched filtering, Doppler processing, and CNN training before making broader claims.
