"""RCMN: Reliability-Calibrated Multi-scale Normality for anomaly detection.

A training-free pipeline with three pluggable strategy stages:
  - memory:     how normal feature vectors are subsampled into a memory bank
  - fusion:     how multi-scale / multi-layer anomaly maps are combined
  - calibration: how continuous anomaly maps are binarized into masks

Each stage is selected by a string key in the config, enabling clean ablation.
"""