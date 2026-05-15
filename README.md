# ML.trade — Predictive Modelling of the Stock Market

A hierarchical ensemble ML pipeline for short-horizon directional forecasting of US equities.

## Overview

This project implements a full machine learning pipeline that processes 1-hour OHLCV data for liquid US equities and generates directional forecasts. The system combines multiple model architectures with regime detection to adapt to changing market conditions.

## Architecture

1. **Correlation Clustering** — Stock universe grouped using Ward hierarchical clustering on pairwise return correlations
2. **Per-Group LSTM** — Sequence models trained on each cluster (TensorFlow/Keras)
3. **Tree Ensemble** — Stacked Random Forest + XGBoost classifiers
4. **HMM Regime Detector** — Hidden Markov Model that dynamically switches between BULL/BEAR/CRISIS regimes, adjusting entry thresholds, position sizing, leverage, and stop-losses
5. **Meta-Learning (SAML)** — MAML/Reptile-style initialization for regime-adaptive fine-tuning

## Performance

- **Live data:** Sharpe 1.70
- **Cross-seed validation:** Average Sharpe 2.02 across 9 seed/regime combinations (best run 4.20)
- **Feature engineering:** Linear predictive signal +0.24pp → +1.02pp, tree-based signal +0.50pp → +3.47pp on 1.1M-sample benchmark

## Tech Stack

- Python, TensorFlow/Keras, XGBoost, scikit-learn
- HMM, TA-Lib, NVIDIA CUDA

## Requirements

See requirements_cloud.txt for the full dependency list.
