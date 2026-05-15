# An Ensemble-Based Deep Learning Framework for Predictive Modelling of The Stock Market Using Artificial Intelligence

A three-stage hierarchical pipeline that clusters high-beta US equities, trains LSTM / XGBoost / RandomForest ensembles per group with walk-forward validation and Deflated Sharpe Ratio correction, then feeds surviving group signals into a regime-adaptive long/short portfolio backtested under BULL / BEAR / CRISIS conditions detected by a Gaussian HMM.

## Pipeline Architecture

The system runs as a three-stage pipeline:

1. **Correlation Clustering** — Ward hierarchical clustering on pairwise log-return correlations selects high-beta liquid stocks and groups them into clusters of 5.
2. **Group Tournament** — Each group is trained independently (or jointly via multi-task learning). Walk-forward validation across multiple shifted windows; groups must show positive Deflated Sharpe Ratio in a majority of windows to survive.
3. **Combined Backtest** — Surviving groups feed direction signals into a portfolio engine. A fitted Gaussian HMM provides per-timestep regime-adaptive thresholds (BULL / BEAR / CRISIS).

### Key Design Decisions

- **Triple-Barrier Method** (López de Prado) with ATR-clipped barriers for label generation
- **Deflated Sharpe Ratio** accounting for correlated trials across groups sharing the same market features
- **Continuous rolling Z-score** through val/test — no frozen train-end statistics
- **Regime HMM fitted on training window only** — prevents future regime leakage
- **Regime-adaptive position sizing, leverage, and stop-losses** that switch dynamically based on HMM state

## Tech Stack

- **Python** — core implementation
- **TensorFlow/Keras** — LSTM and Multi-Task Learning models
- **XGBoost, scikit-learn** — tree-based ensemble models
- **hmmlearn** — Gaussian HMM for regime detection
- **TA-Lib** — technical indicator computation
- **NVIDIA CUDA** — GPU-accelerated training
- **MAML/Reptile-style meta-learning** (SAML) for regime-adaptive initialization

## Performance

- **Live data:** Sharpe 1.70
- **Cross-seed validation:** Average Sharpe 2.02 across 9 seed/regime combinations (best run 4.20)
- **Feature engineering:** Linear predictive signal +0.24pp → +1.02pp, tree-based signal +0.50pp → +3.47pp on 1.1M-sample benchmark

## Installation

### Prerequisites

- Python 3.9+
- **TA-Lib C library** — must be installed before the Python wrapper:
  - **Windows**: download the wheel from [https://github.com/cgohlke/talib-build/releases](https://github.com/cgohlke/talib-build/releases)
  - **macOS**: `brew install ta-lib`
  - **Linux**: see [https://ta-lib.github.io/ta-lib-python/](https://ta-lib.github.io/ta-lib-python/)

### Install

```bash
pip install -r requirements_cloud.txt
```

## Usage

All modes are invoked through `src/pipeline.py`:

| Mode | Command | Description |
|------|---------|-------------|
| Default | `python src/pipeline.py` | Standard train/val/test anchored split |
| Full proportional | `python src/pipeline.py --full` | Proportional percentage split |
| Bear stress-test | `python src/pipeline.py --bear` | Bear market validation window |
| COVID stress-test | `python src/pipeline.py --covid` | COVID crash validation window |
| Live (yfinance) | `python src/pipeline.py --live` | Downloads fresh data |
| Multi-task learning | `python src/pipeline.py --mtl` | Shared LSTM trunk + private group heads |
| Compare LSTM vs MTL | `python src/pipeline.py --compare` | Side-by-side evaluation |
| Sentiment blending | `python src/pipeline.py --sentiment_alpha` | FinBERT confidence-weighted signal blending |

## Project Structure

```
src/
├── pipeline.py                  Main orchestrator (CLI, 3-stage pipeline, DSR, walk-forward)
├── data_loader.py               OHLCV data loading
├── yfinance_loader.py           Yahoo Finance downloader
├── feature_engineer.py          Feature engineering pipeline (TA-Lib + custom + regime)
├── data_preprocessor.py         Train/val/test splits, rolling Z-score, sequence generation
├── lstm_model.py                LSTM model, triple-barrier labeling, group training
├── mtl_lstm_model.py            Multi-task LSTM (shared trunk + private group heads)
├── ensemble_model.py            3-model voting ensemble (LSTM + XGBoost + RandomForest)
├── regime_detector.py           3-state Gaussian HMM (BULL / BEAR / CRISIS)
├── regime_specialist.py         Regime-specialist model training
├── saml_meta_learner.py         State-Aware Meta-Learning for regime-adaptive initialization
├── sentiment_engine.py          FinBERT scoring with confidence gate
├── news_loader.py               News dataset loading
├── PredictionEngine.py          Portfolio backtest engine with regime-adaptive config
├── model_evaluator.py           Classification/regression metrics and plotting
├── baseline_test.py             Sanity-check: LR + RF to verify feature predictive signal
├── feature_importance_experiment.py  Feature importance ranking
├── sweep_stage3.py              Stage-3 hyperparameter sweep
├── gpu_utils.py                 GPU acceleration utilities
├── onnx_directml.py             ONNX export for accelerated inference
├── quantile_barrier.py          Adaptive quantile-based barrier labeling
├── app.py                       Web application
├── demo_bridge.py / demo_runner.py  Demo pipeline
├── finnhub_loader.py            Finnhub API news loader
├── llm_guard.py                 LLM safety guard
├── sweep_selector.py            Sweep configuration
├── __init__.py
```

## Data

The pipeline expects hourly OHLCV data files. For live mode, yfinance downloads and resamples to hourly. The `data/` directory is gitignored — place data files locally before running.
