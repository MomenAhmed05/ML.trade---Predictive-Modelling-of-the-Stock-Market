import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Optional
import pickle
import json
from pathlib import Path
import warnings

from regime_specialist import RegimeSpecialistLSTM, route_to_specialist

# ────────────────────────────────────────────────────────────────────────────────
# Specialist blend weight constants (Fix 4 — configurable, not hardcoded)
# Base ensemble weight + specialist weight must sum to 1.0
# ────────────────────────────────────────────────────────────────────────────────
SPECIALIST_BLEND_BASE: float = 0.60  # Weight given to base ensemble predictions
SPECIALIST_BLEND_SPECIALIST: float = 0.40  # Weight given to regime specialist predictions

class EnsembleModel:
    """
    Consensus Voting System: Combines LSTM temporal learning with tree-based state thresholds.
    """
    def __init__(self, lstm_model=None, save_dir: str = "models"):
        self.lstm_model = lstm_model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )

    def _extract_2d(self, X_3d: np.ndarray) -> np.ndarray:
        return X_3d[:, -1, :]

    def train_trees(self, X_train_3d: np.ndarray, y_train_cls: np.ndarray):
        print("\n" + "="*60)
        print("TRAINING ENSEMBLE TREE MODELS (XGBoost & Random Forest)")
        print("="*60)
        X_train_2d = self._extract_2d(X_train_3d)
        if len(y_train_cls.shape) > 1 and y_train_cls.shape[1] > 1:
            y_train_1d = np.argmax(y_train_cls, axis=1)
        else:
            y_train_1d = y_train_cls
        print(f"Tree models training on shapes: X={X_train_2d.shape}, y={y_train_1d.shape}")
        print("Training Random Forest...")
        self.rf_model.fit(X_train_2d, y_train_1d)
        print("Training XGBoost...")
        self.xgb_model.fit(X_train_2d, y_train_1d)
        print("Tree models training complete.")

    def save_trees(self):
        rf_path  = self.save_dir / "rf_model.pkl"
        xgb_path = self.save_dir / "xgb_model.pkl"
        with open(rf_path,  "wb") as f: pickle.dump(self.rf_model,  f)
        with open(xgb_path, "wb") as f: pickle.dump(self.xgb_model, f)
        print(f"Saved Random Forest to {rf_path}")
        print(f"Saved XGBoost to {xgb_path}")

    def load_trees(self):
        rf_path  = self.save_dir / "rf_model.pkl"
        xgb_path = self.save_dir / "xgb_model.pkl"
        if rf_path.exists() and xgb_path.exists():
            with open(rf_path,  "rb") as f: self.rf_model  = pickle.load(f)
            with open(xgb_path, "rb") as f: self.xgb_model = pickle.load(f)
            print(f"Loaded Random Forest and XGBoost models.")
        else:
            raise FileNotFoundError("Saved tree models not found.")

    def predict(self, X_test_3d: np.ndarray, agreement_threshold: int = 2) -> Dict[str, np.ndarray]:
        if self.lstm_model is None or self.lstm_model.model is None:
            raise ValueError("LSTM model is not loaded/built.")
        lstm_preds      = self.lstm_model.predict(X_test_3d)
        lstm_probs      = lstm_preds["direction_probabilities"][:, 1]
        lstm_class      = np.argmax(lstm_preds["direction_probabilities"], axis=1)
        X_test_2d       = self._extract_2d(X_test_3d)
        rf_probs        = self.rf_model.predict_proba(X_test_2d)[:, 1]
        rf_class        = self.rf_model.predict(X_test_2d)
        xgb_probs       = self.xgb_model.predict_proba(X_test_2d)[:, 1]
        xgb_class       = self.xgb_model.predict(X_test_2d)
        total_votes_up  = lstm_class + rf_class + xgb_class
        ensemble_class  = (total_votes_up >= agreement_threshold).astype(int)
        ensemble_prob_up = (lstm_probs + rf_probs + xgb_probs) / 3.0
        return {
            "ensemble_class":     ensemble_class,
            "ensemble_direction": np.where(ensemble_class == 0, "DOWN", "UP"),
            "ensemble_prob_up":   ensemble_prob_up,
            "votes_up":           total_votes_up,
            "lstm_class":         lstm_class,
            "rf_class":           rf_class,
            "xgb_class":          xgb_class,
            "lstm_probs":         lstm_probs,
            "rf_probs":           rf_probs,
            "xgb_probs":          xgb_probs,
        }


# --------------------------------------------------------------------------------
# Helper: build a DataPreprocessor anchored to an optional end date.
# --------------------------------------------------------------------------------

def _make_preprocessor(
    lookback: int,
    horizon: int,
    enriched_df,
    anchor_end_date: Optional[str],
    use_full_split: bool = False,
    split_months: tuple = None,
):
    """
    Returns (preprocessor, splits, truncated_df).
    """
    from data_preprocessor import DataPreprocessor
    import pandas as pd

    df = enriched_df
    if anchor_end_date is not None:
        try:
            end_ts = pd.Timestamp(anchor_end_date)
            idx    = pd.to_datetime(df.index, errors="coerce")
            mask   = idx <= end_ts
            if mask.sum() < (lookback + horizon) * 2:
                raise ValueError(
                    f"Only {mask.sum()} rows before {anchor_end_date}; insufficient for splits."
                )
            df = df.loc[mask]
        except Exception as exc:
            raise ValueError(
                f"anchor_end_date='{anchor_end_date}' could not be applied: {exc}"
            ) from exc

    pre    = DataPreprocessor(
        lookback_window=lookback,
        forecast_horizon=horizon,
        use_walk_forward=not use_full_split,
        use_full_split=use_full_split,
        split_months=split_months,
    )
    splits = pre.preprocess(df)
    return pre, splits, df


def _resolve_model_dir(group_id: int, model_dir: Optional[str], model_dir_prefix: Optional[str]) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    if model_dir_prefix:
        return Path(f"models/{model_dir_prefix}/group_{group_id}")
    return Path(f"models/group_{group_id}")


# --------------------------------------------------------------------------------
# run_val_backtest
# --------------------------------------------------------------------------------

# EXP15: Module-level regime config override for val-backtest alignment.
# Set by pipeline.py before calling run_val_backtest() so that val thresholds
# match the actual test backtest thresholds (improves tournament selectivity).
_val_regime_config: dict = None


def run_val_backtest(
    tickers: list,
    group_id: int,
    lookback: int = 24,
    horizon: int = 8,
    data_path: str = "data/raw",
    model_dir: str = None,
    model_dir_prefix: Optional[str] = None,
    anchor_end_date: Optional[str] = None,
    sentiment_alpha: float = 0.0,
    use_full_split: bool = False,
    mtl_model=None,          # BUG 1 + BUG 2 fix: accept and use MTLGroupAdapter
    split_months: tuple = None,
) -> Dict[str, Any]:
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from lstm_model import LSTMModel, make_direction_onehot_from_raw
    from PredictionEngine import PortfolioEngine

    model_dir_path = _resolve_model_dir(group_id, model_dir, model_dir_prefix)

    group_results_dir = Path(f"results/group_{group_id}")
    group_results_dir.mkdir(parents=True, exist_ok=True)

    LONG_THRESHOLD        = 0.53
    SHORT_THRESHOLD       = 0.45
    MIN_VOL_RATIO         = 0.003
    VOL_REGIME_PERCENTILE = 95

    # --- Build the LSTM component (MTLGroupAdapter or plain LSTMModel) ---
    if mtl_model is not None:
        # BUG 2 fix: use the already-trained MTL head instead of loading from disk
        from mtl_lstm_model import MTLGroupAdapter
        lstm = MTLGroupAdapter(mtl_model, group_id)
    else:
        lstm = LSTMModel(
            model_path=str(model_dir_path / "lstm_model.weights.h5"),
            history_path=str(model_dir_path / "training_history.pkl"),
        )
        for ticker in tickers:
            try:
                loader   = DataLoader(data_path, ticker)
                enriched = FeatureEngineer().compute_indicators(loader.load_data())
                _, splits, _ = _make_preprocessor(lookback, horizon, enriched, anchor_end_date, use_full_split, split_months=split_months)
                lstm.build_model(input_shape=(lookback, splits["X_train"].shape[2]))
                lstm.load_model()
                break
            except Exception:
                continue

    ensemble = EnsembleModel(lstm_model=lstm, save_dir=str(model_dir_path))

    try:
        ensemble.load_trees()
        print(f"  Group {group_id}: loaded pre-trained trees.")
    except FileNotFoundError:
        all_X_train, all_y_train = [], []
        for ticker in tickers:
            try:
                loader   = DataLoader(data_path, ticker)
                enriched = FeatureEngineer().compute_indicators(loader.load_data())
                _, splits, enriched_trunc = _make_preprocessor(lookback, horizon, enriched, anchor_end_date, use_full_split, split_months=split_months)
                labels, masks, _ = make_direction_onehot_from_raw(
                    enriched_df=enriched_trunc,
                    split_indices={"idx_train": splits["idx_train"], "idx_val": splits["idx_val"], "idx_test": splits["idx_test"]},
                    lookback_window=lookback, forecast_horizon=horizon,
                )
                all_X_train.append(splits["X_train"][masks["train"]])
                all_y_train.append(labels["train"][masks["train"]])
            except Exception:
                continue

        if not all_X_train:
            return {"val_sharpe": 0.0, "val_return_pct": 0.0, "val_trades": 0, "val_max_dd": 0.0}

        ensemble.train_trees(np.concatenate(all_X_train), np.concatenate(all_y_train))
        ensemble.save_trees()

    sent_engine = None
    news_loader = None
    if sentiment_alpha > 0.0:
        from news_loader import NewsLoader
        from sentiment_engine import SentimentEngine
        news_loader = NewsLoader()
        news_loader.load()
        sent_engine = SentimentEngine()
        sent_engine.load_model()
        print(f"[Sentiment] Gate active for val backtest: alpha={sentiment_alpha}")

    import pandas as pd
    all_prices, all_probs, all_long_g, all_short_g, all_atr_vol = [], [], [], [], []
    ticker_val_returns = []
    for ticker in tickers:
        try:
            loader   = DataLoader(data_path, ticker)
            enriched = FeatureEngineer().compute_indicators(loader.load_data())
            _, splits, enriched_trunc = _make_preprocessor(lookback, horizon, enriched, anchor_end_date, use_full_split, split_months=split_months)
            labels, masks, _ = make_direction_onehot_from_raw(
                enriched_df=enriched_trunc,
                split_indices={"idx_train": splits["idx_train"], "idx_val": splits["idx_val"], "idx_test": splits["idx_test"]},
                lookback_window=lookback, forecast_horizon=horizon,
            )
            preds     = ensemble.predict(splits["X_val"], agreement_threshold=2)
            row_idx   = splits["idx_val"] + lookback - 1
            prices    = enriched_trunc["Close"].values[row_idx]
            atr_raw   = np.clip(enriched_trunc["ATR_14"].values[row_idx], 0,
                                np.percentile(enriched_trunc["ATR_14"].values[row_idx], 99))
            vol_ratio = np.where(prices > 0, atr_raw / prices, 0.0)

            prob_up = preds["ensemble_prob_up"].ravel()

            if sent_engine is not None and news_loader is not None:
                candle_ts = enriched_trunc.index[row_idx]
                feats     = sent_engine.get_sentiment_features(
                    ticker, pd.Series(candle_ts), news_loader
                )
                conf = feats["sentiment_confidence"].values if "sentiment_confidence" in feats.columns else None
                prob_up = sent_engine.apply_sentiment_gate(
                    prob_up, feats["sentiment_score"].values,
                    alpha=sentiment_alpha, confidence=conf,
                )

            all_prices.append(prices)
            all_probs.append(prob_up)
            all_long_g.append((vol_ratio >= MIN_VOL_RATIO))
            all_short_g.append((vol_ratio >= MIN_VOL_RATIO))
            all_atr_vol.append(vol_ratio)
            ticker_ret = float(prices[-1] / prices[0] - 1) if len(prices) > 1 and prices[0] > 0 else 0.0
            ticker_val_returns.append(ticker_ret)
        except Exception as e:
            print(f"  Skipping {ticker} in val backtest: {e}")
            continue

    if not all_prices:
        return {"val_sharpe": 0.0, "val_return_pct": 0.0, "val_trades": 0, "val_max_dd": 0.0}

    # Simple tail-based alignment (most reliable for this use case)
    lengths = [len(p) for p in all_prices]
    min_len = int(np.percentile(lengths, 10))
    n_dropped = sum(1 for l in lengths if l < min_len)
    if n_dropped:
        print(f"  {n_dropped} ticker(s) below 10th-percentile length dropped")

    # Filter out tickers below min_len before stacking to avoid dimension mismatch
    filtered_prices = [p[-min_len:] for p in all_prices if len(p) >= min_len]
    filtered_probs = [p[-min_len:] for p in all_probs if len(p) >= min_len]
    filtered_long_g = [g[-min_len:] for g in all_long_g if len(g) >= min_len]
    filtered_short_g = [g[-min_len:] for g in all_short_g if len(g) >= min_len]
    filtered_vol = [v[-min_len:] for v in all_atr_vol if len(v) >= min_len]

    if not filtered_prices:
        print("ERROR: No tickers remaining after filtering")
        return {"val_sharpe": -999.0, "val_return_pct": 0.0, "val_trades": 0, "val_max_dd": 0.0, "val_ticker_positive_ratio": 0.0}

    price_matrix = np.column_stack(filtered_prices)
    prob_matrix = np.column_stack(filtered_probs)
    long_g_matrix = np.column_stack(filtered_long_g)
    short_g_matrix = np.column_stack(filtered_short_g)
    vol_matrix = np.column_stack(filtered_vol)

    vol_p95 = np.percentile(vol_matrix, VOL_REGIME_PERCENTILE, axis=0)
    high_vol_regime = vol_matrix > vol_p95[np.newaxis, :]
    print(f"Vol-regime halt: {high_vol_regime.mean()*100:.1f}% of timesteps")

    signal_matrix = np.zeros_like(prob_matrix, dtype=int)
    signal_matrix[(prob_matrix >= LONG_THRESHOLD)  & long_g_matrix  & ~high_vol_regime] =  1
    signal_matrix[(prob_matrix <= SHORT_THRESHOLD) & short_g_matrix] = -1
    print(f"Long {np.sum(signal_matrix==1)} | Short {np.sum(signal_matrix==-1)} | Flat {np.sum(signal_matrix==0)}")

    engine = PortfolioEngine(
        initial_capital=10_000.0, base_trade_size_pct=0.20,
        transaction_fee=0.0002, long_safety_sl=0.05, short_safety_sl=0.05,
        horizon=horizon,
    )
    metrics = engine.run_portfolio_backtest(price_matrix, signal_matrix, prob_matrix)
    print(f"Return: {metrics['total_return_pct']:+.2f}% | Sharpe: {metrics['sharpe_ratio']:.2f} "
          f"| MaxDD: {metrics['max_drawdown_pct']:.2f}% | Trades: {metrics['total_trades']}")
    engine.plot_equity_curve(save_path="equity_curve_portfolio.png")

    # Return val backtest results
    val_ticker_positive = sum(1 for ret in ticker_val_returns if ret > 0) / len(ticker_val_returns) if ticker_val_returns else 1.0
    return {
        "val_sharpe": metrics['sharpe_ratio'],
        "val_return_pct": metrics['total_return_pct'],
        "val_trades": metrics['total_trades'],
        "val_max_dd": metrics['max_drawdown_pct'],
        "val_ticker_positive_ratio": val_ticker_positive,
    }
def run_ensemble(
    tickers: list,
    group_id: int,
    lookback: int = 24,
    horizon: int = 8,
    data_path: str = "data/raw",
    model_dir: str = None,
    model_dir_prefix: Optional[str] = None,
    anchor_end_date: Optional[str] = None,
    sentiment_alpha: float = 0.0,
    use_full_split: bool = False,
    mtl_model=None, # BUG 1 + BUG 2 fix: accept and use MTLGroupAdapter
    split_months: tuple = None,
) -> Dict[str, Any]:
    """
    DEPRECATED: Use run_ensemble_regime_aware() instead.
    This function delegates to the regime-aware version with default BULL regime.
    """
    warnings.warn(
        "run_ensemble() is deprecated. Use run_ensemble_regime_aware() for regime-aware predictions.",
        DeprecationWarning,
        stacklevel=2
    )

    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from lstm_model import LSTMModel, make_direction_onehot_from_raw

    model_dir_path = _resolve_model_dir(group_id, model_dir, model_dir_prefix)

    LONG_THRESHOLD  = 0.53
    SHORT_THRESHOLD = 0.45
    MIN_VOL_RATIO   = 0.003

    # --- Build the LSTM component (MTLGroupAdapter or plain LSTMModel) ---
    if mtl_model is not None:
        from mtl_lstm_model import MTLGroupAdapter
        lstm = MTLGroupAdapter(mtl_model, group_id)
    else:
        lstm = LSTMModel(
            model_path=str(model_dir_path / "lstm_model.weights.h5"),
            history_path=str(model_dir_path / "training_history.pkl"),
        )
        for ticker in tickers:
            try:
                loader   = DataLoader(data_path, ticker)
                enriched = FeatureEngineer().compute_indicators(loader.load_data())
                _, splits, _ = _make_preprocessor(lookback, horizon, enriched, anchor_end_date, use_full_split, split_months=split_months)
                lstm.build_model(input_shape=(lookback, splits["X_train"].shape[2]))
                lstm.load_model()
                break
            except Exception:
                continue

    ensemble = EnsembleModel(lstm_model=lstm, save_dir=str(model_dir_path))

    try:
        ensemble.load_trees()
        print(f"  Group {group_id}: loaded pre-trained trees.")
    except FileNotFoundError:
        all_X_train, all_y_train = [], []
        for ticker in tickers:
            try:
                loader   = DataLoader(data_path, ticker)
                enriched = FeatureEngineer().compute_indicators(loader.load_data())
                _, splits, enriched_trunc = _make_preprocessor(lookback, horizon, enriched, anchor_end_date, use_full_split, split_months=split_months)
                labels, masks, _ = make_direction_onehot_from_raw(
                    enriched_df=enriched_trunc,
                    split_indices={"idx_train": splits["idx_train"], "idx_val": splits["idx_val"], "idx_test": splits["idx_test"]},
                    lookback_window=lookback, forecast_horizon=horizon,
                )
                all_X_train.append(splits["X_train"][masks["train"]])
                all_y_train.append(labels["train"][masks["train"]])
            except Exception:
                continue

        ensemble.train_trees(np.concatenate(all_X_train), np.concatenate(all_y_train))
        ensemble.save_trees()

    sent_engine = None
    news_loader = None
    if sentiment_alpha > 0.0:
        from news_loader import NewsLoader
        from sentiment_engine import SentimentEngine
        news_loader = NewsLoader()
        news_loader.load()
        sent_engine = SentimentEngine()
        sent_engine.load_model()
        print(f"[Sentiment] Gate active for test ensemble: alpha={sentiment_alpha}")

    import pandas as pd
    all_prices, all_probs, all_votes = [], [], []
    all_long_g, all_short_g, all_atr_vol = [], [], []
    all_enriched, all_row_idx = [], []
    all_enriched_full = []  # Full data (not truncated to test window) for regime detector
    # GLM enrichment data (per-ticker, parallel to all_prices)
    all_lstm_probs, all_rf_probs, all_xgb_probs = [], [], []
    all_sentiment_scores, all_sentiment_mag, all_sentiment_vol = [], [], []
    all_headlines_text = []   # list of lists of headline strings per candle

    for ticker in tickers:
        try:
            loader   = DataLoader(data_path, ticker)
            enriched = FeatureEngineer().compute_indicators(loader.load_data())
            _, splits, enriched_trunc = _make_preprocessor(lookback, horizon, enriched, anchor_end_date, use_full_split, split_months=split_months)
            labels, masks, _ = make_direction_onehot_from_raw(
                enriched_df=enriched_trunc,
                split_indices={"idx_train": splits["idx_train"], "idx_val": splits["idx_val"], "idx_test": splits["idx_test"]},
                lookback_window=lookback, forecast_horizon=horizon,
            )

            preds_full = ensemble.predict(splits["X_test"], agreement_threshold=2)
            row_idx    = splits["idx_test"] + lookback - 1
            prices     = enriched_trunc["Close"].values[row_idx]
            atr_raw    = np.clip(enriched_trunc["ATR_14"].values[row_idx], 0,
                                 np.percentile(enriched_trunc["ATR_14"].values[row_idx], 99))
            vol_ratio  = np.where(prices > 0, atr_raw / prices, 0.0)

            prob_up = preds_full["ensemble_prob_up"].ravel()

            # Collect per-model probs for GLM enrichment
            ticker_lstm_probs = preds_full["lstm_probs"].ravel()
            ticker_rf_probs   = preds_full["rf_probs"].ravel()
            ticker_xgb_probs  = preds_full["xgb_probs"].ravel()

            # Sentiment features + raw headline text
            ticker_sent_score = np.zeros(len(prob_up))
            ticker_sent_mag   = np.zeros(len(prob_up))
            ticker_sent_vol   = np.zeros(len(prob_up))
            ticker_headlines   = [[] for _ in range(len(prob_up))]

            if sent_engine is not None and news_loader is not None:
                candle_ts = enriched_trunc.index[row_idx]
                feats     = sent_engine.get_sentiment_features(
                    ticker, pd.Series(candle_ts), news_loader
                )
                conf = feats["sentiment_confidence"].values if "sentiment_confidence" in feats.columns else None
                prob_up = sent_engine.apply_sentiment_gate(
                    prob_up, feats["sentiment_score"].values,
                    alpha=sentiment_alpha, confidence=conf,
                )
                ticker_sent_score = feats["sentiment_score"].values
                ticker_sent_mag   = feats["sentiment_magnitude"].values if "sentiment_magnitude" in feats.columns else ticker_sent_mag
                ticker_sent_vol   = feats["sentiment_volume"].values if "sentiment_volume" in feats.columns else ticker_sent_vol

                # Collect raw headline text per candle for GLM
                try:
                    headlines_df = news_loader.get_headlines(
                        ticker, pd.Series(candle_ts), lookback_hours=24
                    )
                    if not headlines_df.empty and "headline" in headlines_df.columns:
                        cts = pd.to_datetime(candle_ts, errors="coerce")
                        if cts.dt.tz is None:
                            cts = cts.tz_localize("UTC")
                        ct_index = {ct: idx for idx, ct in enumerate(cts)}
                        hdf = headlines_df.copy()
                        if hdf["candle_ts"].dt.tz is None:
                            hdf["candle_ts"] = hdf["candle_ts"].dt.tz_localize("UTC")
                        for ct, group in hdf.groupby("candle_ts"):
                            idx = ct_index.get(ct)
                            if idx is not None:
                                ticker_headlines[idx] = group["headline"].tolist()[:5]  # cap at 5
                except Exception:
                    pass  # headlines are optional enrichment; fail silently

            long_gate  = (vol_ratio >= MIN_VOL_RATIO)
            short_gate = (vol_ratio >= MIN_VOL_RATIO)

            long_fires  = int((long_gate  & (prob_up >= LONG_THRESHOLD)).sum())
            short_fires = int((short_gate & (prob_up <= SHORT_THRESHOLD)).sum())
            print(f"  {ticker}: long signals {long_fires} | short signals {short_fires}")

            all_prices.append(prices)
            all_probs.append(prob_up)
            all_votes.append(preds_full["votes_up"].ravel())
            all_long_g.append(long_gate)
            all_short_g.append(short_gate)
            all_atr_vol.append(vol_ratio)
            all_enriched.append(enriched_trunc)
            all_row_idx.append(row_idx)
            all_enriched_full.append(enriched)  # Full data before truncation
            all_lstm_probs.append(ticker_lstm_probs)
            all_rf_probs.append(ticker_rf_probs)
            all_xgb_probs.append(ticker_xgb_probs)
            all_sentiment_scores.append(ticker_sent_score)
            all_sentiment_mag.append(ticker_sent_mag)
            all_sentiment_vol.append(ticker_sent_vol)
            all_headlines_text.append(ticker_headlines)
        except Exception as e:
            print(f"  Skipping {ticker} in ensemble prediction: {e}")
            continue

    return {
        "prices": all_prices,
        "probs": all_probs,
        "votes": all_votes,
        "long_g": all_long_g,
        "short_g": all_short_g,
        "atr_vol": all_atr_vol,
        "enriched": all_enriched,
        "enriched_full": all_enriched_full,
        "row_idx": all_row_idx,
        # GLM enrichment
        "lstm_probs": all_lstm_probs,
        "rf_probs": all_rf_probs,
        "xgb_probs": all_xgb_probs,
        "sentiment_scores": all_sentiment_scores,
        "sentiment_mag": all_sentiment_mag,
        "sentiment_vol": all_sentiment_vol,
        "headlines_text": all_headlines_text,
    }


# Global cache for regime specialists to avoid reloading during backtest
_specialist_cache: Dict[str, RegimeSpecialistLSTM] = {}


def _get_specialist(specialist_name: str, specialists_dir: Path = Path("models/regime_specialists")) -> RegimeSpecialistLSTM:
	"""Load and cache a specialist model."""
	global _specialist_cache
	if specialist_name not in _specialist_cache:
		specialists_path = specialists_dir / specialist_name.lower()
		# Find the actual model weights file (may be nested in group subdirs)
		model_files = list(specialists_path.rglob("*.weights.h5")) if specialists_path.exists() else []
		if not model_files:
			raise FileNotFoundError(f"Specialist model not found: {specialists_path}")
		# Load the first found model weights
		model_file = model_files[0]
		specialist = RegimeSpecialistLSTM(specialist_name=specialist_name)
		specialist.build_model()
		specialist.model.load_weights(str(model_file))
		_specialist_cache[specialist_name] = specialist
		print(f"Loaded and cached specialist: {specialist_name}")
	return _specialist_cache[specialist_name]


def run_ensemble_regime_aware(
    tickers: list,
    group_id: int,
    regime_series: np.ndarray,
    regime_confidence: Dict[int, Dict[str, float]],
    lookback: int = 24,
    horizon: int = 8,
    data_path: str = "data/raw",
    model_dir: str = None,
    model_dir_prefix: Optional[str] = None,
    anchor_end_date: Optional[str] = None,
    sentiment_alpha: float = 0.0,
    use_full_split: bool = False,
    mtl_model=None,
    split_months: tuple = None,
    specialists_dir: str = "models/regime_specialists",
) -> Dict[str, Any]:
    """
    Regime-aware ensemble prediction with hard model routing.

    Args:
        tickers: List of ticker symbols
        group_id: Group identifier for model loading
        regime_series: Array of regime labels per timestep (0=bull, 1=bear, 2=crisis)
        regime_confidence: Dict mapping timestep -> {regime: confidence_score}
        lookback: Lookback window for LSTM
        horizon: Forecast horizon
        data_path: Path to raw data
        model_dir: Custom model directory (overrides model_dir_prefix)
        model_dir_prefix: Prefix for model directory path
        anchor_end_date: Optional end date to anchor the backtest
        sentiment_alpha: Sentiment gating alpha (0.0 = disabled)
        use_full_split: Whether to use full train/val/test splits
        mtl_model: Optional MTL model adapter
        split_months: Tuple of (train_months, val_months, test_months)
        specialists_dir: Directory containing regime specialist models

    Returns:
        Dict with prediction results including regime-aware routing info
    """
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from lstm_model import LSTMModel, make_direction_onehot_from_raw

    model_dir_path = _resolve_model_dir(group_id, model_dir, model_dir_prefix)
    specialists_path = Path(specialists_dir)

    # EXP15: Use regime-specific thresholds if pipeline set the override.
    if _val_regime_config:
        LONG_THRESHOLD = _val_regime_config.get('long_threshold', 0.53)
        SHORT_THRESHOLD = _val_regime_config.get('short_threshold', 0.45)
        print(f"  [EXP15] Val thresholds aligned: LONG={LONG_THRESHOLD}, SHORT={SHORT_THRESHOLD}")
    else:
        LONG_THRESHOLD = 0.53
        SHORT_THRESHOLD = 0.45
    MIN_VOL_RATIO = 0.003

    # --- Build the LSTM component (MTLGroupAdapter or plain LSTMModel) ---
    if mtl_model is not None:
        from mtl_lstm_model import MTLGroupAdapter
        lstm = MTLGroupAdapter(mtl_model, group_id)
    else:
        lstm = LSTMModel(
            model_path=str(model_dir_path / "lstm_model.weights.h5"),
            history_path=str(model_dir_path / "training_history.pkl"),
        )
    for ticker in tickers:
        try:
            loader = DataLoader(data_path, ticker)
            enriched = FeatureEngineer().compute_indicators(loader.load_data())
            _, splits, _ = _make_preprocessor(lookback, horizon, enriched, anchor_end_date, use_full_split, split_months=split_months)
            lstm.build_model(input_shape=(lookback, splits["X_train"].shape[2]))
            lstm.load_model()
            break
        except Exception:
            continue

    ensemble = EnsembleModel(lstm_model=lstm, save_dir=str(model_dir_path))

    # Try to load base tree models
    try:
        ensemble.load_trees()
        print(f" Group {group_id}: loaded pre-trained trees.")
    except FileNotFoundError:
        all_X_train, all_y_train = [], []
        for ticker in tickers:
            try:
                loader = DataLoader(data_path, ticker)
                enriched = FeatureEngineer().compute_indicators(loader.load_data())
                _, splits, enriched_trunc = _make_preprocessor(lookback, horizon, enriched, anchor_end_date, use_full_split, split_months=split_months)
                labels, masks, _ = make_direction_onehot_from_raw(
                    enriched_df=enriched_trunc,
                    split_indices={"idx_train": splits["idx_train"], "idx_val": splits["idx_val"], "idx_test": splits["idx_test"]},
                    lookback_window=lookback, forecast_horizon=horizon,
                )
                all_X_train.append(splits["X_train"][masks["train"]])
                all_y_train.append(labels["train"][masks["train"]])
            except Exception:
                continue

        if not all_X_train:
            return {
                "prices": [], "probs": [], "votes": [],
                "long_g": [], "short_g": [], "atr_vol": [],
                "enriched": [], "enriched_full": [], "row_idx": [],
                "lstm_probs": [], "rf_probs": [], "xgb_probs": [],
                "sentiment_scores": [], "sentiment_mag": [], "sentiment_vol": [],
                "headlines_text": [], "regime_routing": [],
            }

        ensemble.train_trees(np.concatenate(all_X_train), np.concatenate(all_y_train))
        ensemble.save_trees()

    # Initialize sentiment engine if needed
    sent_engine = None
    news_loader = None
    if sentiment_alpha > 0.0:
        from news_loader import NewsLoader
        from sentiment_engine import SentimentEngine
        news_loader = NewsLoader()
        news_loader.load()
        sent_engine = SentimentEngine()
        sent_engine.load_model()
        print(f"[Sentiment] Gate active for regime-aware ensemble: alpha={sentiment_alpha}")

    import pandas as pd
    all_prices, all_probs, all_votes = [], [], []
    all_long_g, all_short_g, all_atr_vol = [], [], []
    all_enriched, all_row_idx = [], []
    all_enriched_full = []
    all_lstm_probs, all_rf_probs, all_xgb_probs = [], [], []
    all_sentiment_scores, all_sentiment_mag, all_sentiment_vol = [], [], []
    all_headlines_text = []
    all_regime_routing = []  # Track which specialist was used per timestep

    for ticker in tickers:
        try:
            loader = DataLoader(data_path, ticker)
            enriched = FeatureEngineer().compute_indicators(loader.load_data())
            _, splits, enriched_trunc = _make_preprocessor(lookback, horizon, enriched, anchor_end_date, use_full_split, split_months=split_months)
            labels, masks, _ = make_direction_onehot_from_raw(
                enriched_df=enriched_trunc,
                split_indices={"idx_train": splits["idx_train"], "idx_val": splits["idx_val"], "idx_test": splits["idx_test"]},
                lookback_window=lookback, forecast_horizon=horizon,
            )

            # Get base ensemble predictions
            preds_full = ensemble.predict(splits["X_test"], agreement_threshold=2)
            row_idx = splits["idx_test"] + lookback - 1
            prices = enriched_trunc["Close"].values[row_idx]
            atr_raw = np.clip(enriched_trunc["ATR_14"].values[row_idx], 0,
                            np.percentile(enriched_trunc["ATR_14"].values[row_idx], 99))
            vol_ratio = np.where(prices > 0, atr_raw / prices, 0.0)

            prob_up = preds_full["ensemble_prob_up"].ravel()

            # Apply regime-aware routing
            n_timesteps = len(prob_up)
            regime_routing = []

            for t in range(n_timesteps):
                # Get regime for this timestep
                regime = regime_series[t + lookback - 1] if t + lookback - 1 < len(regime_series) else "BULL"
                
                # Get regime confidence for this timestep
                timestep_conf = regime_confidence.get(t + lookback - 1, {
                    "BULL": 1.0, "BEAR": 0.0, "CRISIS": 0.0
                })

                # Route to appropriate specialist
                specialist_name = route_to_specialist(regime, timestep_conf)
                regime_routing.append(specialist_name)

                # Load specialist and get adjusted prediction
                try:
                    specialist = _get_specialist(specialist_name, specialists_path)
                    # Get specialist prediction based on current features
                    X_2d = ensemble._extract_2d(splits["X_test"][t:t+1])
                    specialist_pred = specialist.predict(X_2d)[0]
                    # Blend base ensemble prob with specialist prediction
                    prob_up[t] = SPECIALIST_BLEND_BASE * prob_up[t] + SPECIALIST_BLEND_SPECIALIST * specialist_pred
                except (FileNotFoundError, Exception) as e:
                    # Fallback to base ensemble if specialist unavailable
                    pass

            # Collect per-model probs
            ticker_lstm_probs = preds_full["lstm_probs"].ravel()
            ticker_rf_probs = preds_full["rf_probs"].ravel()
            ticker_xgb_probs = preds_full["xgb_probs"].ravel()

            # Sentiment features
            ticker_sent_score = np.zeros(len(prob_up))
            ticker_sent_mag = np.zeros(len(prob_up))
            ticker_sent_vol = np.zeros(len(prob_up))
            ticker_headlines = [[] for _ in range(len(prob_up))]

            if sent_engine is not None and news_loader is not None:
                candle_ts = enriched_trunc.index[row_idx]
                feats = sent_engine.get_sentiment_features(
                    ticker, pd.Series(candle_ts), news_loader
                )
                conf = feats["sentiment_confidence"].values if "sentiment_confidence" in feats.columns else None
                prob_up = sent_engine.apply_sentiment_gate(
                    prob_up, feats["sentiment_score"].values,
                    alpha=sentiment_alpha, confidence=conf,
                )
                ticker_sent_score = feats["sentiment_score"].values
                ticker_sent_mag = feats["sentiment_magnitude"].values if "sentiment_magnitude" in feats.columns else ticker_sent_mag
                ticker_sent_vol = feats["sentiment_volume"].values if "sentiment_volume" in feats.columns else ticker_sent_vol

                # Collect headlines
                try:
                    headlines_df = news_loader.get_headlines(
                        ticker, pd.Series(candle_ts), lookback_hours=24
                    )
                    if not headlines_df.empty and "headline" in headlines_df.columns:
                        cts = pd.to_datetime(candle_ts, errors="coerce")
                        if cts.dt.tz is None:
                            cts = cts.tz_localize("UTC")
                        ct_index = {ct: idx for idx, ct in enumerate(cts)}
                        hdf = headlines_df.copy()
                        if hdf["candle_ts"].dt.tz is None:
                            hdf["candle_ts"] = hdf["candle_ts"].dt.tz_localize("UTC")
                        for ct, group in hdf.groupby("candle_ts"):
                            idx = ct_index.get(ct)
                            if idx is not None:
                                ticker_headlines[idx] = group["headline"].tolist()[:5]
                except Exception:
                    pass

            long_gate = (vol_ratio >= MIN_VOL_RATIO)
            short_gate = (vol_ratio >= MIN_VOL_RATIO)

            long_fires = int((long_gate & (prob_up >= LONG_THRESHOLD)).sum())
            short_fires = int((short_gate & (prob_up <= SHORT_THRESHOLD)).sum())
            print(f" {ticker}: long signals {long_fires} | short signals {short_fires} | regime routing: {dict(zip(*np.unique(regime_routing, return_counts=True)))}")

            all_prices.append(prices)
            all_probs.append(prob_up)
            all_votes.append(preds_full["votes_up"].ravel())
            all_long_g.append(long_gate)
            all_short_g.append(short_gate)
            all_atr_vol.append(vol_ratio)
            all_enriched.append(enriched_trunc)
            all_row_idx.append(row_idx)
            all_enriched_full.append(enriched)
            all_lstm_probs.append(ticker_lstm_probs)
            all_rf_probs.append(ticker_rf_probs)
            all_xgb_probs.append(ticker_xgb_probs)
            all_sentiment_scores.append(ticker_sent_score)
            all_sentiment_mag.append(ticker_sent_mag)
            all_sentiment_vol.append(ticker_sent_vol)
            all_headlines_text.append(ticker_headlines)
            all_regime_routing.append(regime_routing)

        except Exception as e:
            print(f" Skipping {ticker} in regime-aware ensemble: {e}")
            continue

    return {
        "prices": all_prices,
        "probs": all_probs,
        "votes": all_votes,
        "long_g": all_long_g,
        "short_g": all_short_g,
        "atr_vol": all_atr_vol,
        "enriched": all_enriched,
        "enriched_full": all_enriched_full,
        "row_idx": all_row_idx,
        "lstm_probs": all_lstm_probs,
        "rf_probs": all_rf_probs,
        "xgb_probs": all_xgb_probs,
        "sentiment_scores": all_sentiment_scores,
        "sentiment_mag": all_sentiment_mag,
        "sentiment_vol": all_sentiment_vol,
        "headlines_text": all_headlines_text,
        "regime_routing": all_regime_routing,  # Per-ticker list of specialist names per timestep
    }


def clear_specialist_cache():
    """Clear the specialist model cache. Call this after backtest completion."""
    global _specialist_cache
    _specialist_cache.clear()
    print("Cleared specialist model cache")


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from feature_engineer import FeatureEngineer
    from lstm_model import LSTMModel, make_direction_onehot_from_raw
    from PredictionEngine import PortfolioEngine

    LOOKBACK = 24
    HORIZON  = 8
    LONG_THRESHOLD        = 0.53
    SHORT_THRESHOLD       = 0.45
    MIN_VOL_RATIO         = 0.003
    VOL_REGIME_PERCENTILE = 95

    tickers_file = Path("models/selected_tickers.json")
    if not tickers_file.exists():
        print("ERROR: No selected_tickers.json found. Run lstm_model.py first.")
        exit(1)

    with open(tickers_file, "r") as f:
        tickers = json.load(f)
    print(f"Loaded {len(tickers)} tickers from previous LSTM run: {tickers}")

    lstm     = LSTMModel()
    loader   = DataLoader("data/raw", tickers[0])
    enriched = FeatureEngineer().compute_indicators(loader.load_data())
    pre      = DataPreprocessor(lookback_window=LOOKBACK, forecast_horizon=HORIZON, use_walk_forward=True)
    splits   = pre.preprocess(enriched)
    lstm.build_model(input_shape=(LOOKBACK, splits["X_train"].shape[2]))
    try:
        lstm.load_model()
        print("Loaded pre-trained LSTM weights.")
    except Exception:
        print("ERROR: No saved LSTM found. Run lstm_model.py first.")
        exit(1)

    ensemble = EnsembleModel(lstm_model=lstm)

    print("\n" + "=" * 60)
    print("GATHERING TRAINING DATA FOR TREES ACROSS ALL TICKERS")
    print("=" * 60)
    all_X_train, all_y_train = [], []
    for ticker in tickers:
        loader   = DataLoader("data/raw", ticker)
        enriched = FeatureEngineer().compute_indicators(loader.load_data())
        pre      = DataPreprocessor(lookback_window=LOOKBACK, forecast_horizon=HORIZON, use_walk_forward=True)
        splits   = pre.preprocess(enriched)
        labels, masks, _ = make_direction_onehot_from_raw(
            enriched_df=enriched,
            split_indices={"idx_train": splits["idx_train"], "idx_val": splits["idx_val"], "idx_test": splits["idx_test"]},
            lookback_window=LOOKBACK, forecast_horizon=HORIZON,
        )
        all_X_train.append(splits["X_train"][masks["train"]])
        all_y_train.append(labels["train"][masks["train"]])
    ensemble.train_trees(np.concatenate(all_X_train), np.concatenate(all_y_train))
    ensemble.save_trees()

    print("\n" + "=" * 60)
    print("GATHERING TEST PREDICTIONS ACROSS PORTFOLIO")
    print("=" * 60)
    all_prices, all_probs, all_votes = [], [], []
    all_long_g, all_short_g, all_atr_vol = [], [], []
    eval_y_true, eval_preds_23, eval_preds_33 = [], [], []

    for ticker in tickers:
        loader   = DataLoader("data/raw", ticker)
        enriched = FeatureEngineer().compute_indicators(loader.load_data())
        pre      = DataPreprocessor(lookback_window=LOOKBACK, forecast_horizon=HORIZON, use_walk_forward=True)
        splits   = pre.preprocess(enriched)
        labels, masks, _ = make_direction_onehot_from_raw(
            enriched_df=enriched,
            split_indices={"idx_train": splits["idx_train"], "idx_val": splits["idx_val"], "idx_test": splits["idx_test"]},
            lookback_window=LOOKBACK, forecast_horizon=HORIZON,
        )
        y_test_1d  = np.argmax(labels["test"], axis=1)
        X_test_sig = splits["X_test"][masks["test"]]
        if len(X_test_sig) > 0:
            ep = ensemble.predict(X_test_sig, agreement_threshold=2)
            eval_y_true.append(y_test_1d)
            eval_preds_23.append((ep["votes_up"] >= 2).astype(int))
            eval_preds_33.append((ep["votes_up"] == 3).astype(int))

        preds_full = ensemble.predict(splits["X_test"], agreement_threshold=2)
        row_idx    = splits["idx_test"] + LOOKBACK - 1
        prices     = enriched["Close"].values[row_idx]
        atr_raw    = np.clip(enriched["ATR_14"].values[row_idx], 0,
                             np.percentile(enriched["ATR_14"].values[row_idx], 99))
        vol_ratio  = np.where(prices > 0, atr_raw / prices, 0.0)
        long_gate  = (vol_ratio >= MIN_VOL_RATIO)
        short_gate = (vol_ratio >= MIN_VOL_RATIO)
        prob_up    = preds_full["ensemble_prob_up"].ravel()
        print(f"  {ticker}: long {int((long_gate & (prob_up >= LONG_THRESHOLD)).sum())} "
              f"| short {int((short_gate & (prob_up <= SHORT_THRESHOLD)).sum())}")
        all_prices.append(prices); all_probs.append(prob_up)
        all_votes.append(preds_full["votes_up"].ravel())
        all_long_g.append(long_gate); all_short_g.append(short_gate); all_atr_vol.append(vol_ratio)

    print("\n" + "=" * 60)
    print("ENSEMBLE VOTING METRICS (SIGNIFICANT MOVES ONLY)")
    print("=" * 60)
    if eval_y_true:
        yt  = np.concatenate(eval_y_true)
        p23 = np.concatenate(eval_preds_23)
        p33 = np.concatenate(eval_preds_33)
        for label, preds in [("2/3 Majority", p23), ("3/3 Unanimous", p33)]:
            print(f"{label}:")
            print(f"  Accuracy:  {accuracy_score(yt, preds):.4f}")
            print(f"  Precision: {precision_score(yt, preds, zero_division=0):.4f}")
            print(f"  Recall:    {recall_score(yt, preds, zero_division=0):.4f}")
            print(f"  F1 Score:  {f1_score(yt, preds, zero_division=0):.4f}")

    lengths        = [len(p) for p in all_prices]
    min_len        = int(np.percentile(lengths, 10))
    n_dropped = sum(1 for l in lengths if l < min_len)
    if n_dropped:
        print(f"  {n_dropped} ticker(s) below 10th-percentile length dropped from alignment")
    price_matrix   = np.column_stack([p[-min_len:] for p in all_prices])
    prob_matrix    = np.column_stack([p[-min_len:] for p in all_probs])
    long_g_matrix  = np.column_stack([g[-min_len:] for g in all_long_g])
    short_g_matrix = np.column_stack([g[-min_len:] for g in all_short_g])
    vol_matrix     = np.column_stack([v[-min_len:] for v in all_atr_vol])

    vol_p95         = np.percentile(vol_matrix, VOL_REGIME_PERCENTILE, axis=0)
    high_vol_regime = vol_matrix > vol_p95[np.newaxis, :]
    print(f"Vol-regime halt: {high_vol_regime.mean()*100:.1f}% of timesteps")

    signal_matrix = np.zeros_like(prob_matrix, dtype=int)
    signal_matrix[(prob_matrix >= LONG_THRESHOLD)  & long_g_matrix  & ~high_vol_regime] =  1
    signal_matrix[(prob_matrix <= SHORT_THRESHOLD) & short_g_matrix] = -1
    print(f"Long {np.sum(signal_matrix==1)} | Short {np.sum(signal_matrix==-1)} | Flat {np.sum(signal_matrix==0)}")

    engine = PortfolioEngine(
        initial_capital=10_000.0, base_trade_size_pct=0.20,
        transaction_fee=0.0002, long_safety_sl=0.05, short_safety_sl=0.05,
        horizon=HORIZON,
    )
    metrics = engine.run_portfolio_backtest(price_matrix, signal_matrix, prob_matrix)
    print(f"Return: {metrics['total_return_pct']:+.2f}% | Sharpe: {metrics['sharpe_ratio']:.2f} "
          f"| MaxDD: {metrics['max_drawdown_pct']:.2f}% | Trades: {metrics['total_trades']}")
    engine.plot_equity_curve(save_path="equity_curve_portfolio.png")
