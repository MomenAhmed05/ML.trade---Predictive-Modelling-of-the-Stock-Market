"""
Sentiment Engine

Scores financial news headlines with FinBERT (Araci, 2019) and aggregates
them into a single per-candle sentiment signal in [-1, +1].

FinBERT is a BERT model fine-tuned on financial text.  It outputs three
class probabilities: positive, negative, neutral.  The composite score is:

    s = P(positive) - P(negative)   in [-1, +1]

Headlines are weighted by source reliability tier before aggregation:

    Tier 1 (major financial press)  : weight 1.0
    Tier 2 (financial aggregators)  : weight 0.7
    Tier 3 (retail / social)        : weight 0.4

Four features are returned per candle so the downstream gate can learn
non-linear interactions:

    sentiment_score      - weighted mean composite [-1, +1]
    sentiment_magnitude  - mean |s|, how strong the opinions are
    sentiment_volume     - log(1 + n_headlines), attention/buzz proxy
    sentiment_confidence - weighted mean max-softmax [0.33, 1.0]

Post-hoc signal conditioning formula (applied in ensemble_model.py):

    if confidence >= 0.80:
        effective_alpha = alpha          (full sentiment influence)
    else:
        effective_alpha = 0              (sentiment suppressed)

    p_final = clip(p_ensemble * (1 + effective_alpha * sentiment_score), 0, 1)

where alpha is the base sentiment influence weight (default 0.25).
A hard confidence gate at 0.80 ensures only high-conviction FinBERT
predictions influence the ensemble.  Low-confidence predictions from
noisy or sparse headlines are completely suppressed.
With alpha=0 the ensemble signal is completely unchanged (ablation baseline).

Usage:
    engine = SentimentEngine()
    engine.load_model()                    # downloads FinBERT once, ~400 MB
    feats  = engine.get_sentiment_features(ticker, candle_timestamps, news_loader)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

# Source-tier weight map.  Domain matched on URL / source field substring.
_SOURCE_TIER_WEIGHTS = {
    # Tier 1 - major financial press
    "reuters":      1.0,
    "bloomberg":    1.0,
    "wsj":          1.0,
    "ft":           1.0,
    "cnbc":         1.0,
    "apnews":       1.0,
    # Tier 2 - financial aggregators
    "seekingalpha": 0.7,
    "motleyfool":   0.7,
    "marketwatch":  0.7,
    "yahoo":        0.7,
    "benzinga":     0.7,
    "thestreet":    0.7,
    # Tier 3 - retail / social
    "reddit":       0.4,
    "stocktwits":   0.4,
    "twitter":      0.4,
}
_DEFAULT_WEIGHT = 0.7   # fallback for unknown sources (treated as Tier 2)


class SentimentEngine:
    """
    FinBERT-based sentiment scorer with source-tier weighting.

    Attributes:
        model_name:  HuggingFace model ID (default: ProsusAI/finbert).
        batch_size:  Inference batch size.  Reduce if GPU OOM.
        _pipeline:   Loaded transformers pipeline (None until load_model()).
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipeline = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Downloads and caches FinBERT via HuggingFace transformers.
        First call downloads ~400 MB; subsequent calls are instant.

        Install requirement (one-time):
            pip install transformers torch
        """
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "`transformers` not installed. Run: pip install transformers torch"
            )

        print(f"[SentimentEngine] Loading FinBERT ({self.model_name})...")
        self._pipeline = hf_pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            top_k=None,         # return all three class scores
            truncation=True,
            max_length=512,
        )
        print("[SentimentEngine] FinBERT ready.")

    def score_headlines(self, headlines: list):
        """
        Runs FinBERT on a list of headline strings and returns composite
        scores AND confidence values.

            composite_score = P(positive) - P(negative)   in [-1, +1]
            confidence      = max(P(pos), P(neg), P(neu)) in [0.33, 1.0]

        Args:
            headlines: List of raw headline strings.

        Returns:
            Tuple of (scores, confidences) - two NumPy arrays of shape (N,).
        """
        if self._pipeline is None:
            raise RuntimeError("Call load_model() before score_headlines().")
        if not headlines:
            return np.array([], dtype=float), np.array([], dtype=float)

        scores = []
        confidences = []
        for i in range(0, len(headlines), self.batch_size):
            batch   = headlines[i : i + self.batch_size]
            results = self._pipeline(batch)
            for result in results:
                label_map = {r["label"].lower(): r["score"] for r in result}
                pos = label_map.get("positive", 0.0)
                neg = label_map.get("negative", 0.0)
                neu = label_map.get("neutral", 0.0)
                scores.append(pos - neg)
                confidences.append(max(pos, neg, neu))

        return np.array(scores, dtype=float), np.array(confidences, dtype=float)

    def get_sentiment_features(
        self,
        ticker: str,
        candle_timestamps: pd.Series,
        news_loader,
        lookback_hours: int = 24,
        source_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Builds three sentiment features for every candle timestamp.

        Workflow:
          1. Fetch headlines from NewsLoader for (ticker, candle_timestamps)
          2. Score each headline with FinBERT
          3. Apply source-tier weight
          4. Aggregate into (sentiment_score, sentiment_magnitude,
             sentiment_volume) per candle
          5. Zero-pad candles with no news coverage

        Args:
            ticker:            Ticker symbol.
            candle_timestamps: Hourly candle open times (DatetimeSeries).
            news_loader:       Initialised NewsLoader instance.
            lookback_hours:    Aggregation window fed to NewsLoader.
            source_col:        Optional column name in the headlines DataFrame
                               that identifies the news source for tier-weighting.
                               If None, all headlines use the default weight.

        Returns:
            DataFrame indexed by candle_timestamps with columns:
                sentiment_score      [-1, +1]
                sentiment_magnitude  [0,  1]
                sentiment_volume     [0, inf)  (log-scaled)
        """
        headlines_df = news_loader.get_headlines(
            ticker, candle_timestamps, lookback_hours=lookback_hours
        )

        n         = len(candle_timestamps)
        out_score = np.zeros(n, dtype=float)
        out_mag   = np.zeros(n, dtype=float)
        out_vol   = np.zeros(n, dtype=float)
        out_conf  = np.zeros(n, dtype=float)

        if headlines_df.empty:
            return self._build_output_df(candle_timestamps, out_score, out_mag, out_vol, out_conf)

        raw_scores, raw_conf = self.score_headlines(headlines_df["headline"].tolist())
        headlines_df = headlines_df.copy()
        headlines_df["composite"] = raw_scores
        headlines_df["confidence"] = raw_conf

        if source_col and source_col in headlines_df.columns:
            headlines_df["weight"] = headlines_df[source_col].apply(self._source_weight)
        else:
            headlines_df["weight"] = _DEFAULT_WEIGHT

        # --- FIX vs experiment-fix: normalise both sides to UTC before matching ---
        # The parquet cache may store candle_ts as timezone-naive if it was
        # written before the UTC localisation fix.  Coercing both to UTC-aware
        # prevents the silent idx=None path that drops all headline data.
        cts = pd.to_datetime(candle_timestamps, errors="coerce")
        if cts.dt.tz is None:
            cts = cts.dt.tz_localize("UTC")
        else:
            cts = cts.dt.tz_convert("UTC")

        if headlines_df["candle_ts"].dt.tz is None:
            headlines_df["candle_ts"] = pd.to_datetime(
                headlines_df["candle_ts"]
            ).dt.tz_localize("UTC")
        else:
            headlines_df["candle_ts"] = headlines_df["candle_ts"].dt.tz_convert("UTC")

        ct_index = {ct: idx for idx, ct in enumerate(cts)}

        for ct, group in headlines_df.groupby("candle_ts"):
            idx = ct_index.get(ct)
            if idx is None:
                continue
            w     = group["weight"].values
            s     = group["composite"].values
            c     = group["confidence"].values
            w_sum = w.sum()
            if w_sum > 0:
                out_score[idx] = float(np.dot(w, s) / w_sum)
                out_mag[idx]   = float(np.dot(w, np.abs(s)) / w_sum)
                out_conf[idx]  = float(np.dot(w, c) / w_sum)
            out_vol[idx] = float(np.log1p(len(group)))

        return self._build_output_df(candle_timestamps, out_score, out_mag, out_vol, out_conf)

    def apply_sentiment_gate(
        self,
        prob_array: np.ndarray,
        sentiment_scores: np.ndarray,
        alpha: float = 0.25,
        confidence: np.ndarray = None,
        confidence_floor: float = 0.80,
    ) -> np.ndarray:
        """
        Post-hoc signal conditioning with hard confidence gate:

            if confidence >= confidence_floor:
                effective_alpha = alpha
            else:
                effective_alpha = 0   (sentiment completely ignored)

            p_final = clip(p_ensemble * (1 + effective_alpha * sentiment_score), 0, 1)

        Only applies sentiment when FinBERT is highly confident (default >=0.80).
        Low-confidence predictions (noisy/mixed headlines) are fully suppressed,
        letting the ensemble signal pass through unchanged.

        Args:
            prob_array:        Raw ensemble UP-probabilities, shape (T,).
            sentiment_scores:  Sentiment scores aligned to prob_array, shape (T,).
            alpha:             Base sentiment influence weight (0.0 = off).
            confidence:        Per-timestep FinBERT confidence in [0.33, 1.0].
                               If None, alpha is used as-is (backward compatible).
            confidence_floor:  Minimum confidence to apply sentiment (default 0.80).
                               Below this threshold, sentiment has zero influence.

        Returns:
            Conditioned probability array of shape (T,), values in [0, 1].
        """
        if confidence is not None:
            mask = confidence >= confidence_floor
            effective_alpha = np.where(mask, alpha, 0.0)
        else:
            effective_alpha = alpha
        conditioned = prob_array * (1.0 + effective_alpha * sentiment_scores)
        return np.clip(conditioned, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _source_weight(self, source: str) -> float:
        """Returns the tier weight for a given source string."""
        if not isinstance(source, str):
            return _DEFAULT_WEIGHT
        source_lower = source.lower()
        for key, weight in _SOURCE_TIER_WEIGHTS.items():
            if key in source_lower:
                return weight
        return _DEFAULT_WEIGHT

    @staticmethod
    def _build_output_df(
        candle_timestamps: pd.Series,
        score: np.ndarray,
        magnitude: np.ndarray,
        volume: np.ndarray,
        confidence: np.ndarray = None,
    ) -> pd.DataFrame:
        """Packages the feature arrays into a tidy DataFrame."""
        data = {
            "sentiment_score":     score,
            "sentiment_magnitude": magnitude,
            "sentiment_volume":    volume,
        }
        if confidence is not None:
            data["sentiment_confidence"] = confidence
        return pd.DataFrame(data, index=candle_timestamps)
