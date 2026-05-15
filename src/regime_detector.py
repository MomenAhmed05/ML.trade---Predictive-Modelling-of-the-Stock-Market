import numpy as np
import pandas as pd
import pickle
from pathlib import Path


class RegimeDetector:
    """
    3-state Gaussian HMM regime classifier.

    States are mapped to:
      BULL   — the default / safe fallback
      BEAR   — only assigned when a state has genuinely negative drift
      CRISIS — only assigned when a state has genuinely elevated volatility

    Feature set is deliberately balanced between directional signals
    (return_20d, return_60d, drawdown_from_peak_60, down_bar_ratio_60)
    and volatility context (vol_percentile_60, ROC_20). Previous versions
    were vol-dominated (4 of 5 features were vol/range indicators), which
    caused the HMM to cluster on volatility buckets and mislabel
    "high-vol but trending up" periods as BEAR.

    Labels are gated by absolute thresholds — not just sharpe rank — so
    that if training data has no genuinely bearish or crisis state, the
    detector refuses to assign those labels. REGIME_CONFIG then only
    activates defensive params when the evidence supports them.
    """

    REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "CRISIS"}

    # Feature-matrix column indices (match _build_feature_matrix order)
    # Used by the labeller to reason about state means.
    _COL_RET_20D      = 0
    _COL_RET_60D      = 1
    _COL_DD_60        = 2
    _COL_DOWN_RATIO   = 3
    _COL_VOL_PCTILE   = 4
    _COL_ROC_20       = 5

    # Absolute thresholds for label validation. These are applied to the
    # state MEANS (averaged feature value across all bars the state occupies).
    # A state must clear its threshold by more than just ranking lowest/highest.
    _RETURN_BEAR_THRESH = -0.005   # mean return_60d must be < -0.5% per 60h window
    _VOL_CRISIS_THRESH  =  0.65    # mean vol_percentile must be > 65th pct
    _CRISIS_RETURN_CAP  =  0.0     # CRISIS also requires non-positive drift

    def __init__(self, n_components: int = 3, save_path: str = "models/regime_detector.pkl"):
        self.n_components = n_components
        self.save_path = Path(save_path)
        self.model = None
        self.state_to_regime = {}

    # ------------------------------------------------------------------
    # Feature engineering — computed locally from Close + vol_percentile_60
    # ------------------------------------------------------------------
    def _build_feature_matrix(self, enriched_df) -> np.ndarray:
        """
        Build the regime-detection feature matrix.

        Expects `enriched_df` to contain at minimum:
          - Close              : raw close price (for drift/drawdown features)
          - vol_percentile_60  : rolling vol percentile (already upstream)

        All other features are computed on the fly so the pipeline's existing
        enriched feature set does not need to change.
        """
        required_cols = ["Close", "vol_percentile_60"]
        missing = [c for c in required_cols if c not in enriched_df.columns]
        if missing:
            raise ValueError(
                f"[RegimeDetector] Missing columns in enriched_df: {missing}"
            )

        close = enriched_df["Close"].astype(float)
        ret_1 = close.pct_change()

        # 1. Multi-horizon log return — the "is this trending up or down" signal
        return_20d = np.log(close / close.shift(20))
        return_60d = np.log(close / close.shift(60))

        # 2. Drawdown from 60-bar peak — captures sustained bear pressure
        rolling_peak = close.rolling(window=60, min_periods=10).max()
        drawdown_from_peak_60 = (rolling_peak - close) / rolling_peak.replace(0, np.nan)

        # 3. Down-bar ratio — fraction of last 60 bars that closed red.
        #    Complements raw returns (a flat-but-chippy market can have 60%
        #    down bars even while net return ~ 0).
        down_bars = (ret_1 < 0).astype(float)
        down_bar_ratio_60 = down_bars.rolling(window=60, min_periods=10).mean()

        # 4. Volatility percentile — the one vol feature we keep
        vol_percentile_60 = enriched_df["vol_percentile_60"].astype(float)

        # 5. ROC_20 — 20-period rate of change. Less noisy than the existing
        #    ROC_10, and carries residual momentum signal independent of the
        #    log returns above.
        roc_20 = close.pct_change(20) * 100.0

        feat_df = pd.DataFrame({
            "return_20d":            return_20d,
            "return_60d":            return_60d,
            "drawdown_from_peak_60": drawdown_from_peak_60,
            "down_bar_ratio_60":     down_bar_ratio_60,
            "vol_percentile_60":     vol_percentile_60,
            "ROC_20":                roc_20,
        })

        # NOTE: we ffill then fill remaining leading-NaN rows with 0.
        # Zero-filling warmup rows biases them toward "flat / at peak / no
        # downbars" which naturally steers them into a BULL-like state — the
        # safest default for the handful of leading bars that lack history.
        X = feat_df.ffill().fillna(0).values.astype(np.float32)
        return X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, enriched_df):
        """
        Fit the HMM on a single-ticker enriched DataFrame.

        For the multi-ticker training path, use fit_and_save which computes
        features per-ticker (critical: rolling / shift operations must never
        cross ticker boundaries or log returns become meaningless).
        """
        X = self._build_feature_matrix(enriched_df)
        self._fit_on_matrix(X)

    def _fit_on_matrix(self, X: np.ndarray, lengths=None):
        """
        Internal: fit HMM on a pre-built feature matrix + do state labelling.

        If X stacks multiple independent sequences (e.g., per-ticker feature
        matrices concatenated with np.vstack), pass `lengths` so hmmlearn
        doesn't learn transitions across sequence boundaries.
        """
        from hmmlearn.hmm import GaussianHMM
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        self.model.fit(X, lengths=lengths)

        means = self.model.means_
        dir_col = self._COL_RET_60D
        vol_col = self._COL_VOL_PCTILE

        # Rank candidate states — lowest direction → bear, highest vol → crisis
        by_direction = np.argsort(means[:, dir_col])          # ascending
        by_vol       = np.argsort(means[:, vol_col])[::-1]    # descending

        bear_candidate   = int(by_direction[0])
        crisis_candidate = int(by_vol[0])

        # Validate against absolute thresholds, not just ranks
        bear_valid = means[bear_candidate, dir_col] < self._RETURN_BEAR_THRESH
        crisis_valid = (
            means[crisis_candidate, vol_col] > self._VOL_CRISIS_THRESH
            and means[crisis_candidate, dir_col] < self._CRISIS_RETURN_CAP
        )

        bear_state   = bear_candidate   if bear_valid   else None
        crisis_state = crisis_candidate if crisis_valid else None

        # Resolve conflict: if the same state qualifies for both, keep whichever
        # label the state clears its threshold by more convincingly.
        if bear_state is not None and bear_state == crisis_state:
            bear_margin   = self._RETURN_BEAR_THRESH - means[bear_candidate,   dir_col]
            crisis_margin = means[crisis_candidate, vol_col] - self._VOL_CRISIS_THRESH
            if crisis_margin >= bear_margin:
                bear_state = None
            else:
                crisis_state = None

        # Assign labels — any state that did not qualify for BEAR or CRISIS
        # falls back to BULL (the permissive default).
        self.state_to_regime = {}
        for s in range(self.n_components):
            if s == bear_state:
                self.state_to_regime[s] = "BEAR"
            elif s == crisis_state:
                self.state_to_regime[s] = "CRISIS"
            else:
                self.state_to_regime[s] = "BULL"

        print(f"[RegimeDetector] State mapping: {self.state_to_regime}")
        if bear_state is None:
            print(
                f"[RegimeDetector] No genuine BEAR state in training data "
                f"(lowest mean return_60d = "
                f"{means[bear_candidate, dir_col]:+.4f}, threshold "
                f"{self._RETURN_BEAR_THRESH}). All non-CRISIS bars will "
                f"be labelled BULL."
            )
        if crisis_state is None:
            print(
                f"[RegimeDetector] No genuine CRISIS state in training data "
                f"(highest vol_percentile mean = "
                f"{means[crisis_candidate, vol_col]:.3f}, threshold "
                f"{self._VOL_CRISIS_THRESH})."
            )
        print(f"[RegimeDetector] State means:\n{means}")

        # ── Fix 5: post-fit sanity warnings ──
        # Catch cases where a state was labelled but only marginally clears
        # its threshold — a retrain signal for future runs.
        for state, label in self.state_to_regime.items():
            m_dir = means[state, dir_col]
            m_vol = means[state, vol_col]
            if label == "BEAR" and m_dir > self._RETURN_BEAR_THRESH * 1.5:
                # shouldn't fire (bear_valid checks <threshold), but defensive
                print(
                    f"[RegimeDetector] WARNING: state {state} labelled BEAR "
                    f"but return_60d mean {m_dir:+.4f} is only marginally "
                    f"below threshold."
                )
            if label == "CRISIS" and m_vol < self._VOL_CRISIS_THRESH * 1.1:
                print(
                    f"[RegimeDetector] WARNING: state {state} labelled CRISIS "
                    f"but vol_percentile mean {m_vol:.3f} only marginally "
                    f"above threshold — elevated vol may not be extreme."
                )
            if label == "BULL" and m_dir < self._RETURN_BEAR_THRESH:
                print(
                    f"[RegimeDetector] WARNING: state {state} labelled BULL "
                    f"but return_60d mean {m_dir:+.4f} is actually "
                    f"negative — another state cleared BEAR first."
                )

        # ── Diagnostic: per-state occupancy on the training window ──
        try:
            states_train = self.model.predict(X)
            print("[RegimeDetector] Training-window state occupancy:")
            print(
                f"  {'State':<6}{'Label':<8}{'Count':>9}{'Pct':>8}"
                f"{'ret20d':>10}{'ret60d':>10}{'dd60':>8}"
                f"{'down%':>8}{'vol%':>8}{'ROC20':>9}"
            )
            total = len(states_train)
            for s in range(self.n_components):
                mask  = states_train == s
                count = int(mask.sum())
                pct   = count / max(total, 1) * 100.0
                label = self.state_to_regime.get(s, "?")
                m     = means[s]
                print(
                    f"  {s:<6}{label:<8}{count:>9}{pct:>7.1f}%"
                    f"{m[0]:>+10.4f}{m[1]:>+10.4f}{m[2]:>8.3f}"
                    f"{m[3]:>8.3f}{m[4]:>8.3f}{m[5]:>+9.3f}"
                )
        except Exception as _diag_exc:
            print(f"[RegimeDetector] Diagnostic print failed (non-fatal): {_diag_exc}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_regime_series(self, enriched_df) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("[RegimeDetector] Model not fitted. Call fit() first.")
        X      = self._build_feature_matrix(enriched_df)
        states = self.model.predict(X)
        regimes = np.array([self.state_to_regime[s] for s in states], dtype=object)
        # Validate output
        valid = {"BULL", "BEAR", "CRISIS"}
        unknown = set(regimes) - valid
        if unknown:
            print(f"[RegimeDetector] WARNING: Unknown regime labels found: {unknown}. Replacing with BULL.")
            regimes[~np.isin(regimes, list(valid))] = "BULL"
        return regimes

    def predict_current_regime(self, enriched_df, lookback: int = 120) -> str:
        """
        Return the most common regime over the trailing `lookback` bars.

        Default lookback bumped from 48 → 120 because the new feature set
        uses 60-bar rolling windows; fewer than ~60 bars of history produces
        all-NaN warmup rows that ffill can't fix.
        """
        recent  = enriched_df.iloc[-lookback:]
        regimes = self.predict_regime_series(recent)
        unique, counts = np.unique(regimes, return_counts=True)
        return unique[np.argmax(counts)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, save_path: str = None):
        """
        Persist the fitted detector.

        Parameters
        ----------
        save_path : str, optional
            Override the instance save_path. If omitted, uses self.save_path.
            Accepts either a string path or None.
        """
        if save_path is not None:
            self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)
        print(f"[RegimeDetector] Saved to {self.save_path}")

    @classmethod
    def load(cls, save_path: str = "models/regime_detector.pkl"):
        with open(save_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def fit_and_save(
        cls,
        enriched_dfs: list,
        save_path: str = "models/regime_detector.pkl",
        n_components: int = 3,
    ) -> "RegimeDetector":
        """
        Convenience classmethod: concatenate a list of enriched DataFrames,
        fit the HMM, save, and return the fitted detector.

        This is the canonical entry point for pipeline.py so that fit() and
        save() are never called separately (avoids the path-mismatch footgun).

        Parameters
        ----------
        enriched_dfs : list of pd.DataFrame
            Training-window enriched DataFrames (one per ticker, pre-truncated
            to anchor_end_date before being passed in).
        save_path : str
            Where to persist the fitted model.
        n_components : int
            Number of HMM states (default 3 → BULL / BEAR / CRISIS).

        Returns
        -------
        RegimeDetector
            Fitted and saved detector instance.
        """
        if not enriched_dfs:
            raise ValueError("[RegimeDetector] fit_and_save called with empty enriched_dfs list.")

        detector = cls(n_components=n_components, save_path=save_path)

        # CRITICAL: features must be built per-ticker, not on a concatenated +
        # sorted DataFrame. Rolling / shift operations across ticker boundaries
        # produce nonsense (log returns computed between MSFT price and AAPL
        # price 20 rows later). We build one feature matrix per ticker, then
        # stack them along axis 0 — the HMM's i.i.d. emission assumption means
        # the rows don't need to be chronologically coherent across tickers,
        # only within each ticker.
        feature_matrices = []
        total_rows = 0
        for df in enriched_dfs:
            X = detector._build_feature_matrix(df)
            if len(X) > 0:
                feature_matrices.append(X)
                total_rows += len(X)

        if not feature_matrices:
            raise ValueError("[RegimeDetector] fit_and_save produced zero feature rows.")

        X_combined = np.vstack(feature_matrices)
        lengths = [len(m) for m in feature_matrices]
        print(f"[RegimeDetector] Fitting on {total_rows} rows from "
              f"{len(feature_matrices)} ticker(s) (per-ticker features stacked).")

        detector._fit_on_matrix(X_combined, lengths=lengths)
        detector.save()
        return detector
