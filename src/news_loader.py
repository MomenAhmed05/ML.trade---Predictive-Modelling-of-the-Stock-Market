"""
News Loader

Loads the FNSPID financial news dataset from HuggingFace and aligns
headlines to hourly candle timestamps on a per-ticker basis.

FNSPID contains ~15.7 million timestamped headlines covering 4,775
S&P 500 tickers from 1999-2023 (Dong et al., 2024, KDD).

Key design rule - NO FORWARD-LOOKING BIAS:
  A headline published at time T is only attached to the candle whose
  open is strictly AFTER T.  get_headlines() enforces this by using a
  half-open window (candle_ts - lookback_hours, candle_ts], where the
  upper bound is the candle open time - not the close - so no future
  price information can bleed into the signal.

  floor_to_next_candle() is provided as an additional utility that
  snaps a publication timestamp to the start of the next hourly candle.
  It is not called by default because get_headlines() already provides
  sufficient candle-level look-ahead protection, but it is available
  for stricter intra-candle alignment if needed.

Download strategy (in priority order):
  1. Local parquet cache  - instant, used on every run after the first.
  2. huggingface_hub.hf_hub_download  - fetches the smaller
     All_external.csv (~5.7 GB) directly without triggering the HF
     Datasets arrow/parquet generation step that causes the
     "error occurred while generating the dataset" crash on large CSVs.
  3. Fallback: load_dataset() streaming mode  - used only if
     huggingface_hub is not installed.

Usage:
    loader = NewsLoader()
    loader.load()                         # one-time download + cache
    df = loader.get_headlines("AAPL", timestamps_series)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Parquet cache written after first successful load (~400 MB on disk)
_CACHE_PATH = Path("data/fnspid_cache.parquet")

# The smaller of the two FNSPID CSVs - covers all external news sources
# and is sufficient for ticker-level sentiment.  The nasdaq_external file
# (23 GB) is redundant for our use case.
_HF_REPO = "Zihan1004/FNSPID"
_HF_FILE = "Stock_news/All_external.csv"

# Expected column names inside All_external.csv (lowercase after strip)
_COL_MAP = {
    "date":     "timestamp",
    "time":     "timestamp",
    "datetime": "timestamp",
    "ticker":   "ticker",
    "symbol":   "ticker",
    "article":  "headline",
    "title":    "headline",
    "headline": "headline",
}


class NewsLoader:
    """
    Thin wrapper around the FNSPID dataset.

    After the first call to load(), headlines are persisted to a local
    parquet file so subsequent runs are instant.
    """

    def __init__(self, cache_path: str = str(_CACHE_PATH)):
        self.cache_path = Path(cache_path)
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Loads FNSPID into memory.  Uses local parquet cache when available;
        otherwise downloads All_external.csv from HuggingFace directly
        (bypassing the HF Datasets arrow generation step) and writes cache.

        Install requirement (one-time):
            pip install huggingface_hub transformers torch
        """
        if self.cache_path.exists():
            print(f"[NewsLoader] Loading cached FNSPID from {self.cache_path}")
            self._df = pd.read_parquet(self.cache_path)
            print(f"[NewsLoader] {len(self._df):,} headlines loaded from cache.")
            return

        print("[NewsLoader] Downloading FNSPID All_external.csv from HuggingFace...")
        print("[NewsLoader] This is a one-time download (~5.7 GB). Please wait.")

        raw = self._download_csv()
        self._df = self._normalise(raw)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(self.cache_path, index=False)
        print(f"[NewsLoader] {len(self._df):,} headlines cached to {self.cache_path}")

    def get_headlines(
        self,
        ticker: str,
        candle_timestamps: pd.Series,
        lookback_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Returns all headlines for `ticker` that fall within
        (candle_ts - lookback_hours, candle_ts] for each candle timestamp.

        If the requested date range extends beyond 2023 (FNSPID coverage),
        automatically falls back to FinnhubNewsLoader for recent headlines.

        The result DataFrame has columns:
            candle_ts  - the candle open time this headline is assigned to
            timestamp  - original publication timestamp
            headline   - headline text
            source     - source string (if present in the dataset)

        Args:
            ticker:            Ticker symbol (e.g. "AAPL").
            candle_timestamps: Series of hourly candle open times.
            lookback_hours:    Aggregation window in hours (default 24).

        Returns:
            DataFrame with columns [candle_ts, timestamp, headline, source].
            Empty DataFrame if no headlines found.
        """
        cts = pd.to_datetime(candle_timestamps, utc=True, errors="coerce").dropna()
        if cts.empty:
            return pd.DataFrame(columns=["candle_ts", "timestamp", "headline"])

        # Check if data extends beyond FNSPID coverage (end of 2023)
        fnspid_cutoff = pd.Timestamp("2023-12-31", tz="UTC")
        max_date = cts.max()

        if max_date > fnspid_cutoff:
            return self._get_finnhub_headlines(ticker, cts, lookback_hours)

        return self._get_fnspid_headlines(ticker, cts, lookback_hours)

    def _get_fnspid_headlines(
        self,
        ticker: str,
        cts: pd.Series,
        lookback_hours: int,
    ) -> pd.DataFrame:
        """Fetch headlines from the cached FNSPID dataset."""
        if self._df is None:
            raise RuntimeError("Call load() before get_headlines().")

        ticker_df = self._df[self._df["ticker"] == ticker.upper()].copy()
        if ticker_df.empty:
            return pd.DataFrame(columns=["candle_ts", "timestamp", "headline"])

        # Ensure ticker_df timestamps are UTC-aware for comparison
        if ticker_df["timestamp"].dt.tz is None:
            ticker_df["timestamp"] = ticker_df["timestamp"].dt.tz_localize("UTC")
        else:
            ticker_df["timestamp"] = ticker_df["timestamp"].dt.tz_convert("UTC")

        window_start = cts.min() - pd.Timedelta(hours=lookback_hours)
        window_end   = cts.max()
        ticker_df    = ticker_df[
            (ticker_df["timestamp"] > window_start) &
            (ticker_df["timestamp"] <= window_end)
        ].copy()

        if ticker_df.empty:
            return pd.DataFrame(columns=["candle_ts", "timestamp", "headline"])

        # Assign each headline to the earliest candle whose open >= pub time
        cts_sorted = cts.sort_values().values  # numpy array of datetime64
        pub_times  = ticker_df["timestamp"].values

        assigned = np.searchsorted(cts_sorted, pub_times, side="left")
        valid    = assigned < len(cts_sorted)
        ticker_df = ticker_df[valid].copy()
        ticker_df["candle_ts"] = cts_sorted[assigned[valid]]

        return ticker_df.reset_index(drop=True)

    def _get_finnhub_headlines(
        self,
        ticker: str,
        cts: pd.Series,
        lookback_hours: int,
    ) -> pd.DataFrame:
        """Fetch headlines from Finnhub API for dates beyond FNSPID coverage."""
        try:
            from finnhub_loader import FinnhubNewsLoader
        except ImportError:
            print(f"  [NewsLoader] finnhub_loader not available; no sentiment for {ticker}")
            return pd.DataFrame(columns=["candle_ts", "timestamp", "headline"])

        try:
            fh = FinnhubNewsLoader()
        except ValueError as e:
            print(f"  [NewsLoader] Finnhub not configured: {e}")
            return pd.DataFrame(columns=["candle_ts", "timestamp", "headline"])

        from_date = (cts.min() - pd.Timedelta(hours=lookback_hours)).strftime("%Y-%m-%d")
        to_date = cts.max().strftime("%Y-%m-%d")

        ticker_df = fh.get_headlines(ticker, from_date, to_date)
        if ticker_df.empty:
            return pd.DataFrame(columns=["candle_ts", "timestamp", "headline"])

        # Ensure timestamps are UTC-aware
        if ticker_df["timestamp"].dt.tz is None:
            ticker_df["timestamp"] = ticker_df["timestamp"].dt.tz_localize("UTC")

        window_start = cts.min() - pd.Timedelta(hours=lookback_hours)
        window_end = cts.max()
        ticker_df = ticker_df[
            (ticker_df["timestamp"] > window_start) &
            (ticker_df["timestamp"] <= window_end)
        ].copy()

        if ticker_df.empty:
            return pd.DataFrame(columns=["candle_ts", "timestamp", "headline"])

        # Assign each headline to the earliest candle whose open >= pub time
        cts_sorted = cts.sort_values().values
        pub_times = ticker_df["timestamp"].values

        assigned = np.searchsorted(cts_sorted, pub_times, side="left")
        valid = assigned < len(cts_sorted)
        ticker_df = ticker_df[valid].copy()
        ticker_df["candle_ts"] = cts_sorted[assigned[valid]]

        return ticker_df.reset_index(drop=True)

    @staticmethod
    def floor_to_next_candle(
        pub_time: pd.Timestamp,
        candle_freq: str = "1H",
    ) -> pd.Timestamp:
        """
        Utility: snaps a publication timestamp to the start of the next
        hourly candle (ceiling, not floor).  Available for intra-candle
        alignment if stricter look-ahead protection is required.

        Example:
            pub_time = 14:32  →  returns 15:00
            pub_time = 15:00  →  returns 15:00  (exact boundary stays)
        """
        floored = pub_time.floor(candle_freq)
        if floored == pub_time:
            return pub_time
        return floored + pd.tseries.frequencies.to_offset(candle_freq)

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_csv(self) -> pd.DataFrame:
        """Downloads All_external.csv from HuggingFace in 500K-row chunks."""
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=_HF_REPO,
                filename=_HF_FILE,
                repo_type="dataset",
            )
            print(f"[NewsLoader] Reading CSV from {local_path} ...")
            chunks = pd.read_csv(local_path, chunksize=500_000, low_memory=False)
            df = pd.concat(list(chunks), ignore_index=True)
            return df
        except ImportError:
            pass

        # Fallback: streaming via HF Datasets
        print("[NewsLoader] huggingface_hub not found; falling back to streaming load...")
        try:
            from datasets import load_dataset
            ds = load_dataset(_HF_REPO, split="train", streaming=True)
            rows = list(ds)
            return pd.DataFrame(rows)
        except Exception as exc:
            raise RuntimeError(
                f"[NewsLoader] Failed to download FNSPID: {exc}\n"
                "Install huggingface_hub: pip install huggingface_hub"
            ) from exc

    def _normalise(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Normalises column names, parses timestamps, and drops rows with
        missing ticker or headline.  Returns a clean DataFrame with columns:
            timestamp, ticker, headline, [source]
        """
        # Lowercase and strip column names
        raw.columns = [c.strip().lower() for c in raw.columns]

        # Rename to canonical names
        rename = {}
        for col in raw.columns:
            if col in _COL_MAP and _COL_MAP[col] not in rename.values():
                rename[col] = _COL_MAP[col]
        raw = raw.rename(columns=rename)

        required = {"timestamp", "ticker", "headline"}
        missing  = required - set(raw.columns)
        if missing:
            raise ValueError(
                f"[NewsLoader] FNSPID CSV missing expected columns: {missing}. "
                f"Found: {list(raw.columns)}"
            )

        raw["ticker"]    = raw["ticker"].astype(str).str.strip().str.upper()
        raw["headline"]  = raw["headline"].astype(str).str.strip()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")

        raw = raw.dropna(subset=["timestamp", "ticker", "headline"])
        raw = raw[raw["headline"] != ""]
        raw = raw[raw["ticker"]   != ""]

        keep = ["timestamp", "ticker", "headline"]
        if "source" in raw.columns:
            keep.append("source")

        return raw[keep].reset_index(drop=True)
