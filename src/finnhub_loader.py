"""
Finnhub News Loader

Fetches recent company news from the Finnhub API for tickers that fall
outside the FNSPID dataset's coverage (1999-2023). Headlines are returned
in the same schema that sentiment_engine.py expects so the existing
FinBERT scoring pipeline works unchanged.

Free tier: 60 API calls per minute, company-news endpoint.
Register at https://finnhub.io to get an API key.

Usage:
    loader = FinnhubNewsLoader()  # reads FINNHUB_API_KEY from env
    headlines = loader.get_headlines("AAPL", "2024-01-01", "2024-06-30")
"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta


_CACHE_DIR = Path("data/finnhub_cache")

# Rate limit: 60 calls/min on free tier
_MIN_CALL_INTERVAL = 1.1  # seconds between calls (safe margin)


class FinnhubNewsLoader:
    """
    Fetches company news from Finnhub and returns DataFrames compatible
    with the existing NewsLoader/SentimentEngine interface.
    """

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = str(_CACHE_DIR)):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Finnhub API key required. Set FINNHUB_API_KEY env var or pass api_key. "
                "Register free at https://finnhub.io"
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call_time = 0.0

    def _rate_limit(self):
        """Enforce minimum interval between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < _MIN_CALL_INTERVAL:
            time.sleep(_MIN_CALL_INTERVAL - elapsed)
        self._last_call_time = time.time()

    def get_headlines(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """
        Fetch news headlines for a ticker in the given date range.

        Args:
            ticker:    Stock symbol (e.g. "AAPL")
            from_date: Start date "YYYY-MM-DD"
            to_date:   End date "YYYY-MM-DD"

        Returns:
            DataFrame with columns [timestamp, ticker, headline, source]
            matching the FNSPID/NewsLoader schema.
        """
        ticker = ticker.upper()

        # Check cache first
        cached = self._load_cache(ticker, from_date, to_date)
        if cached is not None:
            return cached

        # Finnhub company-news endpoint accepts max ~1 year per call,
        # so chunk into 90-day windows
        all_articles = []
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")

        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=90), end)
            articles = self._fetch_chunk(
                ticker,
                current.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            all_articles.extend(articles)
            current = chunk_end + timedelta(days=1)

        if not all_articles:
            print(f"  [Finnhub] {ticker}: no headlines found for {from_date} to {to_date}")
            return pd.DataFrame(columns=["timestamp", "ticker", "headline", "source"])

        df = pd.DataFrame(all_articles)
        df = df.rename(columns={
            "datetime": "timestamp",
            "headline": "headline",
            "source": "source",
        })
        df["ticker"] = ticker
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df[["timestamp", "ticker", "headline", "source"]].drop_duplicates(
            subset=["timestamp", "headline"]
        )
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Cache results
        self._save_cache(ticker, from_date, to_date, df)

        print(f"  [Finnhub] {ticker}: fetched {len(df)} headlines "
              f"({from_date} to {to_date})")
        return df

    def _fetch_chunk(self, ticker: str, from_date: str, to_date: str) -> list:
        """Fetch a single chunk from the Finnhub API."""
        import requests

        self._rate_limit()

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": self.api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                print(f"  [Finnhub] Rate limited, waiting 60s...")
                time.sleep(60)
                return self._fetch_chunk(ticker, from_date, to_date)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  [Finnhub] API error for {ticker}: {e}")
            return []

    def _cache_key(self, ticker: str, from_date: str, to_date: str) -> Path:
        return self.cache_dir / f"{ticker}_{from_date}_{to_date}.parquet"

    def _load_cache(self, ticker: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        cache_file = self._cache_key(ticker, from_date, to_date)
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            print(f"  [Finnhub] {ticker}: loaded {len(df)} headlines from cache")
            return df
        return None

    def _save_cache(self, ticker: str, from_date: str, to_date: str, df: pd.DataFrame):
        cache_file = self._cache_key(ticker, from_date, to_date)
        df.to_parquet(cache_file, index=False)
