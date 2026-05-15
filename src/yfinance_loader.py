"""
Yahoo Finance Data Loader

Downloads recent hourly OHLCV data directly from Yahoo Finance's chart API
and saves it in FirstRateData-compatible .txt format so the existing
DataLoader works unchanged.

Uses plain `requests` - no yfinance or curl_cffi dependency for downloads.
Yahoo Finance provides ~730 calendar days (~2 years) of hourly data.

Usage:
    from yfinance_loader import download_yfinance_tickers

    # Download data for selected tickers
    downloaded = download_yfinance_tickers(
        tickers=["AAPL", "TSLA", "NVDA"],
        cache_dir="data/yfinance_cache"
    )
    # Now use data_path="data/yfinance_cache" with the existing pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
import requests
import logging

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


def _get_yahoo_session() -> requests.Session:
    """
    Create a requests.Session with Yahoo Finance cookies and crumb.
    The crumb is required for Yahoo's v8 chart API.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)

    # Hit a Yahoo endpoint to get cookies (including the consent cookie)
    try:
        session.get("https://fc.yahoo.com", timeout=10)
    except Exception:
        pass  # We just need the cookies, response can fail

    # Get the crumb token
    try:
        crumb_resp = session.get(
            "https://query2.finance.yahoo.com/v1/test/getcrumb",
            timeout=10,
        )
        if crumb_resp.status_code == 200:
            session._crumb = crumb_resp.text.strip()
        else:
            session._crumb = ""
    except Exception:
        session._crumb = ""

    return session


def _download_ticker_raw(
    ticker: str,
    session: requests.Session,
    period: str = "2y",
    interval: str = "1h",
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single ticker from Yahoo Finance chart API.
    Returns a DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume],
    or None on failure.
    """
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "range": period,
        "interval": interval,
        "includePrePost": "false",
    }
    crumb = getattr(session, "_crumb", "")
    if crumb:
        params["crumb"] = crumb

    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    try:
        result = data["chart"]["result"]
        if not result:
            return None
        result = result[0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Open":   quote["open"],
            "High":   quote["high"],
            "Low":    quote["low"],
            "Close":  quote["close"],
            "Volume": quote["volume"],
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))

        df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df
    except (KeyError, TypeError, IndexError):
        return None


def _download_ticker_since(
    ticker: str,
    session: requests.Session,
    since_ts: int,
    interval: str = "1h",
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single ticker starting from a Unix timestamp.
    Used for incremental cache updates - fetches only the missing candles.

    Args:
        ticker:   Ticker symbol
        session:  requests.Session from _get_yahoo_session()
        since_ts: Unix timestamp (seconds) - fetch data from this point onward
        interval: Candle interval (default '1h')

    Returns a DataFrame with DatetimeIndex and OHLCV columns, or None on failure.
    """
    import math
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    now_ts = int(time.time())
    params = {
        "period1": str(since_ts),
        "period2": str(now_ts),
        "interval": interval,
        "includePrePost": "false",
    }
    crumb = getattr(session, "_crumb", "")
    if crumb:
        params["crumb"] = crumb

    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    try:
        result = data["chart"]["result"]
        if not result:
            return None
        result = result[0]
        timestamps = result.get("timestamp")
        if not timestamps:
            return None
        quote = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Open":   quote["open"],
            "High":   quote["high"],
            "Low":    quote["low"],
            "Close":  quote["close"],
            "Volume": quote["volume"],
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))

        df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df
    except (KeyError, TypeError, IndexError):
        return None


def _save_ticker_data(ticker: str, data: pd.DataFrame, cache_dir: str) -> Optional[Path]:
    """
    Save a single ticker's OHLCV DataFrame to FirstRateData-compatible .txt format.
    Returns the path to the saved file, or None if data is insufficient.
    """
    cache_path = Path(cache_dir) / "yfinance_1hour"
    cache_path.mkdir(parents=True, exist_ok=True)
    out_file = cache_path / f"{ticker.upper()}_1hour.txt"

    # Remove weekend rows
    data = data[data.index.dayofweek < 5]

    # Ensure correct column order
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in data.columns:
            return None

    data = data[required_cols].dropna()

    if len(data) < 500:
        return None

    # Write in FirstRateData format: no header, datetime,O,H,L,C,V
    with open(out_file, "w") as f:
        for ts, row in data.iterrows():
            dt_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{dt_str},{row['Open']:.6f},{row['High']:.6f},"
                    f"{row['Low']:.6f},{row['Close']:.6f},{int(row['Volume'])}\n")

    return out_file


def download_yfinance_ticker(
    ticker: str,
    cache_dir: str = "data/yfinance_cache",
    force: bool = False,
) -> Optional[Path]:
    """
    Download ~2 years of hourly OHLCV for a single ticker and save as a
    FirstRateData-compatible .txt file.

    File is saved to: {cache_dir}/yfinance_1hour/{TICKER}_1hour.txt
    Format: YYYY-MM-DD HH:MM:SS,Open,High,Low,Close,Volume (no header)

    Returns the path to the saved file, or None if download failed.
    """
    cache_path = Path(cache_dir) / "yfinance_1hour"
    cache_path.mkdir(parents=True, exist_ok=True)
    out_file = cache_path / f"{ticker.upper()}_1hour.txt"

    if out_file.exists() and not force:
        df = pd.read_csv(out_file, header=None, names=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        if len(df) > 1000:
            print(f"  [yahoo] {ticker}: cached ({len(df)} rows)")
            return out_file

    session = _get_yahoo_session()
    data = _download_ticker_raw(ticker, session)

    if data is None or data.empty:
        print(f"  [yahoo] {ticker}: no data returned")
        return None

    result = _save_ticker_data(ticker, data, cache_dir)
    if result is None:
        print(f"  [yahoo] {ticker}: insufficient data")
        return None

    rows = len(pd.read_csv(result, header=None))
    print(f"  [yahoo] {ticker}: downloaded {rows} hourly bars")
    return result


def download_yfinance_tickers(
    tickers: List[str],
    cache_dir: str = "data/yfinance_cache",
    force: bool = False,
    delay: float = 0.1,
    batch_size: int = 50,
) -> List[str]:
    """
    Download hourly data for multiple tickers using direct Yahoo Finance API.
    Returns list of tickers that were successfully downloaded.

    Downloads tickers individually (Yahoo's chart API is per-ticker) but
    reuses a single session with cookies/crumb for speed. No curl_cffi needed.

    Args:
        tickers:    List of ticker symbols
        cache_dir:  Directory to save .txt files
        force:      Re-download even if cached
        delay:      Seconds between requests (rate limiting)
        batch_size: (unused, kept for API compatibility)
    """
    cache_path = Path(cache_dir) / "yfinance_1hour"
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[yahoo] Downloading hourly data for {len(tickers)} tickers (direct API)...")

    # Check cache first
    successful = []
    to_download = []
    if not force:
        for ticker in tickers:
            out_file = cache_path / f"{ticker.upper()}_1hour.txt"
            if out_file.exists():
                try:
                    df = pd.read_csv(out_file, header=None,
                                     names=["Datetime", "Open", "High", "Low", "Close", "Volume"])
                    if len(df) > 1000:
                        successful.append(ticker)
                        continue
                except Exception:
                    pass
            to_download.append(ticker)
    else:
        to_download = list(tickers)

    if successful:
        print(f"  [yahoo] Reusing {len(successful)} cached tickers")

    if not to_download:
        print(f"[yahoo] All {len(successful)} tickers cached.\n")
        return successful

    # Create one session for all downloads (shares cookies/crumb)
    session = _get_yahoo_session()
    crumb_ok = bool(getattr(session, "_crumb", ""))
    print(f"  [yahoo] Session ready (crumb={'yes' if crumb_ok else 'no'})")

    failed = []
    for i, ticker in enumerate(to_download):
        data = _download_ticker_raw(ticker, session)

        if data is None or data.empty:
            failed.append(ticker)
            if len(failed) <= 5:
                print(f"    {ticker}: no data")
            continue

        result = _save_ticker_data(ticker, data, cache_dir)
        if result is not None:
            successful.append(ticker)
        else:
            failed.append(ticker)
            if len(failed) <= 5:
                print(f"    {ticker}: insufficient data (<500 rows)")

        # Progress every 50 tickers
        if (i + 1) % 50 == 0:
            ok_so_far = len([t for t in to_download[:i+1] if t in successful])
            print(f"  [yahoo] Progress: {i+1}/{len(to_download)} done, {ok_so_far} OK")

        # Rate limit
        if delay > 0:
            time.sleep(delay)

    # Retry failed tickers once with a fresh session
    if failed:
        print(f"  [yahoo] Retrying {len(failed)} failed tickers with fresh session...")
        session = _get_yahoo_session()
        retry_ok = 0
        for ticker in failed:
            data = _download_ticker_raw(ticker, session)
            if data is not None and not data.empty:
                result = _save_ticker_data(ticker, data, cache_dir)
                if result is not None:
                    successful.append(ticker)
                    retry_ok += 1
            if delay > 0:
                time.sleep(delay)
        print(f"  [yahoo] Retry recovered {retry_ok}/{len(failed)} tickers")

    print(f"[yahoo] Downloaded {len(successful)}/{len(tickers)} tickers successfully.\n")
    return successful


def update_cache_incremental(
    tickers: List[str],
    cache_dir: str = "data/yfinance_cache",
    delay: float = 0.05,
) -> Dict[str, int]:
    """
    Incrementally update cache files by downloading only missing candles.

    For each ticker:
      1. Reads the existing cache file to find the last timestamp
      2. Downloads only candles from (last_timestamp + 1h) to now
      3. Appends new rows to the cache file

    If no cache file exists for a ticker, does a full 2y download.

    Args:
        tickers:   List of ticker symbols to update
        cache_dir: Cache directory path
        delay:     Rate-limit pause between requests (seconds)

    Returns:
        Dict with counts: {"updated": N, "fresh": N, "failed": N, "skipped": N}
    """
    cache_path = Path(cache_dir) / "yfinance_1hour"
    cache_path.mkdir(parents=True, exist_ok=True)

    session = _get_yahoo_session()
    stats = {"updated": 0, "fresh": 0, "failed": 0, "skipped": 0}

    for ticker in tickers:
        out_file = cache_path / f"{ticker.upper()}_1hour.txt"

        try:
            if out_file.exists() and out_file.stat().st_size > 0:
                # Read last line to find the most recent timestamp
                existing = pd.read_csv(
                    out_file, header=None,
                    names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
                    parse_dates=["Datetime"],
                )
                if existing.empty:
                    raise ValueError("empty cache")

                last_dt = existing["Datetime"].iloc[-1]
                # Convert to Unix timestamp, add 1 hour so we don't re-fetch the last bar
                since_ts = int(last_dt.timestamp()) + 3600

                # If last bar is less than 1 hour old, nothing to fetch
                if (time.time() - since_ts) < 3600:
                    stats["skipped"] += 1
                    continue

                new_data = _download_ticker_since(ticker, session, since_ts)

                if new_data is None or new_data.empty:
                    stats["skipped"] += 1
                    continue

                # Filter: remove weekends, drop NaN rows
                new_data = new_data[new_data.index.dayofweek < 5]
                new_data = new_data[["Open", "High", "Low", "Close", "Volume"]].dropna()

                # Remove any rows that overlap with existing data
                new_data = new_data[new_data.index > last_dt]

                if new_data.empty:
                    stats["skipped"] += 1
                    continue

                # Append to existing file
                with open(out_file, "a") as f:
                    for ts, row in new_data.iterrows():
                        dt_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                        f.write(
                            f"{dt_str},{row['Open']:.6f},{row['High']:.6f},"
                            f"{row['Low']:.6f},{row['Close']:.6f},{int(row['Volume'])}\n"
                        )

                stats["updated"] += 1
            else:
                # No cache file - full download
                data = _download_ticker_raw(ticker, session)
                if data is not None and not data.empty:
                    result = _save_ticker_data(ticker, data, cache_dir)
                    if result is not None:
                        stats["fresh"] += 1
                    else:
                        stats["failed"] += 1
                else:
                    stats["failed"] += 1
        except Exception:
            stats["failed"] += 1

        if delay > 0:
            time.sleep(delay)

    return stats


def get_yfinance_data_path(cache_dir: str = "data/yfinance_cache") -> str:
    """
    Returns the path that should be passed as data_path to the pipeline.
    DataLoader expects files at: {data_path}/{folder}/{TICKER}_1hour.txt
    """
    return str(Path(cache_dir))


# ============================================================================
# VIX Data Support (for regime feature augmentation)
# ============================================================================

def _download_vix_data(
    cache_dir: str = "data/yfinance_cache",
    force: bool = False,
) -> Dict[str, Optional[Path]]:
    """
    Download VIX spot (^VIX) and 3-month (^VIX3M) futures data from Yahoo Finance.

    These are used to compute vix_term_structure feature in feature_engineer.py.
    VIX data is automatically merged with stock data when using
    download_yfinance_tickers_with_vix().

    Args:
        cache_dir: Cache directory for VIX files
        force: Force re-download even if cached

    Returns:
        Dict with paths: {"VIX": path, "VIX3M": path}
    """
    cache_path = Path(cache_dir) / "yfinance_1hour"
    cache_path.mkdir(parents=True, exist_ok=True)

    vix_symbols = {
        "^VIX": "VIX",
        "^VIX3M": "VIX3M"
    }

    results = {}
    session = _get_yahoo_session()

    print("\n[vix] Downloading VIX data for term structure calculation...")

    for yahoo_sym, clean_name in vix_symbols.items():
        out_file = cache_path / f"{clean_name}_1hour.txt"

        # Check cache
        if out_file.exists() and not force:
            try:
                df = pd.read_csv(out_file, header=None,
                                names=["Datetime", "Open", "High", "Low", "Close", "Volume"])
                if len(df) > 100:
                    results[clean_name] = out_file
                    print(f" [vix] {clean_name}: cached ({len(df)} rows)")
                    continue
            except Exception:
                pass

        # Download VIX
        data = _download_ticker_raw(yahoo_sym, session)

        if data is None or data.empty:
            print(f" [vix] {clean_name}: download failed")
            results[clean_name] = None
            continue

        # Save VIX data
        result = _save_ticker_data(clean_name, data, cache_dir)
        if result is not None:
            results[clean_name] = result
            rows = len(pd.read_csv(result, header=None))
            print(f" [vix] {clean_name}: downloaded {rows} hourly bars")
        else:
            results[clean_name] = None
            print(f" [vix] {clean_name}: insufficient data")

    return results


def _merge_vix_into_stock_df(
    stock_df: pd.DataFrame,
    vix_path: Path,
    prefix: str = "VIX"
) -> pd.DataFrame:
    """
    Merge VIX data columns into stock DataFrame.

    Args:
        stock_df: DataFrame with DatetimeIndex
        vix_path: Path to VIX_1hour.txt file
        prefix: Prefix for VIX columns (VIX_Open, VIX_Close, etc.)

    Returns:
        DataFrame with added VIX columns
    """
    if not vix_path.exists():
        return stock_df

    try:
        vix_df = pd.read_csv(
            vix_path, header=None,
            names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
            parse_dates=["Datetime"]
        )
        vix_df["Datetime"] = pd.to_datetime(vix_df["Datetime"])
        vix_df = vix_df.set_index("Datetime")

        # Rename columns with prefix
        vix_cols = {col: f"{prefix}_{col}" for col in vix_df.columns}
        vix_df = vix_df.rename(columns=vix_cols)

        # Forward fill VIX to match stock timestamps (VIX updates less frequently)
        vix_df = vix_df.reindex(stock_df.index, method="ffill")

        # Join with stock data
        merged = stock_df.join(vix_df, how="left")
        return merged

    except Exception as e:
        print(f" [vix] Warning: Could not merge VIX data: {e}")
        return stock_df


def download_yfinance_tickers_with_vix(
    tickers: List[str],
    cache_dir: str = "data/yfinance_cache",
    force: bool = False,
    delay: float = 0.1,
) -> List[str]:
    """
    Download tickers AND VIX data, merging VIX columns into each ticker file.

    This enables the vix_term_structure feature in regime detection.
    VIX data is downloaded once, then merged into each stock's .txt file.

    Args:
        tickers: List of stock tickers to download
        cache_dir: Cache directory
        force: Force re-download
        delay: Rate limiting delay

    Returns:
        List of successfully downloaded tickers (VIX not included in list)
    """
    # Step 1: Download VIX data first
    vix_paths = _download_vix_data(cache_dir, force)

    # Step 2: Download stock tickers using existing function
    successful = download_yfinance_tickers(tickers, cache_dir, force, delay)

    # Step 3: Merge VIX into stock files
    if vix_paths.get("VIX") and vix_paths.get("VIX3M"):
        print("[vix] Merging VIX data into stock files...")
        cache_path = Path(cache_dir) / "yfinance_1hour"

        for ticker in successful:
            ticker_file = cache_path / f"{ticker.upper()}_1hour.txt"
            if not ticker_file.exists():
                continue

            try:
                # Read stock data
                stock_df = pd.read_csv(
                    ticker_file, header=None,
                    names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
                    parse_dates=["Datetime"]
                )
                stock_df["Datetime"] = pd.to_datetime(stock_df["Datetime"])
                stock_df = stock_df.set_index("Datetime")

                # Merge VIX
                stock_df = _merge_vix_into_stock_df(stock_df, vix_paths["VIX"], "VIX")
                stock_df = _merge_vix_into_stock_df(stock_df, vix_paths["VIX3M"], "VIX3M")

                # Write back
                stock_df = stock_df.reset_index()
                with open(ticker_file, "w") as f:
                    for _, row in stock_df.iterrows():
                        dt_str = row["Datetime"].strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{dt_str},{row['Open']:.6f},{row['High']:.6f},"
                               f"{row['Low']:.6f},{row['Close']:.6f},{int(row['Volume'])},"
                               f"{row.get('VIX_Close', 0):.6f},{row.get('VIX3M_Close', 0):.6f}\n")

            except Exception as e:
                print(f" [vix] Warning: Could not merge VIX into {ticker}: {e}")
                continue

        print("[vix] VIX merge complete. VIX_Close and VIX3M_Close columns added.")
    else:
        print("[vix] VIX data unavailable, stock files will not have VIX columns")

    return successful
