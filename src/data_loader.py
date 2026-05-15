import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import os

class DataLoader:
    """
    Loads and validates OHLCV stock data from FirstRateData .txt files.
    
    FirstRateData format:
    YYYY-MM-DD HH:MM:SS,Open,High,Low,Close,Volume
    
    Attributes:
        source_path: Path to parent data directory containing letter folders
        stock_symbol: Stock ticker symbol (e.g., 'AAPL')
    """
    
    def __init__(self, source_path: str, stock_symbol: str):
        """
        Initialize DataLoader.
        
        Args:
            source_path: Path to data/raw directory
            stock_symbol: Stock ticker to load
        """
        self.source_path = Path(source_path)
        self.stock_symbol = stock_symbol.upper()
        self.data = None
        self.file_path = None
        
    def _find_data_file(self) -> Path:
        """
        Find the data file for the given stock symbol.
        FirstRateData organizes files in folders like:
        - us3000_tickers_A-B_1hour/
        - us3000_tickers_C-D_1hour/
        - etc.
        
        Returns path to the stock data file
        Raises FileNotFoundError if file cannot be found
        """
        # Get first two letters of stock symbol to determine folder
        first_letter = self.stock_symbol[0]
        second_letter = self.stock_symbol[1] if len(self.stock_symbol) > 1 else self.stock_symbol[0]
        
        # Search for matching folder
        # e.g., for "AAPL", look for "A-B" folder
        for folder in self.source_path.iterdir():
            if not folder.is_dir():
                continue
                
            # Check if folder name matches pattern like "us3000_tickers_A-B_1hour"
            if first_letter in folder.name and second_letter in folder.name:
                file_path = folder / f"{self.stock_symbol}_1hour.txt"
                if file_path.exists():
                    return file_path
        
        # If not found, search all subdirectories (fallback)
        for folder in self.source_path.rglob('*'):
            if folder.is_file() and folder.name == f"{self.stock_symbol}_1hour.txt":
                return folder
        
        raise FileNotFoundError(
            f"Data file not found for {self.stock_symbol}. "
            f"Expected file: {self.stock_symbol}_1hour.txt in data/raw/ subdirectories"
        )
    
    def load_data(self) -> pd.DataFrame:
        """
        Load OHLCV data from FirstRateData .txt file.

        Uses an on-disk parquet cache at <source_path>.parent/cache/raw_v1/
        keyed by raw-file mtime: subsequent calls for the same ticker skip
        CSV parsing entirely. Cache writes are best-effort and never affect
        correctness — same DataFrame is produced either way.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index: Datetime
        """
        self.file_path = self._find_data_file()
        raw_mtime = self.file_path.stat().st_mtime

        cache_root = self.source_path.parent / "cache" / "raw_v1"
        cache_path = cache_root / f"{self.stock_symbol}_1hour.parquet"

        df = None
        if cache_path.exists() and cache_path.stat().st_mtime >= raw_mtime:
            try:
                df = pd.read_parquet(cache_path)
            except Exception:
                df = None  # corrupt cache → fall back to CSV

        if df is None:
            print(f"Loading data from: {self.file_path}")
            try:
                df = pd.read_csv(
                    self.file_path,
                    header=None,
                    names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                    parse_dates=['Datetime'],
                    index_col='Datetime'
                )
            except Exception as e:
                raise ValueError(f"Error reading {self.file_path}: {str(e)}")

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

            try:
                cache_root.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)
            except Exception as e:
                print(f"  (raw cache write failed for {self.stock_symbol}: {e})")

        # Tag the DataFrame so downstream caches (FeatureEngineer) can key on it.
        try:
            df.attrs['_dl_ticker']     = self.stock_symbol
            df.attrs['_dl_raw_mtime']  = raw_mtime
            df.attrs['_dl_cache_root'] = str(self.source_path.parent / "cache")
        except Exception:
            pass

        self.data = df
        print(f"Loaded {len(df)} records for {self.stock_symbol}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")

        return df
    
    def validate_data(self) -> bool:
        """
        Validate data for:
        - Missing values
        - Negative prices
        - Timestamp consistency
        - High < Low violations
        
        Returns:
            True if valid, raises Exception otherwise
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self.data
        
        # Check for negative prices
        if (df[['Open', 'High', 'Low', 'Close']] < 0).any().any():
            raise ValueError("Negative prices detected in data")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values detected")
            print(f"Columns with missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
            # Forward fill for minor gaps
            self.data = df.fillna(method='ffill')
            print("Forward-filled missing values")
        
        # Check timestamp consistency
        if not df.index.is_monotonic_increasing:
            print("Warning: Data not in chronological order. Sorting...")
            self.data = df.sort_index()
            print("Data sorted by timestamp")
        
        # Check for OHLC logical consistency (High >= Open, High >= Close, Low <= Open, Low <= Close)
        invalid_high = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
        invalid_low = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
        
        if invalid_high > 0 or invalid_low > 0:
            print(f"Warning: {invalid_high} rows with High < Open/Close, {invalid_low} rows with Low > Open/Close")
        
        print("Data validation passed")
        return True
    
    def get_date_range(self) -> Tuple[str, str]:
        """
        Get available date range for loaded data.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        return (
            self.data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
            self.data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        )
    def get_info(self) -> dict:
        """
        Get summary statistics about loaded data.
        
        Returns:
            Dictionary with data info
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        return {
            'symbol': self.stock_symbol,
            'records': len(self.data),
            'date_range': self.get_date_range(),
            'price_range': f"${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}",
            'avg_volume': f"{self.data['Volume'].mean():,.0f}",
            'missing_values': self.data.isnull().sum().sum()
        }
    
    @staticmethod
    def list_available_stocks(source_path: str) -> List[str]:
        """
        List all available stocks in the data directory.
        
        Args:
            source_path: Path to data/raw directory
            
        Returns:
            List of available stock symbols
        """
        source_path = Path(source_path)
        stocks = []
        
        for file in source_path.rglob('*_1hour.txt'):
            symbol = file.stem.replace('_1hour', '')
            stocks.append(symbol)
        
        return sorted(list(set(stocks)))


# Example usage (for testing)
if __name__ == "__main__":
    # List all available stocks
    available_stocks = DataLoader.list_available_stocks("data/raw")
    print(f"\n Available stocks ({len(available_stocks)}): {', '.join(available_stocks[:10])}...")
    
    # Load a specific stock
    try:
        loader = DataLoader("data/raw", "AAL")
        df = loader.load_data()
        loader.validate_data()
        
        print("\n" + "="*50)
        print("Data Summary:")
        print("="*50)
        for key, value in loader.get_info().items():
            print(f"{key:.<20} {value}")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nLast 5 rows:")
        print(df.tail())
        
    except Exception as e:
        print(f"Error: {e}")
