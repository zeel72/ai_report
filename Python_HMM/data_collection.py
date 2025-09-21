"""
Data Collection and Preprocessing Module
Lab Assignment 5: Gaussian Hidden Markov Models for Financial Time Series Analysis
Course: CS307 - Artificial Intelligence
Week 5
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialDataCollector:
    """
    Class to handle financial data collection and preprocessing from Yahoo Finance
    """
    
    def __init__(self):
        """Initialize the data collector"""
        self.data = None
        self.returns = None
        self.ticker = None
        
    def download_data(self, ticker="AAPL", period="10y", start_date=None, end_date=None):
        """
        Download historical stock price data from Yahoo Finance
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA', '^GSPC')
            period (str): Period to download ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            start_date (str): Start date in 'YYYY-MM-DD' format (optional)
            end_date (str): End date in 'YYYY-MM-DD' format (optional)
            
        Returns:
            pd.DataFrame: Historical stock price data
        """
        try:
            print(f"Downloading {ticker} data...")
            self.ticker = ticker
            
            # Create yfinance ticker object
            stock = yf.Ticker(ticker)
            
            # Download data
            if start_date and end_date:
                self.data = stock.history(start=start_date, end=end_date)
            else:
                self.data = stock.history(period=period)
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
                
            print(f"Successfully downloaded {len(self.data)} data points")
            print(f"Date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
            
            return self.data
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
    
    def preprocess_data(self, price_column='Close'):
        """
        Preprocess the data by calculating returns and handling missing values
        
        Args:
            price_column (str): Column to use for price data
            
        Returns:
            pd.Series: Daily returns
        """
        if self.data is None:
            raise ValueError("No data available. Please download data first.")
        
        print("Preprocessing data...")
        
        # Check available columns and use appropriate price column
        if price_column not in self.data.columns:
            if 'Close' in self.data.columns:
                price_column = 'Close'
                print(f"Using 'Close' column instead of '{price_column}'")
            elif 'Adj Close' in self.data.columns:
                price_column = 'Adj Close'
                print(f"Using 'Adj Close' column instead of original request")
            else:
                print(f"Available columns: {list(self.data.columns)}")
                raise ValueError(f"Price column '{price_column}' not found in data")
        
        # Extract closing prices
        prices = self.data[price_column].copy()
        
        # Handle missing values
        print(f"Missing values before cleaning: {prices.isnull().sum()}")
        
        # Forward fill missing values
        prices = prices.fillna(method='ffill')
        
        # Backward fill any remaining missing values
        prices = prices.fillna(method='bfill')
        
        # Calculate daily returns
        self.returns = prices.pct_change().dropna()
        
        print(f"Calculated returns for {len(self.returns)} trading days")
        print(f"Returns statistics:")
        print(self.returns.describe())
        
        # Check for outliers (returns > 3 standard deviations)
        outliers = np.abs(self.returns) > 3 * self.returns.std()
        print(f"Outliers detected: {outliers.sum()}")
        
        return self.returns
    
    def clean_data(self, remove_outliers=True, outlier_threshold=3):
        """
        Clean the returns data by removing outliers and handling extreme values
        
        Args:
            remove_outliers (bool): Whether to remove outliers
            outlier_threshold (float): Number of standard deviations for outlier detection
            
        Returns:
            pd.Series: Cleaned returns
        """
        if self.returns is None:
            raise ValueError("No returns data available. Please preprocess data first.")
        
        cleaned_returns = self.returns.copy()
        
        if remove_outliers:
            # Calculate outlier bounds
            mean_return = cleaned_returns.mean()
            std_return = cleaned_returns.std()
            lower_bound = mean_return - outlier_threshold * std_return
            upper_bound = mean_return + outlier_threshold * std_return
            
            # Count outliers
            outliers = (cleaned_returns < lower_bound) | (cleaned_returns > upper_bound)
            n_outliers = outliers.sum()
            
            print(f"Removing {n_outliers} outliers ({n_outliers/len(cleaned_returns)*100:.2f}%)")
            
            # Cap outliers instead of removing them to preserve time series structure
            cleaned_returns = np.clip(cleaned_returns, lower_bound, upper_bound)
        
        self.returns = cleaned_returns
        
        print("Data cleaning completed")
        print(f"Final returns statistics:")
        print(self.returns.describe())
        
        return self.returns
    
    def plot_data_overview(self, save_path=None):
        """
        Create overview plots of the stock data and returns
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.data is None or self.returns is None:
            raise ValueError("No data available for plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.ticker} - Data Overview', fontsize=16, fontweight='bold')
        
        # Stock price plot - use available price column
        price_column = 'Close' if 'Close' in self.data.columns else 'Adj Close'
        axes[0, 0].plot(self.data.index, self.data[price_column], linewidth=1, color='blue')
        axes[0, 0].set_title(f'{price_column} Price')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volume plot
        axes[0, 1].plot(self.data.index, self.data['Volume'], linewidth=1, color='orange')
        axes[0, 1].set_title('Trading Volume')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns time series
        axes[1, 0].plot(self.returns.index, self.returns, linewidth=0.8, color='red', alpha=0.7)
        axes[1, 0].set_title('Daily Returns')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Return')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Returns histogram
        axes[1, 1].hist(self.returns, bins=50, density=True, alpha=0.7, color='green')
        axes[1, 1].set_title('Returns Distribution')
        axes[1, 1].set_xlabel('Return')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics to histogram
        mean_ret = self.returns.mean()
        std_ret = self.returns.std()
        axes[1, 1].axvline(mean_ret, color='red', linestyle='--', 
                          label=f'Mean: {mean_ret:.4f}')
        axes[1, 1].axvline(mean_ret + std_ret, color='orange', linestyle='--', 
                          label=f'±1σ: {std_ret:.4f}')
        axes[1, 1].axvline(mean_ret - std_ret, color='orange', linestyle='--')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_multiple_assets(self, tickers=['AAPL', 'GOOGL', 'MSFT', '^GSPC'], period="10y"):
        """
        Download data for multiple assets for comparison
        
        Args:
            tickers (list): List of ticker symbols
            period (str): Period to download
            
        Returns:
            dict: Dictionary of DataFrames for each ticker
        """
        assets_data = {}
        assets_returns = {}
        
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            try:
                # Create new instance for each ticker
                collector = FinancialDataCollector()
                data = collector.download_data(ticker, period)
                
                if data is not None:
                    returns = collector.preprocess_data()
                    cleaned_returns = collector.clean_data()
                    
                    assets_data[ticker] = data
                    assets_returns[ticker] = cleaned_returns
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        return assets_data, assets_returns
    
    def save_data(self, data_path="../data/", include_raw=True):
        """
        Save the processed data to files
        
        Args:
            data_path (str): Directory to save data
            include_raw (bool): Whether to save raw data as well
        """
        if self.data is None or self.returns is None:
            raise ValueError("No data to save")
        
        import os
        os.makedirs(data_path, exist_ok=True)
        
        # Save returns data (main data for HMM)
        returns_file = f"{data_path}/{self.ticker}_returns.csv"
        self.returns.to_csv(returns_file)
        print(f"Returns data saved to {returns_file}")
        
        if include_raw:
            # Save raw price data
            raw_file = f"{data_path}/{self.ticker}_raw_data.csv"
            self.data.to_csv(raw_file)
            print(f"Raw data saved to {raw_file}")


def main():
    """
    Main function to demonstrate data collection functionality
    """
    print("Financial Data Collection and Preprocessing")
    print("=" * 50)
    
    # Initialize data collector
    collector = FinancialDataCollector()
    
    # Download data for Apple (AAPL) for the last 10 years
    data = collector.download_data("AAPL", period="10y")
    
    if data is not None:
        # Preprocess data
        returns = collector.preprocess_data()
        
        # Clean data
        cleaned_returns = collector.clean_data()
        
        # Create overview plots
        collector.plot_data_overview("../results/data_overview.png")
        
        # Save processed data
        collector.save_data()
        
        print("\nData collection and preprocessing completed successfully!")
        
        # Optional: Get data for multiple assets
        print("\nDownloading data for multiple assets...")
        assets_data, assets_returns = collector.get_multiple_assets(
            tickers=['AAPL', 'TSLA', '^GSPC', 'GOOGL'], 
            period="5y"
        )
        
        print(f"Successfully processed {len(assets_returns)} assets")


if __name__ == "__main__":
    main()