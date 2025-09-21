"""
Standardised MCPT testing framework.
Provides easy-to-use interface for testing any strategy with Monte Carlo Permutation Tests.
"""
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
# from tqdm import tqdm
from .strategy_base import TradingStrategy, MCPTTester

class MCPTRunner:
    """High-level interface for running MCPT tests on trading strategies."""

    def __init__(self, data_path: str, strategy: TradingStrategy):
        """
        Initialise MCPT runner.

        Args:
            data_path: Path to parquet file with OHLC data
            strategy: Trading strategy to test
        """
        self.data_path = data_path
        self.strategy = strategy
        self.tester = MCPTTester(strategy)
        self.data = None

    def _load_data_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file with auto-format detection.

        Args:
            file_path: Path to data file

        Returns:
            DataFrame with OHLC data
        """

        file_ext = file_path.lower().split('.')[-1]

        try:
            if file_ext == 'csv':
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif file_ext in ['pq', 'parquet']:
                df = pd.read_parquet(file_path)
            elif file_ext == 'json':
                df = pd.read_json(file_path, orient='index')
                df.index = pd.to_datetime(df.index)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, index_col=0, parse_dates=True)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            required_cols = ['open', 'high', 'low', 'close']
            df_cols_lower = [col.lower() for col in df.columns]
            missing_cols = [col for col in required_cols if col not in df_cols_lower]
            if missing_cols:
                raise ValueError(f"Data must contain columns: {required_cols}. Missing: {missing_cols}")
            
            col_mapping = {col.lower(): col for col in df.columns}
            rename_dict = {col_mapping[req_col]: req_col for req_col in required_cols if req_col in df_cols_lower}
            df = df.rename(columns=rename_dict)

            self.data = df
            return df

        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {e}")

    def load_data(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Load and filter data from various file formats.

        Args:
            start_year: Start year for filtering (inclusive)
            end_year: End year for filtering (exclusive)

        Returns:
            Filtered DataFrame
        """
        df = self._load_data_file(self.data_path)

        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df.set_index('date', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)

        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index = df.index.astype("datetime64[s]")

        if start_year is not None or end_year is not None:
            mask = pd.Series(True, index=df.index)
            if start_year is not None:
                mask &= (df.index.year >= start_year)
            if end_year is not None:
                mask &= (df.index.year < end_year)
            df = df[mask]

        self.data = df
        return df

    def run_insample_mcpt(self, n_permutations: int = 1000, **strategy_params) -> Dict[str, Any]:
        """
        Run in-sample MCPT with parameter optimisation.

        Args:
            n_permutations: Number of permutations to test
            **strategy_params: Fixed strategy parameters (if any)

        Returns:
            Dictionary with test results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(f"Running in-sample MCPT for {self.strategy.name}")

        # Optimise if no parameters provided
        if not strategy_params:
            print("Optimising strategy parameters...")
            opt_result = self.strategy.optimise(self.data)
            strategy_params = opt_result.best_params
            print(f"Optimised parameters: {strategy_params}")
            print(f"In-sample profit factor: {opt_result.best_score:.4f}")

        # Run MCPT
        print(f"Testing {n_permutations} permutations...")
        results = self.tester.run_insample_test(
            self.data, n_permutations, **strategy_params
        )

        print(f"In-sample MCPT P-Value: {results['p_value']:.4f}")
        return results

    def run_walkforward_mcpt(self, n_permutations: int = 200, train_lookback: int = 24*365*4,
                           train_step: int = 24*30) -> Dict[str, Any]:
        """
        Run walk-forward MCPT.

        Args:
            n_permutations: Number of permutations to test
            train_lookback: Training window size
            train_step: Retraining frequency

        Returns:
            Dictionary with test results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(f"Running walk-forward MCPT for {self.strategy.name}")
        print(f"Training window: {train_lookback} periods, Step: {train_step} periods")

        # Run MCPT
        print(f"Testing {n_permutations} permutations...")
        results = self.tester.run_walkforward_test(
            self.data, n_permutations, train_lookback, train_step
        )

        print(f"Walk-forward MCPT P-Value: {results['p_value']:.4f}")
        return results

    def plot_results(self, results: Dict[str, Any], test_type: str = "MCPT"):
        """
        Plot MCPT results with both profit factor histogram and cumulative returns.

        Args:
            results: Results from MCPT test
            test_type: Type of test for plot title
        """
        plt.style.use('dark_background')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Profit Factor Histogram
        pd.Series(results['permuted_profit_factors']).hist(
            ax=ax1, color='blue', label='Permutations', alpha=0.7, bins=50
        )
        ax1.axvline(
            results['real_profit_factor'],
            color='red',
            label=f'Real (PF: {results["real_profit_factor"]:.3f})',
            linewidth=2
        )
        ax1.set_xlabel("Profit Factor")
        ax1.set_title(f'Profit Factor Distribution - P-Value: {results["p_value"]:.4f}')
        ax1.grid(False)
        ax1.legend()

        # Cumulative Returns Time Series
        if 'real_cum_returns' in results and 'permuted_cum_returns' in results:
            # Plot permutation curves
            perm_sample = results['permuted_cum_returns']
            for i, perm_cum_ret in enumerate(perm_sample):
                ax2.plot(perm_cum_ret.index, perm_cum_ret.values,
                        color='blue', alpha=0.3, linewidth=0.5)

            # Plot real cumulative returns
            real_cum_ret = results['real_cum_returns']
            ax2.plot(real_cum_ret.index, real_cum_ret.values,
                    color='red', linewidth=2, label='Real Strategy')

            ax2.set_xlabel("Time")
            ax2.set_ylabel("Cumulative Log Returns")
            ax2.set_title("Cumulative Returns: Real vs Permutations")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        plt.suptitle(f'{self.strategy.name} {test_type}', fontsize=16)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    """Demo usage of MCPTRunner framework."""
    print("MCPTRunner Framework")
    print("Use this with the strategy registry and run_mcpt.py for testing.")
    print("Example:")
    print("  from strategy_registry import get_strategy_by_name")
    print("  strategy = get_strategy_by_name('donchian')")
    print("  runner = MCPTRunner('data.csv', strategy)")
    print("  runner.load_data()")
    print("  results = runner.run_insample_mcpt()")