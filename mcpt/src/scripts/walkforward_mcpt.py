"""
Configurable walk-forward MCPT using standardised framework.
Supports command-line arguments for data file, strategy, and timeframe.
"""
import numpy as np
import argparse
from typing import Optional

try:
    from resources.utils.registry_config import get_strategy_by_name, list_strategies
    from ..core.mcpt_runner import MCPTRunner
except ImportError:
    from resources.utils.registry_config import get_strategy_by_name, list_strategies
    from core.mcpt_runner import MCPTRunner


def main(data_file: str,
         strategy_name: str,
         start_year: Optional[int] = None,
         end_year: Optional[int] = None,
         n_permutations: int = 200,
         train_lookback: Optional[int] = None,
         train_step: Optional[int] = None):
    """
    Run configurable walk-forward MCPT.

    Args:
        data_file: Path to parquet data file
        strategy_name: Name of strategy to test
        start_year: Start year for data filtering
        end_year: End year for data filtering
        n_permutations: Number of permutations to test
        train_lookback: Training window size in periods
        train_step: Retraining frequency in periods
    """
    try:
        strategy = get_strategy_by_name(strategy_name)
        runner = MCPTRunner(data_file, strategy)
        runner.load_data(start_year=start_year, end_year=end_year)
        print(f"Loaded data: {len(runner.data)} rows from {runner.data.index[0]} to {runner.data.index[-1]}")
        data_size = len(runner.data)
        if train_lookback is None:
            train_lookback = min(1000, max(500, data_size // 3))  # Default 5k, min 500, max 1/3 of data
        if train_step is None:
            train_step = max(30, train_lookback // 20)  # Default 1/20th of lookback, min 30

        print(f"Using training window: {train_lookback} periods, step: {train_step} periods")

        results = runner.run_walkforward_mcpt(
            n_permutations=n_permutations,
            train_lookback=train_lookback,
            train_step=train_step
        )

        runner.plot_results(results, f"Walk-Forward MCPT")


        print(f"\nResults:")
        print(f"Strategy: {strategy.name}")
        print(f"Data File: {data_file}")
        print(f"Time Period: {start_year}-{end_year}")
        print(f"Data Points: {len(runner.data)}")
        print(f"Training Window: {results['train_lookback']} periods")
        print(f"Retraining Step: {results['train_step']} periods")
        print(f"Real Profit Factor: {results['real_profit_factor']:.6f}")
        print(f"Mean Permuted PF: {np.mean(results['permuted_profit_factors']):.6f}")
        print(f"Std Permuted PF: {np.std(results['permuted_profit_factors']):.6f}")
        print(f"P-Value: {results['p_value']:.6f}")

        return results

    except FileNotFoundError:
        print(f"Data file '{data_file}' not found.")
        print("Please ensure the data file exists in the specified path.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run configurable walk-forward MCPT testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data-file', '-d', required=True,
                       help='Path to data file (CSV, Parquet, JSON, Excel)')
    parser.add_argument('--strategy', '-s', required=True,
                       help='Strategy name (registered in strategy registry)')
    parser.add_argument('--list-strategies', action='store_true',
                       help='List available strategies and exit')
    parser.add_argument('--start-year', type=int,
                       help='Start year for data filtering')
    parser.add_argument('--end-year', type=int,
                       help='End year for data filtering')
    parser.add_argument('--permutations', '-p', type=int, default=200,
                       help='Number of permutations to test')
    parser.add_argument('--train-lookback', type=int,
                       help='Training window size in periods (auto-adjusted if not specified)')
    parser.add_argument('--train-step', type=int,
                       help='Retraining frequency in periods (auto-adjusted if not specified)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Handle list strategies command
    if args.list_strategies:
        list_strategies()
        exit(0)

    results = main(
        data_file=args.data_file,
        strategy_name=args.strategy,
        start_year=args.start_year,
        end_year=args.end_year,
        n_permutations=args.permutations,
        train_lookback=args.train_lookback,
        train_step=args.train_step
    )