"""
Configurable in-sample MCPT using standardised framework.
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
         n_permutations: int = 1000):
    """
    Run configurable in-sample MCPT.

    Args:
        data_file: Path to parquet data file
        strategy_name: Name of strategy to test
        start_year: Start year for data filtering
        end_year: End year for data filtering
        n_permutations: Number of permutations to test
    """
    print(f"Data File: {data_file}")
    print(f"Strategy: {strategy_name}")
    print(f"Permutations: {n_permutations}")

    try:
        strategy = get_strategy_by_name(strategy_name)
        runner = MCPTRunner(data_file, strategy)
        runner.load_data(start_year=start_year, end_year=end_year)
        print(f"Loaded data: {len(runner.data)} rows from {runner.data.index[0]} to {runner.data.index[-1]}")

        results = runner.run_insample_mcpt(n_permutations=n_permutations)

        runner.plot_results(results, f"In-Sample MCPT")
  
        print(f"\nResults:")
        print(f"Strategy: {strategy.name}")
        print(f"Data File: {data_file}")
        print(f"Time Period: {start_year}-{end_year}")
        print(f"Data Points: {len(runner.data)}")
        print(f"Optimised Parameters: {results['strategy_params']}")
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
        description='Run configurable in-sample MCPT testing',
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
    parser.add_argument('--permutations', '-p', type=int, default=1000,
                       help='Number of permutations to test')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.list_strategies:
        list_strategies()
        exit(0)

    results = main(
        data_file=args.data_file,
        strategy_name=args.strategy,
        start_year=args.start_year,
        end_year=args.end_year,
        n_permutations=args.permutations
    )