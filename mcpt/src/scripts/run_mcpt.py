"""
Unified MCPT runner.
"""
import argparse
import sys

try:
    from resources.utils.config_manager import ConfigManager, MCPTConfig, print_config
    from resources.utils.registry_config import get_strategy_by_name, list_strategies
    from . import insample_mcpt
    from . import walkforward_mcpt
except ImportError:
    from resources.utils.config_manager import ConfigManager, MCPTConfig, print_config
    from resources.utils.registry_config import get_strategy_by_name, list_strategies
    import src.scripts.insample_mcpt as insample_mcpt
    import src.scripts.walkforward_mcpt as walkforward_mcpt


def run_insample_mcpt(config: MCPTConfig):
    """Run in-sample MCPT using the standalone implementation."""
    print("In-Sample Testing:")
    print_config(config)

    return insample_mcpt.main(
        data_file=config.data_file,
        strategy_name=config.strategy_name,
        start_year=config.start_year,
        end_year=config.end_year,
        n_permutations=config.n_permutations
    )


def run_walkforward_mcpt(config: MCPTConfig):
    """Run walk-forward MCPT using the standalone implementation."""
    print("Walk-Forward Testing:")
    print_config(config)

    return walkforward_mcpt.main(
        data_file=config.data_file,
        strategy_name=config.strategy_name,
        start_year=config.start_year,
        end_year=config.end_year,
        n_permutations=config.n_permutations,
        train_lookback=config.train_lookback,
        train_step=config.train_step
    )


def main():
    """Main entry point for MCPT testing."""
    parser = argparse.ArgumentParser(
        description='Run MCPT testing with configurable parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('test_type', nargs='?', choices=['insample', 'walkforward', 'both'],
                       help='Type of MCPT test to run')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to configuration file')
    parser.add_argument('--list-strategies', action='store_true',
                       help='List all registered strategies and exit')
    parser.add_argument('--data-file', '-d', type=str,
                       help='Path to data file (CSV, Parquet, JSON, Excel)')
    parser.add_argument('--strategy', '-s', type=str,
                       help='Strategy name (use --list-strategies to see available)')
    parser.add_argument('--start-year', type=int,
                       help='Start year for data filtering')
    parser.add_argument('--end-year', type=int,
                       help='End year for data filtering')
    parser.add_argument('--permutations', '-p', type=int,
                       help='Number of permutations to test')
    parser.add_argument('--train-lookback', type=int, 
                       help='Training window size in periods')   # Walk-forward specific
    parser.add_argument('--train-step', type=int,
                       help='Retraining frequency in periods')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting results')

    args = parser.parse_args()

    if args.list_strategies:
        list_strategies()
        return

    if not args.test_type:
        parser.error("test_type is required")

    if not args.config:
        if not args.data_file:
            parser.error("--data-file is required when not using --config")
        if not args.strategy:
            parser.error("--strategy is required when not using --config")

    if args.config:
        config = ConfigManager.load_config(args.config)
        # Update with any command-line overrides
        config = ConfigManager.update_config_from_args(config, args)
    else:
        # Create custom config from command-line arguments
        config = ConfigManager.create_config_from_args(
            data_file=args.data_file,
            strategy_name=args.strategy,
            start_year=args.start_year,
            end_year=args.end_year,
            n_permutations=args.permutations,
            train_lookback=args.train_lookback,
            train_step=args.train_step
        )


    if args.no_plot:
        config.plot_results = False

    if args.test_type == 'insample':
        results = run_insample_mcpt(config)
        if results:
            print(f"\nTest completed successfully. P-Value: {results['p_value']:.6f}")
        else:
            print("Test failed.")
            sys.exit(1)

    elif args.test_type == 'walkforward':
        results = run_walkforward_mcpt(config)
        if results:
            print(f"\nTest completed successfully. P-Value: {results['p_value']:.6f}")
        else:
            print("Test failed.")
            sys.exit(1)

    elif args.test_type == 'both':
        print("="*60)
        print("RUNNING BOTH IN-SAMPLE AND WALK-FORWARD MCPT TESTS")
        print("="*60)

        # Run in-sample first
        print("\n" + "="*20 + " PHASE 1: IN-SAMPLE MCPT " + "="*20)
        insample_results = run_insample_mcpt(config)

        if not insample_results:
            print("In-sample test failed. Aborting.")
            sys.exit(1)

        # Run walk-forward second
        print("\n" + "="*20 + " PHASE 2: WALK-FORWARD MCPT " + "="*20)
        walkforward_results = run_walkforward_mcpt(config)

        if not walkforward_results:
            print("Walk-forward test failed.")
            sys.exit(1)

        # Summary
        print("\n" + "="*25 + " SUMMARY " + "="*25)
        print(f"Strategy: {config.strategy_name}")
        print(f"Data File: {config.data_file}")
        print(f"In-Sample P-Value: {insample_results['p_value']:.6f}")
        print(f"Walk-Forward P-Value: {walkforward_results['p_value']:.6f}")
        print("="*60)

    else:
        print(f"Unknown test type: {args.test_type}")
        sys.exit(1)


if __name__ == '__main__':
    main()