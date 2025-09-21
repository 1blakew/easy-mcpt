"""
Configuration management for MCPT testing framework.
Supports JSON configuration files for easy parameter management.
"""
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class MCPTConfig:
    """Configuration for MCPT testing."""
    data_file: str  # Required - no default, user must specify
    strategy_name: str  # Required - no default, user must specify
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    n_permutations: int = 1000

    # Walk-forward parameters
    train_lookback: int = 24 * 365 * 4
    train_step: int = 24 * 30

    # Output settings
    save_results: bool = True
    plot_results: bool = True
    output_dir: str = 'results'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPTConfig':
        """Create from dictionary."""
        return cls(**data)


class ConfigManager:
    """Manages MCPT configuration files."""

    @staticmethod
    def load_config(config_file: str) -> MCPTConfig:
        """
        Load configuration from JSON file.

        Args:
            config_file: Path to configuration file

        Returns:
            MCPTConfig instance
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found. Please create one or specify data_file and strategy_name.")

        try:
            with open(config_file, 'r') as f:
                data = json.load(f)

            # Validate required fields
            if 'data_file' not in data:
                raise ValueError("Config file must specify 'data_file'")
            if 'strategy_name' not in data:
                raise ValueError("Config file must specify 'strategy_name'")

            # Validate and create config
            config = MCPTConfig.from_dict(data)
            print(f"Loaded configuration from: {config_file}")
            return config

        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Error loading config file '{config_file}': {e}")

    @staticmethod
    def save_config(config: MCPTConfig, config_file: str):
        """
        Save configuration to JSON file.

        Args:
            config: MCPTConfig instance
            config_file: Path to save configuration
        """
        try:
            config_dir = os.path.dirname(config_file)
            if config_dir:  # Only create directory if path has a directory part
                os.makedirs(config_dir, exist_ok=True)

            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)

            print(f"Configuration saved to: {config_file}")

        except Exception as e:
            print(f"Error saving config file '{config_file}': {e}")

    @staticmethod
    def create_config_from_args(data_file: str, strategy_name: str, **kwargs) -> MCPTConfig:
        """
        Create configuration from command-line arguments.

        Args:
            data_file: Path to data file
            strategy_name: Name of strategy
            **kwargs: Additional configuration parameters

        Returns:
            MCPTConfig instance
        """
        return MCPTConfig(
            data_file=data_file,
            strategy_name=strategy_name,
            start_year=kwargs.get('start_year'),
            end_year=kwargs.get('end_year'),
            n_permutations=kwargs.get('n_permutations', 1000),
            train_lookback=kwargs.get('train_lookback', 24*365*4),
            train_step=kwargs.get('train_step', 24*30),
            save_results=kwargs.get('save_results', True),
            plot_results=kwargs.get('plot_results', True),
            output_dir=kwargs.get('output_dir', 'results')
        )

    @staticmethod
    def update_config_from_args(config: MCPTConfig, args) -> MCPTConfig:
        """
        Update configuration with command-line arguments.

        Args:
            config: Base configuration
            args: Parsed command-line arguments

        Returns:
            Updated configuration
        """

        if hasattr(args, 'data_file') and args.data_file:
            config.data_file = args.data_file
        if hasattr(args, 'strategy') and args.strategy:
            config.strategy_name = args.strategy
        if hasattr(args, 'start_year') and args.start_year is not None:
            config.start_year = args.start_year
        if hasattr(args, 'end_year') and args.end_year is not None:
            config.end_year = args.end_year
        if hasattr(args, 'permutations') and args.permutations is not None:
            config.n_permutations = args.permutations
        if hasattr(args, 'train_lookback') and args.train_lookback is not None:
            config.train_lookback = args.train_lookback
        if hasattr(args, 'train_step') and args.train_step is not None:
            config.train_step = args.train_step

        return config


def print_config(config: MCPTConfig):
    print("MCPT Config")
    print(f"Data File: {config.data_file}")
    print(f"Strategy: {config.strategy_name}")
    print(f"Time Period: {config.start_year}-{config.end_year}")
    print(f"Permutations: {config.n_permutations}")
    print(f"Training Window: {config.train_lookback} periods")
    print(f"Retraining Step: {config.train_step} periods")
    print(f"Save Results: {config.save_results}")
    print(f"Plot Results: {config.plot_results}")
    print(f"Output Directory: {config.output_dir}")
    print("=" * 27)


if __name__ == '__main__':
    print("MCPT Configuration Manager")
    print("Use this to create configurations from command-line arguments.")
    print("Example usage in scripts:")
    print("  - config = ConfigManager.create_config_from_args('data/myfile.csv', 'my_strategy')")
    print("  - config = ConfigManager.load_config('my_config.json')")