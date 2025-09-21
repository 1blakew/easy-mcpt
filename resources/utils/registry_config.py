"""
Strategy registry system for mapping strategy names to implementation locations.
Allows users to define custom strategies and import paths.
"""
import json
import os
import importlib
from typing import Dict, Type
try:
    from mcpt.src.core.strategy_base import TradingStrategy
except ImportError:
    from core.strategy_base import TradingStrategy # type: ignore


class StrategyRegistry:
    """Registry for managing strategy name-to-implementation mappings."""

    def __init__(self, registry_file: str = 'resources/registry/strats.reg.json'):
        self.registry_file = registry_file
        self.strategies: Dict[str, Dict[str, str]] = {}
        self.loaded_strategies: Dict[str, Type[TradingStrategy]] = {}
        self.load_registry()

    def load_registry(self):
        """Load strategy mappings from registry file."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    self.strategies = json.load(f)
                # Suppress loading message in subprocesses
                if os.getenv('MCPT_SUBPROCESS') != '1':
                    print(f"Loaded {len(self.strategies)} strategies from {self.registry_file}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading registry: {e}")
                self.create_default_registry()
        else:
            self.create_default_registry()

    def save_registry(self):
        """Save current strategy mappings to registry file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.strategies, f, indent=2)
            print(f"Registry saved to {self.registry_file}")
        except Exception as e:
            print(f"Error saving registry: {e}")

    def create_default_registry(self):
        """Create default strategy registry."""
        self.strategies = {
            "example: donchian": {
                "module": "donchian_strategy",
                "class": "DonchianStrategy",
                "description": "Donchian Breakout Strategy"
            }
        }
        self.save_registry()
        print("Created default strategy registry")

    def register_strategy(self, name: str, module_path: str, class_name: str, description: str = ""):
        """
        Register a new strategy.

        Args:
            name: Strategy identifier name
            module_path: Python import path (e.g., 'strategies.moving_average')
            class_name: Class name in the module (e.g., 'MovingAverageStrategy')
            description: Human-readable description
        """
        self.strategies[name] = {
            "module": module_path,
            "class": class_name,
            "description": description or f"{class_name} strategy"
        }
        self.save_registry()
        print(f"Registered strategy '{name}' -> {module_path}.{class_name}")

    def get_strategy(self, name: str) -> TradingStrategy:
        """
        Load and return strategy instance by name.

        Args:
            name: Strategy name from registry (case-insensitive)

        Returns:
            Strategy instance
        """
        # Convert to lowercase for case-insensitive lookup
        name_lower = name.lower()

        # Check loaded strategies (also case-insensitive)
        for loaded_name in self.loaded_strategies:
            if loaded_name.lower() == name_lower:
                return self.loaded_strategies[loaded_name]()

        # Find matching strategy name (case-insensitive)
        actual_name = None
        for strategy_name in self.strategies:
            if strategy_name.lower() == name_lower:
                actual_name = strategy_name
                break

        if actual_name is None:
            available = ', '.join(self.strategies.keys())
            raise ValueError(f"Strategy '{name}' not found. Available: {available}")

        strategy_info = self.strategies[actual_name]

        try:
            # Import the module
            module = importlib.import_module(strategy_info["module"])

            # Get the class
            strategy_class = getattr(module, strategy_info["class"])

            # Verify it's a valid strategy
            if not issubclass(strategy_class, TradingStrategy):
                raise ValueError(f"Class {strategy_info['class']} is not a TradingStrategy")

            # Cache the class using actual name
            self.loaded_strategies[actual_name] = strategy_class

            # Return instance
            return strategy_class()

        except ImportError as e:
            raise ImportError(f"Cannot import {strategy_info['module']}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Class {strategy_info['class']} not found in {strategy_info['module']}: {e}")

    def list_strategies(self) -> Dict[str, str]:
        """List all registered strategies with descriptions."""
        return {name.upper(): info["description"] for name, info in self.strategies.items()}

    def remove_strategy(self, name: str):
        """Remove a strategy from the registry."""
        if name in self.strategies:
            del self.strategies[name]
            if name in self.loaded_strategies:
                del self.loaded_strategies[name]
            self.save_registry()
            print(f"Removed strategy '{name}'")
        else:
            print(f"Strategy '{name}' not found in registry")


# Global registry instance
registry = StrategyRegistry()


def get_strategy_by_name(name: str) -> TradingStrategy:
    """Convenience function to get strategy by name."""
    return registry.get_strategy(name)


def register_strategy(name: str, module_path: str, class_name: str, description: str = ""):
    """Convenience function to register a strategy."""
    registry.register_strategy(name, module_path, class_name, description)


def list_strategies():
    """List all available strategies."""
    strategies = registry.list_strategies()
    print("Available Strategies:")
    for name, desc in strategies.items():
        print(f"  {name}: {desc}")
    return strategies


if __name__ == '__main__':
    """Command-line interface for managing strategy registry."""
    import argparse

    parser = argparse.ArgumentParser(description='Manage strategy registry')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List strategies
    list_parser = subparsers.add_parser('list', help='List all strategies')

    # Register strategy
    register_parser = subparsers.add_parser('register', help='Register new strategy')
    register_parser.add_argument('name', help='Strategy name')
    register_parser.add_argument('module', help='Module path (e.g., strategies.moving_average)')
    register_parser.add_argument('class_name', help='Class name (e.g., MovingAverageStrategy)')
    register_parser.add_argument('--description', '-d', help='Strategy description')

    # Remove strategy
    remove_parser = subparsers.add_parser('remove', help='Remove strategy')
    remove_parser.add_argument('name', help='Strategy name to remove')

    # Test strategy
    test_parser = subparsers.add_parser('test', help='Test loading a strategy')
    test_parser.add_argument('name', help='Strategy name to test')

    args = parser.parse_args()

    if args.command == 'list':
        list_strategies()
    elif args.command == 'register':
        register_strategy(args.name, args.module, args.class_name, args.description or "")
    elif args.command == 'remove':
        registry.remove_strategy(args.name)
    elif args.command == 'test':
        try:
            strategy = get_strategy_by_name(args.name)
            print(f"Successfully loaded: {strategy.name}")
        except Exception as e:
            print(f"Error loading strategy: {e}")
    else:
        parser.print_help()