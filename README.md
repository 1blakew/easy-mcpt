# MCPT Trading Strategy Framework

A comprehensive framework for implementing and testing trading strategies using Monte Carlo Permutation Tests (MCPT) with parallel processing and visualisation.

## Directory Structure

```
easy-mcpt/
├── mcpt/                   # Core MCPT engine
│   └── src/
│       ├── core/           # Framework
│       └── scripts/        # Entry points & runners
└── resources/
    ├── registry/           # Strategy configuration registry
    ├── strategies/         # Strategies & template
    ├── test/               # Example strategies & data
    └── utils/              # Utility functions
```

## Quick Start

### Command Line Interface

**Syntax:**

```bash
python -m mcpt.src.scripts.run_mcpt <test_type> [options]
```

**Parameters:**

- **test_type** : {'insample', 'walkforward', 'both'}
    Type of MCPT test to run.

- **--data-file, -d** : str
    Path to data file. Supported formats: CSV, Parquet, JSON, Excel.

- **--strategy, -s** : str
    Strategy name from registry. Use `--list-strategies` to see available options.

- **--config, -c** : str, optional
    Path to configuration file. Overrides command-line arguments.

- **--list-strategies** : bool, optional
    List all registered strategies and exit.

- **--permutations, -p** : int, optional
    Number of permutations to test. Default varies by test type.

- **--start-year** : int, optional
    Start year for data filtering.

- **--end-year** : int, optional
    End year for data filtering.

- **--train-lookback** : int, optional
    Training window size in periods (walk-forward only).

- **--train-step** : int, optional
    Retraining frequency in periods (walk-forward only).

- **--no-plot** : bool, optional
    Disable plotting results.

- **--help, -h** : bool, optional
    Show help message and exit.

**Examples:**

Basic usage:

```bash
python -m mcpt.src.scripts.run_mcpt insample --data-file data.csv --strategy bollinger
```

With time filtering and custom permutations:

```bash
python -m mcpt.src.scripts.run_mcpt walkforward --data-file data.csv --strategy donchian --start-year 2020 --end-year 2023 --permutations 500
```

List available strategies:

```bash
python -m mcpt.src.scripts.run_mcpt --list-strategies
```

### Run MCPT Tests

```bash
# List available strategies
python -m mcpt.src.scripts.run_mcpt --list-strategies insample

# Run in-sample MCPT test
python -m mcpt.src.scripts.run_mcpt insample --data-file resources/data/data-file.csv --strategy donchian --permutations 1000

# Run walk-forward MCPT test
python -m mcpt.src.scripts.run_mcpt walkforward --data-file resources/data/data-file.csv --strategy donchian --permutations 200

# Run both tests in sequence
python -m mcpt.src.scripts.run_mcpt both --data-file resources/data/data-file.csv --strategy donchian --permutations 200
```

## Key Features

- **Parallel Processing**
- **Visualisation**
- **Strategy Registry**
- **Adaptive Logic**
- **Supports Multiple File Types**
- **Three Testing Modes**

## Implementing New Strategies

```python
from mcpt.src.core.strategy_base import TradingStrategy, StrategyResult
from resources.utils.registry_config import register_strategy
import pandas as pd

class MyStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("My Custom Strategy")

    def generate_signal(self, ohlc: pd.DataFrame, **params) -> StrategyResult:
        # Normalise OHLC columns (case-insensitive)
        ohlc = self._normalise_ohlc(ohlc)

        # Implement your strategy logic here
        signal = pd.Series(0, index=ohlc.index)  # Your signals
        metadata = {'strategy_name': self.name}
        return StrategyResult(signal=signal, metadata=metadata)

    def get_parameter_space(self) -> dict:
        return {'param1': [1, 2, 3], 'param2': [0.1, 0.2, 0.3]}

# Register your strategy
register_strategy("my strategy", "my_module", "MyStrategy", "My custom trading strategy")
```

## Strategy Implementation Guide

## Core Components

- **`mcpt/src/core/`** - Framework foundation (strategy base classes, MCPT testing)
- **`mcpt/src/scripts/`** - Entry points and unified runner
- **`resources/test/`** - Example strategies (Donchian, Bollinger Bands)
- **`resources/strategies/`** - Strategy templates
- **`resources/utils/`** - Strategy registry and configuration management
- **`resources/registry/`** - Strategy configuration files

## MCPT Testing

The framework provides statistical validation of trading strategies through Monte Carlo Permutation Tests:

- **In-Sample MCPT**: Tests strategy performance against permuted data with parallel optimisation
- **Walk-Forward MCPT**: More robust testing with realistic trading conditions and adaptive training windows
- **P-Value < 0.05**: Indicates statistical significance

## Dependencies

- pandas, numpy, matplotlib, tqdm, yfinance

## Getting Started

1. Copy strategy templates from `resources/strategies/`
2. Implement your strategy following the examples in `resources/test/strats/`
3. Test with MCPT using the unified runner
