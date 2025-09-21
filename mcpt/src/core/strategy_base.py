"""
Base classes and interfaces for standardised trading strategy implementation.
Provides plug-and-play architecture for future models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def _insample_signal_func(args):
    """Module-level signal function for in-sample MCPT."""
    os.environ['MCPT_SUBPROCESS'] = '1'
    ohlc, strategy, strategy_params = args
    result = strategy.generate_signal(ohlc, **strategy_params)
    price_returns = np.log(ohlc['close']).diff().shift(-1)
    return result.get_returns(price_returns)


def _walkforward_signal_func(args):
    """Module-level signal function for walk-forward MCPT."""
    os.environ['MCPT_SUBPROCESS'] = '1'
    ohlc, strategy, train_lookback, train_step = args
    price_returns = np.log(ohlc['close']).diff().shift(-1)
    signal = strategy.walk_forward_signal(ohlc, train_lookback, train_step)
    return signal * price_returns


def _run_single_permutation(args):
    ohlc, signal_func, signal_args, perm_kwargs = args
    from .bar_permute import get_permutation

    perm_kwargs_to_use = perm_kwargs or {}
    perm_ohlc = get_permutation(ohlc, **perm_kwargs_to_use)
    perm_returns = signal_func((perm_ohlc,) + signal_args)
    perm_pf = perm_returns[perm_returns > 0].sum() / perm_returns[perm_returns < 0].abs().sum()
    perm_cum_returns = perm_returns.cumsum()
    return perm_pf, perm_cum_returns


@dataclass
class StrategyResult:
    """Standardised result from strategy execution."""
    signal: pd.Series
    metadata: Dict[str, Any]

    def get_returns(self, price_returns: pd.Series) -> pd.Series:
        """Calculate strategy returns given price returns."""
        return self.signal * price_returns

    def get_profit_factor(self, price_returns: pd.Series) -> float:
        """Calculate profit factor for the strategy."""
        strategy_returns = self.get_returns(price_returns)
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]

        if len(negative_returns) == 0:
            return np.inf if len(positive_returns) > 0 else 0.0

        return positive_returns.sum() / negative_returns.abs().sum()


@dataclass
class OptimisationResult:
    """Result from strategy parameter optimisation."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]]


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name

    def _normalize_ohlc(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLC DataFrame column names to lowercase."""
        ohlc = ohlc.copy()
        ohlc.columns = ohlc.columns.str.lower()

        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in ohlc.columns]
        if missing_cols:
            raise ValueError(f"Data must contain columns: {required_cols}. Missing: {missing_cols}")

        return ohlc

    @abstractmethod
    def generate_signal(self, ohlc: pd.DataFrame, **params) -> StrategyResult:
        """
        Generate trading signals for given OHLC data.

        Args:
            ohlc: DataFrame with OHLC data
            **params: Strategy-specific parameters

        Returns:
            StrategyResult containing signals and metadata
        """
        pass

    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Define the parameter space for optimisation.

        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        pass

    def optimise(self, ohlc: pd.DataFrame, score_func: str = 'profit_factor') -> OptimisationResult:
        """
        Optimise strategy parameters.

        Args:
            ohlc: DataFrame with OHLC data
            score_func: Scoring function ('profit_factor', 'sharpe', etc.)

        Returns:
            OptimisationResult with best parameters and score
        """
        param_space = self.get_parameter_space()
        price_returns = np.log(ohlc['close']).diff().shift(-1)

        best_score = float('-inf')
        best_params = {}
        all_results = []

        param_combinations = self._generate_param_combinations(param_space)

        for params in param_combinations:
            try:
                result = self.generate_signal(ohlc, **params)
                score = self._calculate_score(result, price_returns, score_func)

                all_results.append((params.copy(), score))

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                continue

        return OptimisationResult(best_params, best_score, all_results)

    def _generate_param_combinations(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        if not param_space:
            return [{}]

        import itertools
        keys = list(param_space.keys())
        values = list(param_space.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _calculate_score(self, result: StrategyResult, price_returns: pd.Series, score_func: str) -> float:
        """Calculate score for optimisation."""
        if score_func == 'profit_factor':
            return result.get_profit_factor(price_returns)
        elif score_func == 'sharpe':
            strategy_returns = result.get_returns(price_returns)
            return strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        else:
            raise ValueError(f"Unknown score function: {score_func}")


class WalkForwardStrategy(ABC):
    """Base class for strategies that support walk-forward analysis."""

    @abstractmethod
    def walk_forward_signal(self, ohlc: pd.DataFrame, train_lookback: int = 24*365*4,
                           train_step: int = 24*30) -> pd.Series:
        """
        Generate walk-forward signals.

        Args:
            ohlc: DataFrame with OHLC data
            train_lookback: Number of periods for training window
            train_step: Number of periods between retraining

        Returns:
            Series with walk-forward signals
        """
        pass


class MCPTTester:
    """Monte Carlo Permutation Test framework."""

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy

    def _run_permutation_test(self, ohlc: pd.DataFrame, n_permutations: int, signal_func, signal_args, perm_kwargs=None, **extra_results) -> Dict[str, Any]:
        """
        Common MCPT logic for both in-sample and walk-forward testing.

        Args:
            ohlc: DataFrame with OHLC data
            n_permutations: Number of permutations to test
            signal_func: Function that takes ohlc and returns strategy returns
            perm_kwargs: Additional kwargs for get_permutation
            **extra_results: Additional results to include in return dict

        Returns:
            Dictionary with test results
        """
        from .bar_permute import get_permutation

        real_returns = signal_func((ohlc,) + signal_args)
        real_pf = real_returns[real_returns > 0].sum() / real_returns[real_returns < 0].abs().sum()
        real_cum_returns = real_returns.cumsum()

        max_workers = max(1, os.cpu_count() // 2)
        perm_better_count = 1
        permuted_pfs = []
        permuted_cum_returns = []

        # Disable tqdm in subprocesses to prevent multiple progress bars
        os.environ['TQDM_DISABLE'] = '1'

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            perm_args = [(ohlc, signal_func, signal_args, perm_kwargs) for _ in range(1, n_permutations)]
            futures = [executor.submit(_run_single_permutation, args) for args in perm_args]

            # Use tqdm only in main process for clean progress tracking
            with tqdm(total=len(futures), desc="Running permutations", disable=False) as pbar:
                for future in as_completed(futures):
                    perm_pf, perm_cum_ret = future.result()
                    if perm_pf >= real_pf:
                        perm_better_count += 1
                    permuted_pfs.append(perm_pf)
                    permuted_cum_returns.append(perm_cum_ret)
                    pbar.update(1)  # Update progress bar manually

        # Re-enable tqdm after multiprocessing
        if 'TQDM_DISABLE' in os.environ:
            del os.environ['TQDM_DISABLE']

        p_value = perm_better_count / n_permutations

        result = {
            'real_profit_factor': real_pf,
            'permuted_profit_factors': permuted_pfs,
            'real_cum_returns': real_cum_returns,
            'permuted_cum_returns': permuted_cum_returns,
            'p_value': p_value,
            'n_permutations': n_permutations,
        }
        result.update(extra_results)
        return result

    def run_insample_test(self, ohlc: pd.DataFrame, n_permutations: int = 1000,
                         **strategy_params) -> Dict[str, Any]:
        """
        Run in-sample MCPT.

        Args:
            ohlc: DataFrame with OHLC data
            n_permutations: Number of permutations to test
            **strategy_params: Parameters for the strategy

        Returns:
            Dictionary with test results
        """
        return self._run_permutation_test(
            ohlc, n_permutations, _insample_signal_func, (self.strategy, strategy_params),
            strategy_params=strategy_params
        )

    def run_walkforward_test(self, ohlc: pd.DataFrame, n_permutations: int = 200,
                           train_lookback: int = 24*365*4, train_step: int = 24*30) -> Dict[str, Any]:
        """
        Run walk-forward MCPT.

        Args:
            ohlc: DataFrame with OHLC data
            n_permutations: Number of permutations to test
            train_lookback: Training window size
            train_step: Retraining frequency

        Returns:
            Dictionary with test results
        """
        if not isinstance(self.strategy, WalkForwardStrategy):
            raise ValueError("Strategy must inherit from WalkForwardStrategy for walk-forward testing")

        return self._run_permutation_test(
            ohlc, n_permutations, _walkforward_signal_func, (self.strategy, train_lookback, train_step),
            perm_kwargs={'start_index': train_lookback},
            train_lookback=train_lookback,
            train_step=train_step
        )