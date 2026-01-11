"""
Process Design and Process Economics Module
Comprehensive collection of classes for process economics and economic analysis
"""

from .processEconomics import (
    TimeValueOfMoney,
    CostEstimation,
    EconomicAnalysis,
    Depreciation,
    BreakEvenAnalysis,
    CashFlowAnalysis,
    ProfitabilityMetrics,
    SensitivityAnalysis
)

__all__ = [
    'TimeValueOfMoney',
    'CostEstimation',
    'EconomicAnalysis',
    'Depreciation',
    'BreakEvenAnalysis',
    'CashFlowAnalysis',
    'ProfitabilityMetrics',
    'SensitivityAnalysis'
]

__version__ = '0.1.0'
