"""
Process Dynamics and Control Module
Comprehensive collection of classes for process dynamics and control calculations
"""

from .processDynamicsControl import (
    FirstOrderSystems,
    SecondOrderSystems,
    PIDController,
    ZieglerNicholsTuning,
    FrequencyResponse,
    StabilityAnalysis,
    ProcessModels,
    ClosedLoopAnalysis,
    DisturbanceRejection
)

__all__ = [
    'FirstOrderSystems',
    'SecondOrderSystems',
    'PIDController',
    'ZieglerNicholsTuning',
    'FrequencyResponse',
    'StabilityAnalysis',
    'ProcessModels',
    'ClosedLoopAnalysis',
    'DisturbanceRejection'
]

__version__ = '0.1.0'
