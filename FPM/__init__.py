"""
Fluid Properties and Mechanics Module
Comprehensive collection of classes for fluid mechanics calculations
"""

from .fluidProp import (
    FluidProperties,
    PipeFlow,
    PumpCalculations,
    FlowMeasurement,
    FluidStatics,
    DimensionalAnalysis,
    TwoPhaseFlow,
    viscosity,
    density,
    compressibility
)

__all__ = [
    'FluidProperties',
    'PipeFlow',
    'PumpCalculations',
    'FlowMeasurement',
    'FluidStatics',
    'DimensionalAnalysis',
    'TwoPhaseFlow',
    'viscosity',
    'density',
    'compressibility'
]

__version__ = '0.1.0'
