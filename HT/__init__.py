"""
Heat Transfer Module
Comprehensive collection of classes for heat transfer calculations
"""

from .heatTransfer import (
    Conduction,
    Convection,
    Radiation,
    HeatExchangers,
    TransientHeatTransfer,
    Fins,
    DimensionlessNumbers
)

__all__ = [
    'Conduction',
    'Convection',
    'Radiation',
    'HeatExchangers',
    'TransientHeatTransfer',
    'Fins',
    'DimensionlessNumbers'
]

__version__ = '0.1.0'
