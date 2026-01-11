"""
Mass Transfer Module
Comprehensive collection of classes for mass transfer calculations
"""

from .massTransfer import (
    Diffusion,
    MassTransferCoefficients,
    DimensionlessNumbers,
    AbsorptionStripping,
    Distillation,
    Extraction,
    Drying,
    MembraneSeparation,
    MassTransferWithReaction
)

__all__ = [
    'Diffusion',
    'MassTransferCoefficients',
    'DimensionlessNumbers',
    'AbsorptionStripping',
    'Distillation',
    'Extraction',
    'Drying',
    'MembraneSeparation',
    'MassTransferWithReaction'
]

__version__ = '0.1.0'
