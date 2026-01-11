"""
Chemical Reaction Engineering Module
Comprehensive collection of classes for chemical reaction engineering calculations
"""

from .chemicalReaction import (
    ReactionKinetics,
    ReactorDesign,
    ReactionEquilibrium,
    SelectivityAndYield,
    CatalystEffectiveness,
    MultipleReactions,
    NonIdealReactors,
    HeatEffects,
    ReactorSizing
)

__all__ = [
    'ReactionKinetics',
    'ReactorDesign',
    'ReactionEquilibrium',
    'SelectivityAndYield',
    'CatalystEffectiveness',
    'MultipleReactions',
    'NonIdealReactors',
    'HeatEffects',
    'ReactorSizing'
]

__version__ = '0.1.0'
