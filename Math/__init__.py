"""
Computational Mathematics Module (Alias)
This is an alias to 'Useful Computational Math Implementations' for easier importing.
"""

# Import from the "Useful Computational Math Implementations" folder
import sys
from pathlib import Path
import importlib.util

# Get the parent directory
_parent_dir = Path(__file__).parent.parent

# Import directly from computationalMath.py
_math_path = _parent_dir / "Useful Computational Math Implementations" / "computationalMath.py"
if _math_path.exists():
    spec = importlib.util.spec_from_file_location("computational_math", _math_path)
    computational_math = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(computational_math)
    
    # Expose all classes
    NumericalIntegration = computational_math.NumericalIntegration
    LinearAlgebra = computational_math.LinearAlgebra
    RootFinding = computational_math.RootFinding
    Optimization = computational_math.Optimization
    FourierAnalysis = computational_math.FourierAnalysis
    Interpolation = computational_math.Interpolation
    CurveFitting = computational_math.CurveFitting
    Statistics = computational_math.Statistics
    
    __all__ = [
        'NumericalIntegration',
        'LinearAlgebra',
        'RootFinding',
        'Optimization',
        'FourierAnalysis',
        'Interpolation',
        'CurveFitting',
        'Statistics'
    ]
else:
    # If the folder doesn't exist, create empty placeholders
    __all__ = []

__version__ = '0.1.0'
