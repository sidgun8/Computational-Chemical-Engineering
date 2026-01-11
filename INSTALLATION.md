# Installation Guide

## Current Setup Status

✅ **Package Structure Complete**
- All modules have `__init__.py` files
- All imports work correctly
- Package is ready to use

## Installation Options

### Option 1: Work Directly (Recommended for Development)

Since your directory name "Chem Eng" contains a space, you can work directly without installing:

```python
# In your Python scripts or notebooks
import sys
from pathlib import Path

# Add the package directory to Python path
package_dir = Path("/Users/siddharthsrinivasan/Chem Eng")
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

# Now import modules
from FPM import PipeFlow
from CRE import ReactionKinetics
from ChET import EquationsOfState
```

### Option 2: Install in Editable Mode

**Note:** If the directory name has spaces, pip may have issues. To install:

1. **Option A:** Rename the directory to remove spaces (e.g., `ChemEng` or `chemeng`)
   ```bash
   cd "/Users/siddharthsrinivasan"
   mv "Chem Eng" ChemEng
   cd ChemEng
   pip install -e .
   ```

2. **Option B:** Use the full quoted path
   ```bash
   cd "/Users/siddharthsrinivasan/Chem Eng"
   pip install -e "/Users/siddharthsrinivasan/Chem Eng"
   ```

3. **Option C:** Create a symlink (Unix/Mac only)
   ```bash
   cd "/Users/siddharthsrinivasan"
   ln -s "Chem Eng" ChemEng
   cd ChemEng
   pip install -e .
   ```

### Option 3: Install from Source

If you want to distribute this package:

```bash
# Build the package
python setup.py sdist bdist_wheel

# Install it
pip install dist/chemeng-0.1.0.tar.gz
```

## Testing the Installation

After installation (or when working directly), run:

```bash
python3 test_installation.py
```

This will test all module imports and basic functionality.

## Usage

Once set up, you can import modules like:

```python
# Direct imports (most common)
from FPM import PipeFlow, FluidProperties
from CRE import ReactionKinetics, ReactorDesign
from ChET import EquationsOfState
from HT import Conduction, Convection
from MT import Diffusion, MassTransferCoefficients
from PDC import PIDController
from PDPE import TimeValueOfMoney, EconomicAnalysis
from TranPheno import MomentumTransport
from Math import NumericalIntegration, LinearAlgebra

# Example usage
Re = PipeFlow.reynolds_number(rho=1000, v=2.5, D=0.1, mu=0.001)
print(f"Reynolds number: {Re:.2f}")
```

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:** Make sure the package directory is in your Python path (see Option 1 above)

### Issue: "No module named 'FPM'"

**Solution:** Check that `__init__.py` exists in the FPM directory and that you're running Python from the correct directory.

### Issue: Pip installation fails

**Solution:** The directory name "Chem Eng" contains a space. Either:
- Rename the directory to remove spaces
- Use Option 1 (work directly without installing)
- Create a symlink (Option 2C)

## Current Package Structure

```
Chem Eng/
├── __init__.py                    # Main package
├── setup.py                       # Installation script
├── pyproject.toml                 # Modern packaging config
├── requirements.txt               # Dependencies
├── README.md                      # Package documentation
├── INSTALLATION.md                # This file
├── test_installation.py           # Installation test script
│
├── FPM/                           # Fluid Properties & Mechanics
│   ├── __init__.py               ✅
│   ├── fluidProp.py
│   └── reynolds_example.py
│
├── CRE/                           # Chemical Reaction Engineering
│   ├── __init__.py               ✅
│   ├── chemicalReaction.py
│   └── cre_example.py
│
├── ChET/                          # Chemical Engineering Thermodynamics
│   ├── __init__.py               ✅
│   ├── thermodynamics.py
│   └── thermodynamics_example.py
│
├── HT/                            # Heat Transfer
│   ├── __init__.py               ✅
│   ├── heatTransfer.py
│   └── heat_transfer_example.py
│
├── MT/                            # Mass Transfer
│   ├── __init__.py               ✅
│   ├── massTransfer.py
│   └── mass_transfer_example.py
│
├── PDC/                           # Process Dynamics and Control
│   ├── __init__.py               ✅
│   ├── processDynamicsControl.py
│   └── pdc_example.py
│
├── PDPE/                          # Process Design and Process Economics
│   ├── __init__.py               ✅
│   ├── processEconomics.py
│   └── economics_example.py
│
├── TranPheno/                     # Transport Phenomena
│   ├── __init__.py               ✅
│   ├── transportPhenomena.py
│   └── transport_phenomena_example.py
│
├── Math/                          # Computational Mathematics (alias)
│   └── __init__.py               ✅
│
└── Useful Computational Math Implementations/
    ├── __init__.py
    ├── computationalMath.py
    └── math_examples.py
```

All modules are properly configured and ready to use!
