# ChemEng - Chemical Engineering Computational Package

A comprehensive Python package for chemical engineering calculations and modeling. This package provides implementations of essential formulas, methods, and tools commonly used in chemical engineering applications.

## Features

### 🧪 Chemical Reaction Engineering (CRE)
- Reaction kinetics and rate laws (Arrhenius, zero/first/second order reactions)
- Reactor design (batch, CSTR, PFR, PBR)
- Reaction equilibrium and thermodynamics
- Selectivity and yield calculations
- Catalyst effectiveness and multiple reactions
- Non-ideal reactor models and heat effects
- Reactor sizing and optimization

### 🌡️ Chemical Engineering Thermodynamics (ChET)
- Equations of state (Ideal gas, Van der Waals, Peng-Robinson, SRK)
- Thermodynamic properties (enthalpy, entropy, Gibbs free energy)
- Phase equilibria (VLE, LLE, vapor pressure)
- Property relations and correlations
- Chemical reaction thermodynamics
- Work and heat calculations
- Compressibility and mixing
- Heat engines and thermodynamic cycles

### 💧 Fluid Properties & Mechanics (FPM)
- Fluid properties (viscosity, density, compressibility)
- Pipe flow calculations (Reynolds number, friction factors)
- Pump calculations and selection
- Flow measurement techniques
- Fluid statics and hydrostatics
- Dimensional analysis (Reynolds, Froude, Weber, Mach numbers)
- Two-phase flow models

### 🔥 Heat Transfer (HT)
- Conduction (Fourier's law, multi-layer walls, cylinders)
- Convection (forced and natural convection, Nusselt correlations)
- Radiation (Stefan-Boltzmann, view factors)
- Heat exchanger design (LMTD, NTU methods)
- Transient heat transfer (lumped capacitance, semi-infinite solids)
- Extended surfaces (fins)
- Dimensionless numbers (Nusselt, Prandtl, Biot)

### 📊 Mass Transfer (MT)
- Diffusion (Fick's law, molecular diffusion)
- Mass transfer coefficients
- Dimensionless numbers (Sherwood, Schmidt, Lewis)
- Absorption and stripping operations
- Distillation calculations
- Extraction processes
- Drying operations
- Membrane separation
- Mass transfer with reaction

### ⚙️ Process Dynamics and Control (PDC)
- First and second-order system responses
- PID controller design and tuning
- Ziegler-Nichols tuning methods
- Frequency response analysis
- Stability analysis (Routh-Hurwitz, Bode, Nyquist)
- Process modeling and identification
- Closed-loop analysis
- Disturbance rejection

### 💰 Process Design and Process Economics (PDPE)
- Time value of money (present/future value, annuities)
- Cost estimation (equipment, installation, operating costs)
- Economic analysis (NPV, IRR, payback period)
- Depreciation methods
- Break-even analysis
- Cash flow analysis
- Profitability metrics
- Sensitivity analysis

### 🌊 Transport Phenomena (TranPheno)
- Momentum transport (Navier-Stokes, boundary layers)
- Heat transfer (conduction, convection, radiation)
- Mass transfer (diffusion, convection)
- Dimensionless numbers and similarity
- Transport analogies (Reynolds, Chilton-Colburn)
- Boundary layer analysis

### 🧮 Computational Mathematics (Math)
- Numerical integration (Runge-Kutta, Euler, adaptive methods)
- Linear algebra (Gauss-Jordan, LU decomposition, eigenvalues)
- Root finding (Newton-Raphson, bisection, secant)
- Optimization (gradient descent, Newton-Raphson, golden section)
- Fourier analysis (Fourier series, DFT, power spectral density)
- Interpolation (linear, Lagrange, Newton, cubic spline)
- Curve fitting (linear/polynomial regression, exponential/power law)
- Statistics (mean, variance, correlation)

## Installation

### Option 1: Install in Editable Mode (Recommended for Development)

```bash
cd "/Users/siddharthsrinivasan/Chem Eng"
pip install -e .
```

### Option 2: Direct Import (Without Installation)

If you're working directly in the package directory, you can import modules directly:

```python
import sys
from pathlib import Path

# Add the package directory to Python path
package_dir = Path("/Users/siddharthsrinivasan/Chem Eng")
sys.path.insert(0, str(package_dir))

# Now import modules
from FPM import PipeFlow
from CRE import ReactionKinetics
```

### Option 3: Install with Optional Dependencies

```bash
pip install -e ".[dev]"      # Development dependencies
pip install -e ".[all]"      # All optional dependencies (numpy, scipy, matplotlib)
```

## Quick Start

### Example 1: Calculate Reynolds Number (Fluid Mechanics)

```python
from FPM import PipeFlow

# Water flowing in a pipe
rho = 1000    # Density (kg/m³)
v = 2.5       # Velocity (m/s)
D = 0.1       # Diameter (m)
mu = 0.001    # Viscosity (Pa·s)

Re = PipeFlow.reynolds_number(rho=rho, v=v, D=D, mu=mu)
print(f"Reynolds number: {Re:.2f}")
print(f"Flow regime: {'Laminar' if Re < 2300 else 'Turbulent' if Re > 4000 else 'Transitional'}")
```

### Example 2: Reaction Rate Constant (Chemical Reaction Engineering)

```python
from CRE import ReactionKinetics

# Calculate rate constant using Arrhenius equation
A = 1e12           # Pre-exponential factor
E_a = 50000        # Activation energy (J/mol)
T = 400            # Temperature (K)
R = 8.314          # Gas constant (J/(mol·K))

k = ReactionKinetics.arrhenius_equation(A=A, E_a=E_a, T=T, R=R)
print(f"Rate constant: {k:.6e}")
```

### Example 3: Ideal Gas Law (Thermodynamics)

```python
from ChET import EquationsOfState

# Calculate pressure using ideal gas law
n = 1.0        # Moles (mol)
T = 298.15     # Temperature (K)
V = 0.0245     # Volume (m³)

P = EquationsOfState.ideal_gas_pressure(n=n, T=T, V=V)
print(f"Pressure: {P:.2f} Pa = {P/101325:.2f} atm")
```

### Example 4: Heat Conduction (Heat Transfer)

```python
from HT import Conduction

# One-dimensional steady-state conduction
k = 50         # Thermal conductivity (W/(m·K))
A = 1.0        # Area (m²)
L = 0.1        # Thickness (m)
T_hot = 100    # Hot side temperature (°C)
T_cold = 20    # Cold side temperature (°C)

Q = Conduction.steady_state_1d(k=k, A=A, L=L, T1=T_hot, T2=T_cold)
print(f"Heat transfer rate: {Q:.2f} W")
```

### Example 5: Numerical Integration (Mathematics)

```python
from Math import NumericalIntegration
import math

# Solve ODE: dy/dt = -k*y (first-order decay)
def decay_ode(t, y):
    k = 0.5
    return -k * y

times, values = NumericalIntegration.runge_kutta_4(
    f=decay_ode, y0=1.0, t0=0.0, t_end=5.0, h=0.1
)

print(f"Solution at t={times[-1]}: y={values[-1]:.6f}")
print(f"Analytical: {math.exp(-0.5*times[-1]):.6f}")
```

## Package Structure

```
Chem Eng/
├── __init__.py                    # Main package init
├── setup.py                       # Installation script
├── pyproject.toml                 # Modern Python packaging config
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore file
│
├── FPM/                           # Fluid Properties & Mechanics
│   ├── __init__.py
│   ├── fluidProp.py
│   └── reynolds_example.py
│
├── CRE/                           # Chemical Reaction Engineering
│   ├── __init__.py
│   ├── chemicalReaction.py
│   └── cre_example.py
│
├── ChET/                          # Chemical Engineering Thermodynamics
│   ├── __init__.py
│   ├── thermodynamics.py
│   └── thermodynamics_example.py
│
├── HT/                            # Heat Transfer
│   ├── __init__.py
│   ├── heatTransfer.py
│   └── heat_transfer_example.py
│
├── MT/                            # Mass Transfer
│   ├── __init__.py
│   ├── massTransfer.py
│   └── mass_transfer_example.py
│
├── PDC/                           # Process Dynamics and Control
│   ├── __init__.py
│   ├── processDynamicsControl.py
│   └── pdc_example.py
│
├── PDPE/                          # Process Design and Process Economics
│   ├── __init__.py
│   ├── processEconomics.py
│   └── economics_example.py
│
├── TranPheno/                     # Transport Phenomena
│   ├── __init__.py
│   ├── transportPhenomena.py
│   └── transport_phenomena_example.py
│
├── Math/                          # Computational Mathematics (alias)
│   └── __init__.py
│
└── Useful Computational Math Implementations/
    ├── __init__.py
    ├── computationalMath.py
    └── math_examples.py
```

## Import Examples

### Direct Module Imports

```python
# Fluid mechanics
from FPM import PipeFlow, FluidProperties, PumpCalculations

# Reaction engineering
from CRE import ReactionKinetics, ReactorDesign

# Thermodynamics
from ChET import EquationsOfState, ThermodynamicProperties

# Heat transfer
from HT import Conduction, Convection, HeatExchangers

# Mass transfer
from MT import Diffusion, MassTransferCoefficients, Distillation

# Process control
from PDC import PIDController, ZieglerNicholsTuning

# Economics
from PDPE import TimeValueOfMoney, EconomicAnalysis

# Transport phenomena
from TranPheno import MomentumTransport, TransportAnalogy

# Mathematics
from Math import NumericalIntegration, LinearAlgebra, Optimization
```

## Usage in Notebooks and Scripts

Each module directory contains example files demonstrating how to use the classes and methods. For instance:

- `FPM/reynolds_example.py` - Fluid mechanics examples
- `CRE/cre_example.py` - Reaction engineering examples
- `ChET/thermodynamics_example.py` - Thermodynamics examples
- And more...

Run these examples to see how the package works:

```bash
cd FPM
python reynolds_example.py
```

## Dependencies

### Required
- **Python 3.8+** (uses only standard library - no external dependencies required!)

### Optional
- `numpy>=1.20.0` - For array operations (if needed in future)
- `scipy>=1.7.0` - For advanced numerical methods (if needed in future)
- `matplotlib>=3.3.0` - For plotting (if needed in future)

### Development
- `pytest>=7.0` - Testing framework
- `pytest-cov>=4.0` - Coverage reporting
- `black>=23.0` - Code formatting
- `flake8>=6.0` - Linting
- `mypy>=1.0` - Type checking

## Chemical Engineering Applications

This package is designed for:

- **Process Design**: Equipment sizing, optimization
- **Process Simulation**: ODE solving, numerical methods
- **Process Control**: Controller design, tuning, stability analysis
- **Economic Analysis**: Cost estimation, profitability analysis
- **Research & Education**: Teaching tools, research calculations
- **Data Analysis**: Interpolation, curve fitting, regression
- **Optimization**: Process optimization, parameter estimation

## Contributing

This is a personal/research package. Feel free to extend it with additional modules or improvements.

## Notes

- All methods include comprehensive documentation and parameter descriptions
- Units are typically SI (kg, m, s, K, Pa, J, mol)
- Methods include error handling and validation where appropriate
- The package uses static methods in classes for organization
- All modules follow consistent naming conventions

## License

MIT License - See LICENSE file (if provided) or use freely for educational and research purposes.

## Version

Current version: **0.1.0**

## Contact

Created by Siddharth Srinivasan

---

**Happy Computing! 🧪⚗️🔬**
