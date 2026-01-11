# Quick Start Guide

## Get Started in 30 Seconds

### 1. Add to Python Path (One-time setup in your script)

```python
import sys
from pathlib import Path

# Add package directory to path
sys.path.insert(0, str(Path("/Users/siddharthsrinivasan/Chem Eng")))
```

### 2. Import and Use

```python
# Fluid Mechanics
from FPM import PipeFlow
Re = PipeFlow.reynolds_number(rho=1000, v=2.5, D=0.1, mu=0.001)
print(f"Reynolds number: {Re:.2f}")

# Reaction Engineering
from CRE import ReactionKinetics
k = ReactionKinetics.arrhenius_equation(A=1e12, E_a=50000, T=400)
print(f"Rate constant: {k:.6e}")

# Thermodynamics
from ChET import EquationsOfState
P = EquationsOfState.ideal_gas_pressure(n=1.0, T=298.15, V=0.0245)
print(f"Pressure: {P:.2f} Pa")

# Heat Transfer
from HT import Conduction, Convection, Radiation

# Mass Transfer
from MT import Diffusion, MassTransferCoefficients

# Process Control
from PDC import PIDController, FirstOrderSystems

# Economics
from PDPE import TimeValueOfMoney, EconomicAnalysis

# Transport Phenomena
from TranPheno import MomentumTransport, TransportAnalogy

# Mathematics
from Math import NumericalIntegration, LinearAlgebra, Optimization
```

## Example: Complete Calculation Script

```python
import sys
from pathlib import Path

# Setup
sys.path.insert(0, str(Path("/Users/siddharthsrinivasan/Chem Eng")))

# Imports
from FPM import PipeFlow, FluidProperties
from CRE import ReactionKinetics
from ChET import EquationsOfState

# Example 1: Calculate Reynolds number
rho = 1000    # kg/m³
v = 2.5       # m/s
D = 0.1       # m
mu = 0.001    # Pa·s

Re = PipeFlow.reynolds_number(rho=rho, v=v, D=D, mu=mu)
flow_regime = "Laminar" if Re < 2300 else "Turbulent" if Re > 4000 else "Transitional"
print(f"Re = {Re:.2f} ({flow_regime})")

# Example 2: Calculate reaction rate constant
A = 1e12
E_a = 50000  # J/mol
T = 400      # K

k = ReactionKinetics.arrhenius_equation(A=A, E_a=E_a, T=T)
print(f"k = {k:.6e}")

# Example 3: Ideal gas pressure
n = 1.0      # mol
T = 298.15   # K
V = 0.0245   # m³

P = EquationsOfState.ideal_gas_pressure(n=n, T=T, V=V)
print(f"P = {P:.2f} Pa = {P/101325:.2f} atm")
```

## Available Modules

| Module | Code | Description |
|--------|------|-------------|
| Fluid Properties & Mechanics | `FPM` | Reynolds number, friction factors, pumps, flow measurement |
| Chemical Reaction Engineering | `CRE` | Kinetics, reactor design, equilibrium, catalysis |
| Chemical Engineering Thermodynamics | `ChET` | EOS, properties, phase equilibria, cycles |
| Heat Transfer | `HT` | Conduction, convection, radiation, heat exchangers |
| Mass Transfer | `MT` | Diffusion, distillation, extraction, membranes |
| Process Dynamics & Control | `PDC` | PID control, tuning, stability, frequency response |
| Process Economics | `PDPE` | NPV, IRR, cost estimation, economic analysis |
| Transport Phenomena | `TranPheno` | Momentum, heat, and mass transport analogies |
| Mathematics | `Math` | Numerical methods, linear algebra, optimization |

## Need Help?

- See `README.md` for detailed documentation
- See `INSTALLATION.md` for installation instructions
- Check example files in each module directory (e.g., `FPM/reynolds_example.py`)
- Run `python3 test_installation.py` to verify your setup

## Common Classes

### FPM (Fluid Mechanics)
- `FluidProperties` - Basic fluid properties
- `PipeFlow` - Pipe flow calculations
- `PumpCalculations` - Pump sizing and selection
- `DimensionalAnalysis` - Dimensionless numbers

### CRE (Reaction Engineering)
- `ReactionKinetics` - Rate laws and kinetics
- `ReactorDesign` - Reactor sizing and design
- `ReactionEquilibrium` - Equilibrium calculations

### ChET (Thermodynamics)
- `EquationsOfState` - Ideal gas, VdW, PR, SRK
- `ThermodynamicProperties` - Enthalpy, entropy, Gibbs
- `PhaseEquilibria` - VLE, LLE calculations

### Math
- `NumericalIntegration` - Runge-Kutta, Euler methods
- `LinearAlgebra` - Matrix operations, solving systems
- `RootFinding` - Newton-Raphson, bisection
- `Optimization` - Gradient descent, optimization

Happy Computing! 🧪⚗️🔬
