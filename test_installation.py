"""
Test script to verify package installation and imports
Run this after installing the package: python3 test_installation.py
"""

print("Testing ChemEng package installation...")
print("=" * 50)

# Test FPM module
try:
    from FPM import PipeFlow, FluidProperties
    Re = PipeFlow.reynolds_number(rho=1000, v=2.5, D=0.1, mu=0.001)
    print(f"✓ FPM module: Reynolds number = {Re:.2f}")
except Exception as e:
    print(f"✗ FPM module failed: {e}")

# Test CRE module
try:
    from CRE import ReactionKinetics
    k = ReactionKinetics.arrhenius_equation(A=1e12, E_a=50000, T=400)
    print(f"✓ CRE module: Rate constant = {k:.6e}")
except Exception as e:
    print(f"✗ CRE module failed: {e}")

# Test ChET module
try:
    from ChET import EquationsOfState
    P = EquationsOfState.ideal_gas_pressure(n=1.0, T=298.15, V=0.0245)
    print(f"✓ ChET module: Pressure = {P:.2f} Pa")
except Exception as e:
    print(f"✗ ChET module failed: {e}")

# Test HT module
try:
    from HT import Conduction, Convection, Radiation
    print("✓ HT module: Import successful")
except Exception as e:
    print(f"✗ HT module failed: {e}")

# Test MT module
try:
    from MT import Diffusion, MassTransferCoefficients
    print("✓ MT module: Import successful")
except Exception as e:
    print(f"✗ MT module failed: {e}")

# Test PDC module
try:
    from PDC import PIDController, FirstOrderSystems
    print("✓ PDC module: Import successful")
except Exception as e:
    print(f"✗ PDC module failed: {e}")

# Test PDPE module
try:
    from PDPE import TimeValueOfMoney, EconomicAnalysis
    print("✓ PDPE module: Import successful")
except Exception as e:
    print(f"✗ PDPE module failed: {e}")

# Test TranPheno module
try:
    from TranPheno import MomentumTransport
    print("✓ TranPheno module: Import successful")
except Exception as e:
    print(f"✗ TranPheno module failed: {e}")

# Test Math module
try:
    from Math import NumericalIntegration
    print("✓ Math module: Import successful")
except Exception as e:
    print(f"✗ Math module failed: {e}")

print("=" * 50)
print("Installation test complete!")
