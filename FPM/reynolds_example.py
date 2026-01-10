"""
Example: Calculating Reynolds Number
Demonstrates how to use the PipeFlow class from fluidProp.py
"""

from fluidProp import PipeFlow

# Example 1: Water flowing in a pipe
# Given conditions:
rho_water = 1000  # Density of water at 20°C (kg/m³)
v_water = 2.5     # Flow velocity (m/s)
D_pipe = 0.1      # Pipe diameter (m)
mu_water = 0.001  # Dynamic viscosity of water at 20°C (Pa·s or kg/(m·s))

# Calculate Reynolds number
Re_water = PipeFlow.reynolds_number(rho=rho_water, v=v_water, D=D_pipe, mu=mu_water)

print("Example 1: Water flowing in a pipe")
print(f"Density: {rho_water} kg/m³")
print(f"Velocity: {v_water} m/s")
print(f"Pipe diameter: {D_pipe} m")
print(f"Dynamic viscosity: {mu_water} Pa·s")
print(f"Reynolds number: {Re_water:.2f}")
print(f"Flow regime: {'Laminar' if Re_water < 2300 else 'Turbulent' if Re_water > 4000 else 'Transitional'}\n")

# Example 2: Air flowing in a duct
# Given conditions:
rho_air = 1.225      # Density of air at 15°C and 1 atm (kg/m³)
v_air = 10.0         # Flow velocity (m/s)
D_duct = 0.5         # Duct diameter (m)
mu_air = 1.81e-5     # Dynamic viscosity of air at 15°C (Pa·s)

# Calculate Reynolds number
Re_air = PipeFlow.reynolds_number(rho=rho_air, v=v_air, D=D_duct, mu=mu_air)

print("Example 2: Air flowing in a duct")
print(f"Density: {rho_air} kg/m³")
print(f"Velocity: {v_air} m/s")
print(f"Duct diameter: {D_duct} m")
print(f"Dynamic viscosity: {mu_air} Pa·s")
print(f"Reynolds number: {Re_air:.2e}")
print(f"Flow regime: {'Laminar' if Re_air < 2300 else 'Turbulent' if Re_air > 4000 else 'Transitional'}\n")

# Example 3: Using mass flow rate instead of velocity
# Given conditions:
m_dot = 5.0          # Mass flow rate (kg/s)
D_pipe2 = 0.08       # Pipe diameter (m)
mu_oil = 0.05        # Dynamic viscosity of oil (Pa·s)

# Calculate Reynolds number from mass flow rate
Re_oil = PipeFlow.reynolds_number_mass_flow(m_dot=m_dot, D=D_pipe2, mu=mu_oil)

print("Example 3: Oil flowing in a pipe (using mass flow rate)")
print(f"Mass flow rate: {m_dot} kg/s")
print(f"Pipe diameter: {D_pipe2} m")
print(f"Dynamic viscosity: {mu_oil} Pa·s")
print(f"Reynolds number: {Re_oil:.2f}")
print(f"Flow regime: {'Laminar' if Re_oil < 2300 else 'Turbulent' if Re_oil > 4000 else 'Transitional'}\n")

# Example 4: Complete workflow - Calculate Re, then friction factor
print("Example 4: Complete workflow - Re → Friction Factor")
# Using water example from above
epsilon = 0.000045  # Pipe roughness for commercial steel (m)

# Calculate friction factor using Swamee-Jain (explicit method)
f = PipeFlow.friction_factor_swamee_jain(Re=Re_water, epsilon=epsilon, D=D_pipe)

print(f"Reynolds number: {Re_water:.2f}")
print(f"Relative roughness (ε/D): {epsilon/D_pipe:.6f}")
print(f"Friction factor: {f:.4f}")

