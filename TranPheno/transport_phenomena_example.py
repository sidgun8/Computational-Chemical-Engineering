"""
Example: Transport Phenomena Calculations
Demonstrates how to use the transport phenomena classes from transportPhenomena.py
"""

from transportPhenomena import (
    MomentumTransport, HeatTransfer, MassTransfer, 
    DimensionlessNumbers, TransportAnalogy, BoundaryLayers
)
import math

print("=" * 80)
print("TRANSPORT PHENOMENA EXAMPLES")
print("=" * 80)

# ============================================================================
# Example 1: Momentum Transport - Boundary Layer
# ============================================================================
print("\nExample 1: Momentum Transport - Boundary Layer on Flat Plate")
print("-" * 80)

# Given conditions:
v_inf = 5.0          # Free stream velocity (m/s)
x = 0.5              # Distance from leading edge (m)
rho = 1.225          # Air density (kg/m³)
mu = 1.81e-5         # Air dynamic viscosity (Pa·s)
L_char = 0.5         # Characteristic length (m)

# Calculate Reynolds number
Re_x = DimensionlessNumbers.reynolds_number(rho, v_inf, x, mu)

# Calculate boundary layer thickness
delta = MomentumTransport.boundary_layer_thickness_flat_plate(x, Re_x)

# Calculate displacement and momentum thickness
delta_star = MomentumTransport.displacement_thickness_flat_plate(delta)
theta = MomentumTransport.momentum_thickness_flat_plate(delta)

# Calculate drag coefficient
Re_L = DimensionlessNumbers.reynolds_number(rho, v_inf, L_char, mu)
C_D = MomentumTransport.drag_coefficient_flat_plate_laminar(Re_L)

print(f"Free stream velocity: {v_inf} m/s")
print(f"Distance from leading edge: {x} m")
print(f"Reynolds number at x: {Re_x:.2e}")
print(f"Boundary layer thickness: {delta*1000:.3f} mm")
print(f"Displacement thickness: {delta_star*1000:.3f} mm")
print(f"Momentum thickness: {theta*1000:.3f} mm")
print(f"Drag coefficient (Re_L={Re_L:.2e}): {C_D:.4f}")

# ============================================================================
# Example 2: Heat Transfer - Conduction through Plane Wall
# ============================================================================
print("\nExample 2: Heat Transfer - Conduction through Plane Wall")
print("-" * 80)

# Given conditions:
k_wall = 0.8         # Thermal conductivity of wall (W/(m·K))
A_wall = 10.0        # Wall area (m²)
T_hot = 373.15       # Hot side temperature (100°C in K)
T_cold = 293.15      # Cold side temperature (20°C in K)
L_wall = 0.2         # Wall thickness (m)

# Calculate heat transfer rate
Q = HeatTransfer.heat_conduction_plane_wall(k_wall, A_wall, T_hot, T_cold, L_wall)

# Calculate thermal resistance
R_th = HeatTransfer.thermal_resistance_conduction(L_wall, k_wall, A_wall)

print(f"Thermal conductivity: {k_wall} W/(m·K)")
print(f"Wall area: {A_wall} m²")
print(f"Temperature difference: {T_hot - T_cold} K ({T_hot - 273.15:.1f}°C to {T_cold - 273.15:.1f}°C)")
print(f"Wall thickness: {L_wall} m")
print(f"Heat transfer rate: {Q:.2f} W")
print(f"Thermal resistance: {R_th:.4f} K/W")

# ============================================================================
# Example 3: Heat Transfer - Convection with Nusselt Number
# ============================================================================
print("\nExample 3: Heat Transfer - Forced Convection over Flat Plate")
print("-" * 80)

# Given conditions:
v = 10.0             # Velocity (m/s)
x = 0.3              # Distance from leading edge (m)
rho_water = 1000     # Water density (kg/m³)
mu_water = 0.001     # Water dynamic viscosity (Pa·s)
Cp_water = 4180      # Water specific heat (J/(kg·K))
k_water = 0.6        # Water thermal conductivity (W/(m·K))
T_s = 353.15         # Surface temperature (80°C in K)
T_inf = 293.15       # Bulk temperature (20°C in K)
L = 0.5              # Plate length (m)

# Calculate dimensionless numbers
Re_x = DimensionlessNumbers.reynolds_number(rho_water, v, x, mu_water)
Pr = DimensionlessNumbers.prandtl_number(Cp_water, mu_water, k_water)

# Calculate Nusselt number (laminar)
Nu_x = HeatTransfer.nusselt_number_laminar_flat_plate(Re_x, Pr)

# Calculate heat transfer coefficient
h = HeatTransfer.heat_transfer_coefficient_from_nusselt(Nu_x, k_water, x)

# Calculate heat transfer rate per unit width
A_unit = x * 1.0  # Area per unit width
Q_unit = HeatTransfer.heat_transfer_convection(h, A_unit, T_s, T_inf)

print(f"Velocity: {v} m/s")
print(f"Position: {x} m from leading edge")
print(f"Reynolds number: {Re_x:.2e}")
print(f"Prandtl number: {Pr:.2f}")
print(f"Local Nusselt number: {Nu_x:.2f}")
print(f"Local heat transfer coefficient: {h:.2f} W/(m²·K)")
print(f"Heat transfer rate per meter width: {Q_unit:.2f} W/m")

# ============================================================================
# Example 4: Mass Transfer - Diffusion through Plane Wall
# ============================================================================
print("\nExample 4: Mass Transfer - Diffusion through Membrane")
print("-" * 80)

# Given conditions:
D_AB = 2.5e-9        # Diffusion coefficient (m²/s)
A_membrane = 0.1     # Membrane area (m²)
C_A1 = 100.0         # High concentration (mol/m³)
C_A2 = 10.0          # Low concentration (mol/m³)
L_membrane = 0.001   # Membrane thickness (1 mm)

# Calculate diffusion rate
N_A = MassTransfer.diffusion_steady_state_plane_wall(D_AB, A_membrane, C_A1, C_A2, L_membrane)

# Calculate mass transfer rate in kg/s (assuming molecular weight)
MW_A = 0.044         # Molecular weight (kg/mol) - e.g., CO2
m_dot_A = N_A * MW_A

print(f"Diffusion coefficient: {D_AB*1e9:.2f} × 10⁻⁹ m²/s")
print(f"Membrane area: {A_membrane} m²")
print(f"Concentration difference: {C_A1 - C_A2} mol/m³")
print(f"Membrane thickness: {L_membrane*1000:.1f} mm")
print(f"Molar diffusion rate: {N_A:.2e} mol/s")
print(f"Mass transfer rate: {m_dot_A:.2e} kg/s")

# ============================================================================
# Example 5: Mass Transfer - Convective Mass Transfer
# ============================================================================
print("\nExample 5: Mass Transfer - Convective Mass Transfer")
print("-" * 80)

# Given conditions:
v = 5.0              # Velocity (m/s)
x = 0.2              # Distance from leading edge (m)
rho_air = 1.225      # Air density (kg/m³)
mu_air = 1.81e-5     # Air dynamic viscosity (Pa·s)
D_AB_air = 2.6e-5    # Diffusion coefficient in air (m²/s)
C_As = 50.0          # Surface concentration (mol/m³)
C_A_inf = 10.0       # Bulk concentration (mol/m³)
A_surface = 0.05     # Surface area (m²)

# Calculate dimensionless numbers
Re_x = DimensionlessNumbers.reynolds_number(rho_air, v, x, mu_air)
Sc = DimensionlessNumbers.schmidt_number(mu_air, rho_air, D_AB_air)

# Calculate Sherwood number (laminar)
Sh_x = MassTransfer.sherwood_number_laminar_flat_plate(Re_x, Sc)

# Calculate mass transfer coefficient
k_c = MassTransfer.mass_transfer_coefficient_from_sherwood(Sh_x, D_AB_air, x)

# Calculate mass transfer rate
N_A = MassTransfer.mass_transfer_convection(k_c, A_surface, C_As, C_A_inf)

print(f"Reynolds number: {Re_x:.2e}")
print(f"Schmidt number: {Sc:.2f}")
print(f"Local Sherwood number: {Sh_x:.2f}")
print(f"Mass transfer coefficient: {k_c:.4f} m/s")
print(f"Mass transfer rate: {N_A:.2e} mol/s")

# ============================================================================
# Example 6: Transport Analogies
# ============================================================================
print("\nExample 6: Chilton-Colburn Analogy")
print("-" * 80)

# Given conditions:
C_f = 0.003          # Skin friction coefficient (f/2)
Pr = 7.0             # Prandtl number (water)
Sc = 600.0           # Schmidt number (typical for mass transfer)

# Apply Chilton-Colburn analogy
analogy_results = TransportAnalogy.chilton_colburn_analogy(C_f, Pr=Pr, Sc=Sc)

print(f"Skin friction coefficient (f/2): {C_f}")
print(f"Prandtl number: {Pr}")
print(f"Schmidt number: {Sc:.0f}")
print(f"j_H (heat transfer factor): {analogy_results['j_H']:.4f}")
print(f"j_M (mass transfer factor): {analogy_results['j_M']:.4f}")
print(f"Stanton number (heat): {analogy_results['Stanton_heat']:.4f}")
print(f"Stanton number (mass): {analogy_results['Stanton_mass']:.4f}")

# ============================================================================
# Example 7: Comparison of Boundary Layer Thicknesses
# ============================================================================
print("\nExample 7: Comparison of Boundary Layer Thicknesses")
print("-" * 80)

# Using conditions from Example 3
x = 0.3
Re_x_water = DimensionlessNumbers.reynolds_number(rho_water, v, x, mu_water)
Pr_water = DimensionlessNumbers.prandtl_number(Cp_water, mu_water, k_water)

# Momentum boundary layer
delta_M = MomentumTransport.boundary_layer_thickness_flat_plate(x, Re_x_water)

# Thermal boundary layer
delta_T = HeatTransfer.thermal_boundary_layer_thickness(x, Re_x_water, Pr_water)

# Ratio
ratio = BoundaryLayers.momentum_boundary_layer_ratio(Pr_water)

print(f"Distance from leading edge: {x} m")
print(f"Reynolds number: {Re_x_water:.2e}")
print(f"Prandtl number: {Pr_water:.2f}")
print(f"Momentum boundary layer thickness: {delta_M*1000:.3f} mm")
print(f"Thermal boundary layer thickness: {delta_T*1000:.3f} mm")
print(f"Ratio (δ_T/δ_M): {ratio:.3f}")
print(f"Verification: {delta_T/delta_M:.3f}")

# ============================================================================
# Example 8: Heat Transfer - Radiation
# ============================================================================
print("\nExample 8: Heat Transfer - Radiative Heat Transfer")
print("-" * 80)

# Given conditions:
epsilon = 0.9        # Emissivity
T_surface = 573.15   # Surface temperature (300°C in K)
T_surroundings = 293.15  # Surrounding temperature (20°C in K)
A = 1.0              # Surface area (m²)

# Calculate radiative heat flux (Stefan-Boltzmann)
q_rad = HeatTransfer.stefan_boltzmann_law(epsilon, T_surface)

# Calculate net radiative heat transfer
Q_rad = HeatTransfer.net_radiative_heat_transfer(epsilon, A, T_surface, T_surroundings)

print(f"Emissivity: {epsilon}")
print(f"Surface temperature: {T_surface - 273.15:.1f}°C")
print(f"Surrounding temperature: {T_surroundings - 273.15:.1f}°C")
print(f"Surface area: {A} m²")
print(f"Radiative heat flux: {q_rad:.2f} W/m²")
print(f"Net radiative heat transfer rate: {Q_rad:.2f} W")

# ============================================================================
# Example 9: Dimensionless Numbers Comparison
# ============================================================================
print("\nExample 9: Dimensionless Numbers for Transport Phenomena")
print("-" * 80)

# Using water properties
rho = 1000           # kg/m³
mu = 0.001           # Pa·s
Cp = 4180            # J/(kg·K)
k = 0.6              # W/(m·K)
D_AB = 1e-9          # m²/s (typical for diffusion in liquid)
v = 1.0              # m/s
L = 0.1              # m
h = 1000             # W/(m²·K)
k_c = 1e-5           # m/s

# Calculate all relevant dimensionless numbers
Re = DimensionlessNumbers.reynolds_number(rho, v, L, mu)
Pr = DimensionlessNumbers.prandtl_number(Cp, mu, k)
Sc = DimensionlessNumbers.schmidt_number(mu, rho, D_AB)
Le = DimensionlessNumbers.lewis_number(k, rho, Cp, D_AB)
Nu = DimensionlessNumbers.nusselt_number(h, L, k)
Sh = DimensionlessNumbers.sherwood_number(k_c, L, D_AB)
Pe_H = DimensionlessNumbers.peclet_number_heat(Re, Pr)
Pe_M = DimensionlessNumbers.peclet_number_mass(Re, Sc)

# Calculate diffusivities
alpha = DimensionlessNumbers.thermal_diffusivity(k, rho, Cp)
nu = DimensionlessNumbers.kinematic_viscosity(mu, rho)

print(f"Reynolds number (Re): {Re:.2e}")
print(f"Prandtl number (Pr): {Pr:.2f}")
print(f"Schmidt number (Sc): {Sc:.0f}")
print(f"Lewis number (Le): {Le:.2f}")
print(f"Nusselt number (Nu): {Nu:.2f}")
print(f"Sherwood number (Sh): {Sh:.2f}")
print(f"Peclet number (heat, Pe_H): {Pe_H:.2e}")
print(f"Peclet number (mass, Pe_M): {Pe_M:.2e}")
print(f"Thermal diffusivity (α): {alpha*1e7:.2f} × 10⁻⁷ m²/s")
print(f"Kinematic viscosity (ν): {nu*1e6:.2f} × 10⁻⁶ m²/s")
print(f"Mass diffusivity (D_AB): {D_AB*1e9:.2f} × 10⁻⁹ m²/s")

print("\n" + "=" * 80)
print("END OF EXAMPLES")
print("=" * 80)
