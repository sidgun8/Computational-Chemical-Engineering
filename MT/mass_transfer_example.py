"""
Example: Mass Transfer Calculations
Demonstrates how to use the mass transfer classes from massTransfer.py
"""

from massTransfer import (
    Diffusion, MassTransferCoefficients, DimensionlessNumbers,
    AbsorptionStripping, Distillation, Extraction, Drying
)

# Example 1: Diffusion coefficient calculation
print("=" * 60)
print("Example 1: Diffusion Coefficient in Gas (Fuller-Schettler-Giddings)")
print("=" * 60)

T = 298.15  # Temperature (K)
P = 101325  # Pressure (Pa)
M_A = 32.0  # Molecular weight of O2 (kg/kmol)
M_B = 28.0  # Molecular weight of N2 (kg/kmol)
v_A = 16.6  # Diffusion volume of O2 (cm³/mol)
v_B = 17.9  # Diffusion volume of N2 (cm³/mol)

D_AB = Diffusion.diffusion_coefficient_gas_fuller(
    T=T, P=P, M_A=M_A, M_B=M_B, v_A=v_A, v_B=v_B
)

print(f"Temperature: {T} K")
print(f"Pressure: {P/1000:.2f} kPa")
print(f"Diffusion coefficient (O2 in N2): {D_AB*1e4:.4f} cm²/s")
print(f"Diffusion coefficient: {D_AB:.2e} m²/s\n")

# Example 2: Mass transfer coefficient for flat plate
print("=" * 60)
print("Example 2: Mass Transfer Coefficient - Flat Plate (Laminar)")
print("=" * 60)

Re = 50000  # Reynolds number
Sc = 0.6    # Schmidt number (typical for gases)
D_AB_plate = 2.5e-5  # Diffusion coefficient (m²/s)
L = 0.5     # Length of plate (m)

k_c = MassTransferCoefficients.mass_transfer_coefficient_flat_plate_laminar(
    Re=Re, Sc=Sc, D_AB=D_AB_plate, L=L
)

Sh = DimensionlessNumbers.sherwood_number(k_c=k_c, L=L, D_AB=D_AB_plate)

print(f"Reynolds number: {Re}")
print(f"Schmidt number: {Sc}")
print(f"Diffusion coefficient: {D_AB_plate:.2e} m²/s")
print(f"Plate length: {L} m")
print(f"Mass transfer coefficient: {k_c:.4f} m/s")
print(f"Sherwood number: {Sh:.2f}\n")

# Example 3: Dimensionless numbers
print("=" * 60)
print("Example 3: Dimensionless Numbers")
print("=" * 60)

mu = 1.8e-5   # Dynamic viscosity (Pa·s)
rho = 1.2     # Density (kg/m³)
D_AB_dim = 2.0e-5  # Diffusion coefficient (m²/s)

Sc = DimensionlessNumbers.schmidt_number(mu=mu, rho=rho, D_AB=D_AB_dim)
print(f"Schmidt number: {Sc:.3f}")

k_c_dim = 0.05  # Mass transfer coefficient (m/s)
L_dim = 0.1     # Characteristic length (m)
Sh = DimensionlessNumbers.sherwood_number(k_c=k_c_dim, L=L_dim, D_AB=D_AB_dim)
print(f"Sherwood number: {Sh:.2f}")

j_D = DimensionlessNumbers.j_factor_mass(Sh=Sh, Re=10000, Sc=Sc)
print(f"j-factor: {j_D:.4f}\n")

# Example 4: Absorption tower calculations
print("=" * 60)
print("Example 4: Absorption Tower - Number of Transfer Units")
print("=" * 60)

y_in = 0.05      # Inlet gas mole fraction
y_out = 0.01     # Outlet gas mole fraction
y_star_in = 0.0  # Equilibrium at inlet (pure solvent)
y_star_out = 0.0  # Equilibrium at outlet

NTU = AbsorptionStripping.number_of_transfer_units_absorption(
    y_in=y_in, y_out=y_out, y_star_in=y_star_in, y_star_out=y_star_out
)

K_y = 0.05       # Overall mass transfer coefficient (mol/(m²·s))
G = 10.0         # Gas molar flux (mol/(m²·s))
a = 200.0        # Interfacial area per unit volume (m²/m³)

HTU = AbsorptionStripping.height_of_transfer_unit(K_y=K_y, G=G, a=a)
H_tower = AbsorptionStripping.tower_height(NTU=NTU, HTU=HTU)

print(f"Inlet gas composition: {y_in:.3f}")
print(f"Outlet gas composition: {y_out:.3f}")
print(f"Number of transfer units (NTU): {NTU:.2f}")
print(f"Height of transfer unit (HTU): {HTU:.2f} m")
print(f"Required tower height: {H_tower:.2f} m\n")

# Example 5: Distillation calculations
print("=" * 60)
print("Example 5: Distillation - Relative Volatility and Equilibrium")
print("=" * 60)

P_A_sat = 101325  # Saturation pressure of component A (Pa)
P_B_sat = 40530   # Saturation pressure of component B (Pa)

alpha = Distillation.relative_volatility(P_A_sat=P_A_sat, P_B_sat=P_B_sat)
print(f"Relative volatility: {alpha:.3f}")

x_liquid = 0.5  # Liquid mole fraction
y_vapor = Distillation.equilibrium_relation_ideal(alpha=alpha, x=x_liquid)
print(f"Liquid composition: {x_liquid:.3f}")
print(f"Equilibrium vapor composition: {y_vapor:.3f}")

# Minimum number of stages
x_D = 0.95  # Distillate composition
x_W = 0.05  # Bottoms composition
N_min = Distillation.fenske_equation(alpha=alpha, x_D=x_D, x_W=x_W, N_min=None)
print(f"Minimum number of stages (Fenske): {N_min:.1f}\n")

# Example 6: Extraction calculations
print("=" * 60)
print("Example 6: Liquid-Liquid Extraction")
print("=" * 60)

C_extract = 0.8   # Concentration in extract phase (mol/m³)
C_raffinate = 0.2  # Concentration in raffinate phase (mol/m³)

beta = Extraction.distribution_coefficient(C_extract=C_extract, C_raffinate=C_raffinate)
print(f"Distribution coefficient: {beta:.2f}")

# Selectivity
beta_A = 4.0  # Distribution coefficient of A
beta_B = 0.5  # Distribution coefficient of B
selectivity = Extraction.selectivity(beta_A=beta_A, beta_B=beta_B)
print(f"Selectivity (A over B): {selectivity:.2f}\n")

# Example 7: Drying calculations
print("=" * 60)
print("Example 7: Drying - Moisture Content and Drying Rate")
print("=" * 60)

m_water = 0.3   # Mass of water (kg)
m_dry = 0.7     # Mass of dry solid (kg)
m_total = m_water + m_dry

X_wet = Drying.moisture_content_wet_basis(m_water=m_water, m_total=m_total)
X_dry = Drying.moisture_content_dry_basis(m_water=m_water, m_dry=m_dry)

print(f"Moisture content (wet basis): {X_wet:.3f} kg water/kg wet material")
print(f"Moisture content (dry basis): {X_dry:.3f} kg water/kg dry solid")

# Drying rate
k_dry = 0.001   # Mass transfer coefficient (kg/(m²·s))
A_dry = 1.0     # Surface area (m²)
X = 0.5         # Current moisture content (kg water/kg dry solid)
X_eq = 0.05     # Equilibrium moisture content (kg water/kg dry solid)

rate = Drying.drying_rate_constant_period(k=k_dry, A=A_dry, X=X, X_eq=X_eq)
print(f"Drying rate: {rate*3600:.2f} kg/h\n")

# Example 8: Mass transfer coefficient for sphere
print("=" * 60)
print("Example 8: Mass Transfer Coefficient - Flow Over Sphere")
print("=" * 60)

Re_sphere = 1000  # Reynolds number
Sc_sphere = 0.7   # Schmidt number
D_AB_sphere = 1.5e-5  # Diffusion coefficient (m²/s)
D_sphere = 0.01   # Sphere diameter (m)

k_c_sphere = MassTransferCoefficients.mass_transfer_coefficient_sphere(
    Re=Re_sphere, Sc=Sc_sphere, D_AB=D_AB_sphere, D=D_sphere
)

Sh_sphere = DimensionlessNumbers.sherwood_number(
    k_c=k_c_sphere, L=D_sphere, D_AB=D_AB_sphere
)

print(f"Reynolds number: {Re_sphere}")
print(f"Schmidt number: {Sc_sphere}")
print(f"Sphere diameter: {D_sphere} m")
print(f"Mass transfer coefficient: {k_c_sphere:.4f} m/s")
print(f"Sherwood number: {Sh_sphere:.2f}\n")

print("=" * 60)
print("All examples completed!")
print("=" * 60)

