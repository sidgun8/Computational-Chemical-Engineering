"""
Example usage of Chemical Engineering Thermodynamics formulas
"""

from thermodynamics import (
    EquationsOfState,
    ThermodynamicProperties,
    PhaseEquilibria,
    PropertyRelations,
    ChemicalReactions,
    WorkAndHeat,
    Compressibility,
    Mixing,
    HeatEngines,
    ThermodynamicCycles
)

# Example 1: Ideal Gas Law
print("=" * 60)
print("Example 1: Ideal Gas Law")
print("=" * 60)
n = 1.0  # 1 mole
T = 298.15  # 25°C
P = 101325  # 1 atm in Pa
V = EquationsOfState.ideal_gas_volume(n, T, P)
print(f"Volume of {n} mol ideal gas at {T} K and {P/1000:.2f} kPa: {V*1000:.4f} L")
print()

# Example 2: Enthalpy calculation
print("=" * 60)
print("Example 2: Enthalpy Calculation")
print("=" * 60)
Cp = 29.1  # J/(mol·K) for air
T1 = 298.15  # 25°C
T2 = 398.15  # 125°C
H = ThermodynamicProperties.enthalpy_ideal_gas(Cp, T2, T_ref=T1)
print(f"Enthalpy change for ideal gas from {T1} K to {T2} K: {H/1000:.2f} kJ/mol")
print()

# Example 3: Vapor pressure using Antoine equation
print("=" * 60)
print("Example 3: Vapor Pressure (Antoine Equation)")
print("=" * 60)
# Antoine constants for water (T in °C, P in mmHg)
# Converted for SI units: T in K, P in Pa
T_water = 373.15  # 100°C
A_water = 8.07131
B_water = 1730.63  # °C
C_water = 233.426  # °C
# For water, converting from mmHg to Pa and from °C to K
P_vap_water = 133.322 * 10 ** (A_water - B_water / ((T_water - 273.15) + C_water))
print(f"Vapor pressure of water at {T_water - 273.15:.1f}°C: {P_vap_water/1000:.2f} kPa")
print()

# Example 4: Clausius-Clapeyron equation
print("=" * 60)
print("Example 4: Vapor Pressure (Clausius-Clapeyron)")
print("=" * 60)
T_ref = 373.15  # 100°C
P_ref = 101325  # 1 atm
delta_H_vap = 40600  # J/mol for water
T_new = 323.15  # 50°C
P_vap = PhaseEquilibria.clausius_clapeyron(T_new, T_ref, P_ref, delta_H_vap)
print(f"Vapor pressure of water at {T_new - 273.15:.1f}°C: {P_vap/1000:.2f} kPa")
print()

# Example 5: Raoult's Law
print("=" * 60)
print("Example 5: Binary Mixture (Raoult's Law)")
print("=" * 60)
x_A = 0.5  # 50% mole fraction of component A
P_A_sat = 80000  # 80 kPa
P_B_sat = 40000  # 40 kPa
P_total = PhaseEquilibria.raoults_law_binary_total_pressure(x_A, P_A_sat, P_B_sat)
y_A = PhaseEquilibria.raoults_law_vapor_composition(x_A, P_A_sat, P_B_sat)
print(f"Binary mixture: x_A = {x_A:.2f}")
print(f"Total pressure: {P_total/1000:.2f} kPa")
print(f"Vapor composition y_A: {y_A:.3f}")
print()

# Example 6: Work calculation - Isothermal expansion
print("=" * 60)
print("Example 6: Isothermal Work")
print("=" * 60)
n = 2.0  # 2 moles
T = 298.15  # 25°C
V1 = 0.01  # 10 L
V2 = 0.02  # 20 L
W = WorkAndHeat.work_isothermal_ideal_gas(n, T, V1, V2)
print(f"Work for isothermal expansion of {n} mol at {T} K:")
print(f"  From {V1*1000:.1f} L to {V2*1000:.1f} L: {W:.2f} J")
print()

# Example 7: Equilibrium constant
print("=" * 60)
print("Example 7: Equilibrium Constant")
print("=" * 60)
delta_G_rxn = -50000  # J/mol (negative means spontaneous)
T = 298.15  # 25°C
K = ChemicalReactions.equilibrium_constant_delta_G(delta_G_rxn, T)
print(f"Gibbs free energy of reaction: {delta_G_rxn/1000:.2f} kJ/mol")
print(f"Equilibrium constant at {T} K: {K:.2e}")
print()

# Example 8: Carnot efficiency
print("=" * 60)
print("Example 8: Carnot Cycle Efficiency")
print("=" * 60)
T_hot = 500 + 273.15  # 500°C
T_cold = 50 + 273.15  # 50°C
eta = WorkAndHeat.carnot_efficiency(T_hot, T_cold)
print(f"Hot reservoir: {T_hot - 273.15:.1f}°C ({T_hot:.1f} K)")
print(f"Cold reservoir: {T_cold - 273.15:.1f}°C ({T_cold:.1f} K)")
print(f"Carnot efficiency: {eta*100:.2f}%")
print()

# Example 9: Entropy of mixing
print("=" * 60)
print("Example 9: Entropy of Mixing (Ideal Solution)")
print("=" * 60)
n_A = 1.0  # 1 mole of A
n_B = 1.0  # 1 mole of B
n_total = n_A + n_B
x_A = n_A / n_total
x_B = n_B / n_total
delta_S_mix = Mixing.ideal_solution_entropy([n_A, n_B], [x_A, x_B])
print(f"Mixing {n_A} mol A with {n_B} mol B:")
print(f"Mole fractions: x_A = {x_A:.2f}, x_B = {x_B:.2f}")
print(f"Entropy of mixing: {delta_S_mix:.2f} J/K")
print()

# Example 10: Compressibility factor
print("=" * 60)
print("Example 10: Compressibility Factor")
print("=" * 60)
P = 10000000  # 10 MPa (high pressure)
T = 298.15  # 25°C
n = 1.0  # 1 mole
# For real gas, V would be measured or calculated from EOS
# Here using ideal gas volume as approximation
V_ideal = EquationsOfState.ideal_gas_volume(n, T, P)
# Assume real gas volume is 0.95 * ideal (compression)
V_real = 0.95 * V_ideal
Z = Compressibility.compressibility_factor(P, V_real, n, T)
print(f"Pressure: {P/1e6:.1f} MPa, Temperature: {T:.1f} K")
print(f"Compressibility factor Z: {Z:.3f}")
if Z < 1:
    print("Gas is more compressible than ideal (attractive forces dominant)")
elif Z > 1:
    print("Gas is less compressible than ideal (repulsive forces dominant)")
else:
    print("Gas behaves ideally")
print()

# Example 11: Heat capacity calculation
print("=" * 60)
print("Example 11: Heat Capacity (Polynomial)")
print("=" * 60)
# Polynomial coefficients for water vapor (approximate)
a = 30.36
b = 0.00961
c = 1.184e-5
d = -0.37e-8
T = 373.15  # 100°C
Cp = PropertyRelations.heat_capacity_polynomial(T, a, b, c, d)
print(f"Heat capacity of water vapor at {T - 273.15:.1f}°C: {Cp:.2f} J/(mol·K)")
print()

# Example 12: Otto cycle efficiency
print("=" * 60)
print("Example 12: Otto Cycle Efficiency")
print("=" * 60)
r_compression = 10.0  # Compression ratio 10:1
gamma = 1.4  # For air
eta_otto = HeatEngines.otto_cycle_efficiency(r_compression, gamma)
print(f"Compression ratio: {r_compression:.1f}")
print(f"Specific heat ratio (γ): {gamma:.2f}")
print(f"Otto cycle efficiency: {eta_otto*100:.2f}%")
print()

print("=" * 60)
print("All examples completed!")
print("=" * 60)
