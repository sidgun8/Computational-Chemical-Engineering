"""
Examples: Chemical Reaction Engineering Calculations
Demonstrates how to use various classes from chemicalReaction.py
"""

from chemicalReaction import (
    ReactionKinetics, ReactorDesign, ReactionEquilibrium,
    SelectivityAndYield, CatalystEffectiveness, MultipleReactions,
    HeatEffects, ReactorSizing
)

print("=" * 70)
print("CHEMICAL REACTION ENGINEERING EXAMPLES")
print("=" * 70)

# Example 1: Arrhenius Equation and Rate Constants
print("\n1. REACTION KINETICS - Arrhenius Equation")
print("-" * 70)

A = 1e13  # Pre-exponential factor (1/s)
E_a = 50000  # Activation energy (J/mol)
T1 = 300  # Temperature 1 (K)
T2 = 350  # Temperature 2 (K)

k1 = ReactionKinetics.arrhenius_equation(A=A, E_a=E_a, T=T1)
k2 = ReactionKinetics.arrhenius_equation(A=A, E_a=E_a, T=T2)

print(f"Pre-exponential factor (A): {A:.2e} 1/s")
print(f"Activation energy (E_a): {E_a/1000:.1f} kJ/mol")
print(f"Rate constant at T1 = {T1} K: k1 = {k1:.4e} 1/s")
print(f"Rate constant at T2 = {T2} K: k2 = {k2:.4e} 1/s")
print(f"Ratio k2/k1: {k2/k1:.2f}")

# Verify using temperature dependence function
k2_check = ReactionKinetics.temperature_dependence_k2(k1=k1, E_a=E_a, T1=T1, T2=T2)
print(f"Verification - k2 from temperature dependence: {k2_check:.4e} 1/s")

# Example 2: First-Order Reaction in Batch Reactor
print("\n2. BATCH REACTOR - First-Order Reaction")
print("-" * 70)

C_A0 = 2.0  # Initial concentration (mol/L)
C_A_final = 0.5  # Final concentration (mol/L)
k = 0.1  # Rate constant (1/s)

time_required = ReactorDesign.batch_reactor_time_first_order(
    C_A0=C_A0, C_A=C_A_final, k=k
)

print(f"Initial concentration (C_A0): {C_A0} mol/L")
print(f"Final concentration (C_A): {C_A_final} mol/L")
print(f"Rate constant (k): {k} 1/s")
print(f"Time required: {time_required:.2f} s ({time_required/60:.2f} minutes)")

# Calculate conversion
X = ReactorDesign.conversion_from_conc(C_A0=C_A0, C_A=C_A_final)
print(f"Conversion achieved: {X*100:.1f}%")

# Example 3: CSTR Design - First-Order Reaction
print("\n3. CSTR DESIGN - First-Order Reaction")
print("-" * 70)

F_A0 = 10.0  # Inlet molar flow rate (mol/s)
X_desired = 0.90  # Desired conversion
C_A0 = 2.0  # Inlet concentration (mol/L)
k = 0.1  # Rate constant (1/s)

# Calculate volumetric flow rate
v0 = F_A0 / C_A0  # L/s

# Calculate reactor volume
V_cstr = ReactorDesign.cstr_volume_first_order(
    F_A0=F_A0, X=X_desired, k=k, C_A0=C_A0, v0=v0
)

# Calculate space time
tau = ReactorDesign.space_time(V=V_cstr, v0=v0)

print(f"Inlet molar flow rate (F_A0): {F_A0} mol/s")
print(f"Inlet concentration (C_A0): {C_A0} mol/L")
print(f"Volumetric flow rate (v0): {v0} L/s")
print(f"Desired conversion (X): {X_desired*100:.1f}%")
print(f"Rate constant (k): {k} 1/s")
print(f"Required CSTR volume: {V_cstr:.2f} L ({V_cstr/1000:.3f} m³)")
print(f"Space time (tau): {tau:.2f} s")

# Example 4: PFR Design - Second-Order Reaction
print("\n4. PFR DESIGN - Second-Order Reaction")
print("-" * 70)

F_A0_pfr = 10.0  # Inlet molar flow rate (mol/s)
X_pfr = 0.85  # Desired conversion
C_A0_pfr = 2.0  # Inlet concentration (mol/L)
k_pfr = 0.05  # Rate constant (L/(mol·s))
v0_pfr = F_A0_pfr / C_A0_pfr  # L/s

V_pfr = ReactorDesign.pfr_volume_second_order(
    F_A0=F_A0_pfr, X=X_pfr, k=k_pfr, C_A0=C_A0_pfr, v0=v0_pfr
)

tau_pfr = ReactorDesign.space_time(V=V_pfr, v0=v0_pfr)

print(f"Inlet molar flow rate (F_A0): {F_A0_pfr} mol/s")
print(f"Inlet concentration (C_A0): {C_A0_pfr} mol/L")
print(f"Desired conversion (X): {X_pfr*100:.1f}%")
print(f"Rate constant (k): {k_pfr} L/(mol·s)")
print(f"Required PFR volume: {V_pfr:.2f} L ({V_pfr/1000:.3f} m³)")
print(f"Space time (tau): {tau_pfr:.2f} s")

# Compare with CSTR
print(f"\nComparison for same conversion ({X_pfr*100:.1f}%):")
print(f"PFR volume: {V_pfr:.2f} L")
print(f"CSTR volume: {ReactorDesign.cstr_volume_second_order(F_A0_pfr, X_pfr, k_pfr, C_A0_pfr, v0_pfr):.2f} L")
print(f"Volume ratio (CSTR/PFR): {ReactorDesign.cstr_volume_second_order(F_A0_pfr, X_pfr, k_pfr, C_A0_pfr, v0_pfr)/V_pfr:.2f}")

# Example 5: Equilibrium Constant Calculations
print("\n5. CHEMICAL EQUILIBRIUM")
print("-" * 70)

delta_G = -25000  # Standard Gibbs free energy change (J/mol)
T_eq = 400  # Temperature (K)

K_eq = ReactionEquilibrium.equilibrium_constant_from_gibbs(
    delta_G=delta_G, T=T_eq
)

print(f"Standard Gibbs free energy change (ΔG°): {delta_G/1000:.1f} kJ/mol")
print(f"Temperature: {T_eq} K")
print(f"Equilibrium constant (K): {K_eq:.4e}")

# Calculate K at different temperature using van't Hoff
delta_H = -50000  # Enthalpy change (J/mol)
T2_eq = 450  # New temperature (K)

K_eq2 = ReactionEquilibrium.van_t_hoff_equation(
    K1=K_eq, delta_H=delta_H, T1=T_eq, T2=T2_eq
)

print(f"\nAt T = {T2_eq} K (ΔH° = {delta_H/1000:.1f} kJ/mol):")
print(f"Equilibrium constant (K): {K_eq2:.4e}")
print(f"K increases by factor: {K_eq2/K_eq:.2f}")

# Example 6: Catalyst Effectiveness Factor
print("\n6. CATALYST EFFECTIVENESS FACTOR")
print("-" * 70)

k_cat = 0.1  # Rate constant (1/s)
D_eff = 1e-9  # Effective diffusivity (m²/s)
R_particle = 0.005  # Particle radius (m) = 5 mm
C_A_s = 10.0  # Surface concentration (mol/m³)
n = 1  # First-order reaction

# Calculate Thiele modulus for sphere
phi = CatalystEffectiveness.thiele_modulus_sphere(
    k=k_cat, D_eff=D_eff, R=R_particle, C_A_s=C_A_s, n=n
)

# Calculate effectiveness factor
eta = CatalystEffectiveness.effectiveness_factor_sphere(phi=phi)

print(f"Rate constant (k): {k_cat} 1/s")
print(f"Effective diffusivity (D_eff): {D_eff:.2e} m²/s")
print(f"Particle radius (R): {R_particle*1000:.1f} mm")
print(f"Surface concentration (C_A,s): {C_A_s} mol/m³")
print(f"Thiele modulus (φ): {phi:.3f}")
print(f"Effectiveness factor (η): {eta:.3f}")
print(f"Reaction rate inside particle relative to surface: {eta*100:.1f}%")

# Example 7: Selectivity and Yield
print("\n7. SELECTIVITY AND YIELD")
print("-" * 70)

F_A0_sel = 100.0  # Inlet molar flow rate of A (mol/s)
F_R_out = 70.0  # Outlet molar flow rate of desired product R (mol/s)
F_S_out = 20.0  # Outlet molar flow rate of undesired product S (mol/s)

X_sel = ReactorDesign.conversion_from_molar_flow(F_A0=F_A0_sel, F_A=(F_A0_sel - F_R_out - F_S_out))
S = SelectivityAndYield.selectivity_instantaneous(
    F_R_out=F_R_out, F_S_out=F_S_out
)
Y_R = SelectivityAndYield.yield_reaction(
    F_R_out=F_R_out, F_A0=F_A0_sel, nu_R=1, nu_A=1
)

print(f"Inlet flow rate of A (F_A0): {F_A0_sel} mol/s")
print(f"Outlet flow rate of desired product R (F_R): {F_R_out} mol/s")
print(f"Outlet flow rate of undesired product S (F_S): {F_S_out} mol/s")
print(f"Conversion (X): {X_sel*100:.1f}%")
print(f"Selectivity (S = F_R/F_S): {S:.2f}")
print(f"Yield (Y_R): {Y_R*100:.1f}%")
print(f"Verification: Y = X * S = {SelectivityAndYield.yield_from_conversion_and_selectivity(X_sel, S)*100:.1f}%")

# Example 8: Parallel Reactions - Maximizing Selectivity
print("\n8. PARALLEL REACTIONS - Selectivity Optimization")
print("-" * 70)

k1_par = 0.2  # Rate constant for A -> R (desired)
k2_par = 0.1  # Rate constant for A -> S (undesired)
alpha = 1  # Order of desired reaction
beta = 2  # Order of undesired reaction

# Calculate selectivity at different concentrations
C_A_values = [0.5, 1.0, 2.0, 5.0, 10.0]

print("Concentration dependence of selectivity:")
print(f"{'C_A (mol/L)':<15} {'Selectivity (S)':<20} {'Comment'}")
print("-" * 50)
for C_A in C_A_values:
    S_par = MultipleReactions.parallel_reactions_selectivity(
        k1=k1_par, k2=k2_par, alpha=alpha, beta=beta, C_A=C_A
    )
    comment = "Higher is better" if S_par > 2 else "Low selectivity"
    print(f"{C_A:<15.1f} {S_par:<20.3f} {comment}")

print("\nNote: For alpha < beta, lower C_A gives higher selectivity")
print("       For alpha > beta, higher C_A gives higher selectivity")

# Example 9: Series Reactions - Maximum Intermediate
print("\n9. SERIES REACTIONS - Maximizing Intermediate")
print("-" * 70)

C_A0_series = 1.0  # Initial concentration (mol/L)
k1_series = 0.5  # Rate constant A -> B (1/s)
k2_series = 0.2  # Rate constant B -> C (1/s)

# Calculate optimum time
t_opt = MultipleReactions.optimum_time_max_intermediate(
    C_A0=C_A0_series, k1=k1_series, k2=k2_series
)

# Calculate concentration of B at optimum time
C_B_opt = MultipleReactions.series_reactions_concentration_C(
    C_A0=C_A0_series, k1=k1_series, k2=k2_series, t=t_opt
)

print(f"Reaction: A -> B -> C")
print(f"Rate constant k1 (A->B): {k1_series} 1/s")
print(f"Rate constant k2 (B->C): {k2_series} 1/s")
print(f"Initial concentration of A: {C_A0_series} mol/L")
print(f"Optimum time to maximize B: {t_opt:.2f} s")
print(f"Maximum concentration of B: {C_B_opt:.4f} mol/L")

# Example 10: Adiabatic Temperature Rise
print("\n10. HEAT EFFECTS - Adiabatic Temperature Rise")
print("-" * 70)

X_heat = 0.80  # Conversion
delta_H_rxn = -50000  # Heat of reaction (J/mol) - negative for exothermic
F_A0_heat = 50.0  # Inlet molar flow rate (mol/s)
C_p_total = 5000.0  # Total heat capacity flow rate (J/(s·K))
T0 = 300  # Inlet temperature (K)

delta_T_ad = HeatEffects.adiabatic_temperature_rise(
    X=X_heat, delta_H_rxn=delta_H_rxn, F_A0=F_A0_heat, C_p_total=C_p_total
)

T_out = T0 + delta_T_ad

print(f"Conversion (X): {X_heat*100:.1f}%")
print(f"Heat of reaction (ΔH_rxn): {delta_H_rxn/1000:.1f} kJ/mol")
print(f"Inlet temperature (T0): {T0} K")
print(f"Total heat capacity flow rate (Σ F_i0 * C_pi): {C_p_total} J/(s·K)")
print(f"Adiabatic temperature rise (ΔT_ad): {delta_T_ad:.1f} K")
print(f"Outlet temperature (T_out): {T_out:.1f} K")

# Example 11: Multiple CSTRs in Series
print("\n11. REACTOR DESIGN - Multiple CSTRs in Series")
print("-" * 70)

X_total_series = 0.95  # Desired total conversion
k_multi = 0.1  # Rate constant (1/s)
tau_single = 20  # Space time per reactor (s)

N_cstrs = ReactorSizing.number_of_cstrs_in_series(
    X_total=X_total_series, k=k_multi, tau_single=tau_single
)

print(f"Desired conversion: {X_total_series*100:.1f}%")
print(f"Rate constant (k): {k_multi} 1/s")
print(f"Space time per reactor (tau): {tau_single} s")
print(f"Number of CSTRs required: {N_cstrs}")

# Compare with single CSTR
V_single = ReactorDesign.cstr_volume_first_order(
    F_A0=10, X=X_total_series, k=k_multi, C_A0=1, v0=10
)
print(f"\nSingle CSTR volume for {X_total_series*100:.1f}% conversion: {V_single:.2f} L")
print(f"Multiple CSTRs: {N_cstrs} reactors × {tau_single * 10:.1f} L = {N_cstrs * tau_single * 10:.1f} L total")

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
