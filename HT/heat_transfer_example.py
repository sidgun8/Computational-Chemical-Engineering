"""
Example: Heat Transfer Calculations
Demonstrates how to use the heat transfer classes from heatTransfer.py
"""

from heatTransfer import (
    Conduction, Convection, Radiation, HeatExchangers,
    TransientHeatTransfer, Fins, DimensionlessNumbers
)

print("=" * 70)
print("HEAT TRANSFER CALCULATIONS - EXAMPLES")
print("=" * 70)

# Example 1: Conduction through a plane wall
print("\nExample 1: Conduction through a plane wall")
print("-" * 70)
k_wall = 0.8  # Thermal conductivity of brick (W/(m·K))
A_wall = 10.0  # Wall area (m²)
T_hot = 300.0  # Hot side temperature (K)
T_cold = 280.0  # Cold side temperature (K)
L_wall = 0.2  # Wall thickness (m)

Q_wall = Conduction.plane_wall_heat_rate(k=k_wall, A=A_wall, T1=T_hot, T2=T_cold, L=L_wall)
R_wall = Conduction.plane_wall_thermal_resistance(L=L_wall, k=k_wall, A=A_wall)

print(f"Thermal conductivity: {k_wall} W/(m·K)")
print(f"Wall area: {A_wall} m²")
print(f"Temperature difference: {T_hot - T_cold} K")
print(f"Wall thickness: {L_wall} m")
print(f"Heat transfer rate: {Q_wall:.2f} W")
print(f"Thermal resistance: {R_wall:.4f} K/W")

# Example 2: Convection from a flat plate
print("\nExample 2: Forced convection over a flat plate")
print("-" * 70)
h = 50.0  # Heat transfer coefficient (W/(m²·K))
A_plate = 2.0  # Plate area (m²)
T_surface = 350.0  # Surface temperature (K)
T_fluid = 300.0  # Fluid bulk temperature (K)

Q_conv = Convection.newtons_law_cooling(h=h, A=A_plate, T_s=T_surface, T_inf=T_fluid)
Nu = Convection.nusselt_number(h=h, L=1.0, k=0.025)  # Assuming air properties

print(f"Heat transfer coefficient: {h} W/(m²·K)")
print(f"Surface area: {A_plate} m²")
print(f"Temperature difference: {T_surface - T_fluid} K")
print(f"Heat transfer rate: {Q_conv:.2f} W")
print(f"Nusselt number: {Nu:.2f}")

# Example 3: Radiation heat transfer
print("\nExample 3: Radiation from a blackbody surface")
print("-" * 70)
epsilon = 0.9  # Emissivity
A_rad = 1.0  # Surface area (m²)
T_rad = 500.0  # Surface temperature (K)

E = Radiation.stefan_boltzmann_emissive_power(epsilon=epsilon, T=T_rad)
Q_rad = Radiation.stefan_boltzmann_heat_rate(epsilon=epsilon, A=A_rad, T=T_rad)

print(f"Emissivity: {epsilon}")
print(f"Surface area: {A_rad} m²")
print(f"Surface temperature: {T_rad} K")
print(f"Emissive power: {E:.2f} W/m²")
print(f"Heat transfer rate: {Q_rad:.2f} W")

# Example 4: Heat exchanger - LMTD method
print("\nExample 4: Heat exchanger using LMTD method")
print("-" * 70)
T_hot_in = 360.0  # Hot fluid inlet (K)
T_hot_out = 320.0  # Hot fluid outlet (K)
T_cold_in = 280.0  # Cold fluid inlet (K)
T_cold_out = 310.0  # Cold fluid outlet (K)
U = 500.0  # Overall heat transfer coefficient (W/(m²·K))
A_hx = 5.0  # Heat transfer area (m²)

LMTD = HeatExchangers.log_mean_temperature_difference(
    T_hot_in=T_hot_in, T_hot_out=T_hot_out,
    T_cold_in=T_cold_in, T_cold_out=T_cold_out
)
Q_hx = HeatExchangers.heat_exchanger_heat_rate(U=U, A=A_hx, LMTD=LMTD)

print(f"Hot fluid: {T_hot_in} K → {T_hot_out} K")
print(f"Cold fluid: {T_cold_in} K → {T_cold_out} K")
print(f"LMTD: {LMTD:.2f} K")
print(f"Overall heat transfer coefficient: {U} W/(m²·K)")
print(f"Heat transfer area: {A_hx} m²")
print(f"Heat transfer rate: {Q_hx:.2f} W")

# Example 5: Heat exchanger - NTU method
print("\nExample 5: Heat exchanger using NTU-effectiveness method")
print("-" * 70)
m_dot_hot = 2.0  # Hot fluid mass flow rate (kg/s)
m_dot_cold = 3.0  # Cold fluid mass flow rate (kg/s)
Cp_hot = 4200.0  # Hot fluid specific heat (J/(kg·K))
Cp_cold = 4180.0  # Cold fluid specific heat (J/(kg·K))

C_hot = HeatExchangers.heat_capacity_rate(m_dot=m_dot_hot, Cp=Cp_hot)
C_cold = HeatExchangers.heat_capacity_rate(m_dot=m_dot_cold, Cp=Cp_cold)
C_min = min(C_hot, C_cold)
C_max = max(C_hot, C_cold)

NTU = HeatExchangers.number_of_transfer_units(U=U, A=A_hx, C_min=C_min)
epsilon = HeatExchangers.effectiveness_ntu_method(
    C_min=C_min, C_max=C_max, NTU=NTU, flow_type='counterflow'
)
Q_ntu = HeatExchangers.heat_rate_from_effectiveness(
    epsilon=epsilon, C_min=C_min, T_hot_in=T_hot_in, T_cold_in=T_cold_in
)

print(f"Hot fluid capacity rate: {C_hot:.2f} W/K")
print(f"Cold fluid capacity rate: {C_cold:.2f} W/K")
print(f"NTU: {NTU:.3f}")
print(f"Effectiveness: {epsilon:.3f}")
print(f"Heat transfer rate (NTU method): {Q_ntu:.2f} W")

# Example 6: Dimensionless numbers
print("\nExample 6: Dimensionless numbers for heat transfer")
print("-" * 70)
Cp = 1005.0  # Specific heat of air (J/(kg·K))
mu = 1.81e-5  # Dynamic viscosity of air (Pa·s)
k = 0.025  # Thermal conductivity of air (W/(m·K))
Re = 50000  # Reynolds number
Pr = DimensionlessNumbers.prandtl_number(Cp=Cp, mu=mu, k=k)

# Calculate Nusselt number using Dittus-Boelter correlation
Nu_pipe = Convection.pipe_flow_dittus_boelter(Re=Re, Pr=Pr, heating=True)
h_pipe = Convection.heat_transfer_coefficient_from_nu(Nu=Nu_pipe, k=k, L=0.1)

print(f"Reynolds number: {Re}")
print(f"Prandtl number: {Pr:.3f}")
print(f"Nusselt number (Dittus-Boelter): {Nu_pipe:.2f}")
print(f"Heat transfer coefficient: {h_pipe:.2f} W/(m²·K)")

# Example 7: Natural convection
print("\nExample 7: Natural convection on vertical plate")
print("-" * 70)
g = 9.81  # Gravitational acceleration (m/s²)
beta = 1/300  # Thermal expansion coefficient (1/K) - approximate for air
T_s = 350.0  # Surface temperature (K)
T_inf = 300.0  # Fluid temperature (K)
L = 0.5  # Characteristic length (m)
nu = 1.5e-5  # Kinematic viscosity (m²/s)

Gr = DimensionlessNumbers.grashof_number(
    g=g, beta=beta, T_s=T_s, T_inf=T_inf, L=L, nu=nu
)
Ra = DimensionlessNumbers.rayleigh_number(Pr=Pr, Gr=Gr)
Nu_natural = Convection.natural_convection_vertical_plate(Pr=Pr, Gr=Gr)
h_natural = Convection.heat_transfer_coefficient_from_nu(Nu=Nu_natural, k=k, L=L)

print(f"Grashof number: {Gr:.2e}")
print(f"Rayleigh number: {Ra:.2e}")
print(f"Nusselt number (natural convection): {Nu_natural:.2f}")
print(f"Heat transfer coefficient: {h_natural:.2f} W/(m²·K)")

# Example 8: Transient heat transfer - Lumped capacitance
print("\nExample 8: Transient heat transfer (lumped capacitance)")
print("-" * 70)
T_i = 400.0  # Initial temperature (K)
T_inf = 300.0  # Fluid temperature (K)
h_transient = 25.0  # Heat transfer coefficient (W/(m²·K))
A_transient = 0.1  # Surface area (m²)
rho = 7800.0  # Density of steel (kg/m³)
V = 0.001  # Volume (m³)
Cp_steel = 500.0  # Specific heat of steel (J/(kg·K))
t = 60.0  # Time (s)

# Check Biot number
Bi = TransientHeatTransfer.biot_number(h=h_transient, L=V/A_transient, k=50.0)
print(f"Biot number: {Bi:.4f}")

if Bi < 0.1:
    T_t = TransientHeatTransfer.lumped_capacitance_temperature(
        T_i=T_i, T_inf=T_inf, h=h_transient, A=A_transient,
        rho=rho, V=V, Cp=Cp_steel, t=t
    )
    print(f"Initial temperature: {T_i} K")
    print(f"Fluid temperature: {T_inf} K")
    print(f"Time: {t} s")
    print(f"Temperature at time t: {T_t:.2f} K")
    print(f"Temperature drop: {T_i - T_t:.2f} K")
else:
    print("Biot number > 0.1, lumped capacitance method not valid")

# Example 9: Fin heat transfer
print("\nExample 9: Heat transfer from a rectangular fin")
print("-" * 70)
h_fin = 30.0  # Heat transfer coefficient (W/(m²·K))
P_fin = 0.2  # Perimeter (m)
k_fin = 200.0  # Thermal conductivity of fin material (W/(m·K))
A_c_fin = 0.001  # Cross-sectional area (m²)
T_b_fin = 400.0  # Base temperature (K)
T_inf_fin = 300.0  # Fluid temperature (K)
L_fin = 0.1  # Fin length (m)

Q_fin = Fins.fin_heat_rate_rectangular(
    h=h_fin, P=P_fin, k=k_fin, A_c=A_c_fin,
    T_b=T_b_fin, T_inf=T_inf_fin, L=L_fin
)
eta_fin = Fins.fin_efficiency_rectangular(
    m=Fins.fin_parameter(h=h_fin, P=P_fin, k=k_fin, A_c=A_c_fin),
    L=L_fin
)

print(f"Fin length: {L_fin} m")
print(f"Base temperature: {T_b_fin} K")
print(f"Fluid temperature: {T_inf_fin} K")
print(f"Heat transfer rate from fin: {Q_fin:.2f} W")
print(f"Fin efficiency: {eta_fin:.3f}")

# Example 10: Composite wall
print("\nExample 10: Heat transfer through composite wall")
print("-" * 70)
# Three-layer wall: brick, insulation, brick
k1 = 0.8  # Thermal conductivity layer 1 (W/(m·K))
k2 = 0.04  # Thermal conductivity layer 2 - insulation (W/(m·K))
k3 = 0.8  # Thermal conductivity layer 3 (W/(m·K))
L1 = 0.1  # Thickness layer 1 (m)
L2 = 0.05  # Thickness layer 2 (m)
L3 = 0.1  # Thickness layer 3 (m)
A_comp = 10.0  # Area (m²)
T_hot_comp = 350.0  # Hot side (K)
T_cold_comp = 280.0  # Cold side (K)

R1 = Conduction.plane_wall_thermal_resistance(L=L1, k=k1, A=A_comp)
R2 = Conduction.plane_wall_thermal_resistance(L=L2, k=k2, A=A_comp)
R3 = Conduction.plane_wall_thermal_resistance(L=L3, k=k3, A=A_comp)
R_total = R1 + R2 + R3

Q_comp = Conduction.composite_wall_heat_rate(
    T_hot=T_hot_comp, T_cold=T_cold_comp, R_total=R_total
)

print(f"Layer 1 (brick): L={L1} m, k={k1} W/(m·K), R={R1:.4f} K/W")
print(f"Layer 2 (insulation): L={L2} m, k={k2} W/(m·K), R={R2:.4f} K/W")
print(f"Layer 3 (brick): L={L3} m, k={k3} W/(m·K), R={R3:.4f} K/W")
print(f"Total thermal resistance: {R_total:.4f} K/W")
print(f"Heat transfer rate: {Q_comp:.2f} W")

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)

