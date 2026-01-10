"""
Fluid Mechanics Formulas for Chemical Engineering
Comprehensive collection of classes for fluid mechanics calculations
"""

import math


class FluidProperties:
    """
    Class for calculating basic fluid properties
    """
    
    @staticmethod
    def dynamic_viscosity_sutherland(T, T_ref=273.15, mu_ref=1.716e-5, S=110.4):
        """
        Calculate dynamic viscosity using Sutherland's equation (for gases).
        
        Parameters:
        T: Temperature (K)
        T_ref: Reference temperature (K), default 273.15 K
        mu_ref: Reference viscosity at T_ref (Pa·s), default for air
        S: Sutherland constant (K), default 110.4 K for air
        
        Returns:
        Dynamic viscosity (Pa·s)
        """
        return mu_ref * ((T / T_ref) ** 1.5) * ((T_ref + S) / (T + S))
    
    @staticmethod
    def kinematic_viscosity(mu, rho):
        """
        Calculate kinematic viscosity.
        
        Parameters:
        mu: Dynamic viscosity (Pa·s)
        rho: Density (kg/m³)
        
        Returns:
        Kinematic viscosity (m²/s)
        """
        return mu / rho
    
    @staticmethod
    def density_ideal_gas(P, T, MW, R=8.314462618):
        """
        Calculate density using ideal gas law.
        
        Parameters:
        P: Pressure (Pa)
        T: Temperature (K)
        MW: Molecular weight (kg/kmol)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Density (kg/m³)
        """
        return (P * MW) / (R * T)
    
    @staticmethod
    def density_liquid(T, rho_ref, beta, T_ref=293.15):
        """
        Calculate liquid density with temperature correction.
        
        Parameters:
        T: Temperature (K)
        rho_ref: Reference density at T_ref (kg/m³)
        beta: Thermal expansion coefficient (1/K)
        T_ref: Reference temperature (K), default 293.15 K
        
        Returns:
        Density (kg/m³)
        """
        return rho_ref * (1 - beta * (T - T_ref))
    
    @staticmethod
    def compressibility_factor_ideal():
        """
        Compressibility factor for ideal gas.
        
        Returns:
        Z = 1 (for ideal gas)
        """
        return 1.0
    
    @staticmethod
    def specific_gravity(rho, rho_ref=1000.0):
        """
        Calculate specific gravity.
        
        Parameters:
        rho: Density of fluid (kg/m³)
        rho_ref: Reference density, default 1000 kg/m³ (water)
        
        Returns:
        Specific gravity (dimensionless)
        """
        return rho / rho_ref
    
    @staticmethod
    def surface_tension_water(T):
        """
        Estimate surface tension of water as a function of temperature.
        
        Parameters:
        T: Temperature (K)
        
        Returns:
        Surface tension (N/m)
        """
        # Empirical correlation for water
        T_c = 647.096  # Critical temperature of water (K)
        return 0.2358 * ((T_c - T) / T_c) ** 1.256 * (1 - 0.625 * (T_c - T) / T_c)


class PipeFlow:
    """
    Class for pipe flow calculations
    """
    
    @staticmethod
    def reynolds_number(rho, v, D, mu):
        """
        Calculate Reynolds number.
        
        Parameters:
        rho: Density (kg/m³)
        v: Velocity (m/s)
        D: Diameter (m)
        mu: Dynamic viscosity (Pa·s)
        
        Returns:
        Reynolds number (dimensionless)
        """
        return (rho * v * D) / mu
    
    @staticmethod
    def reynolds_number_mass_flow(m_dot, D, mu):
        """
        Calculate Reynolds number from mass flow rate.
        
        Parameters:
        m_dot: Mass flow rate (kg/s)
        D: Diameter (m)
        mu: Dynamic viscosity (Pa·s)
        
        Returns:
        Reynolds number (dimensionless)
        """
        return (4 * m_dot) / (math.pi * D * mu)
    
    @staticmethod
    def friction_factor_colebrook(Re, epsilon, D, max_iter=100, tol=1e-6):
        """
        Calculate friction factor using Colebrook equation (turbulent flow).
        
        Parameters:
        Re: Reynolds number
        epsilon: Pipe roughness (m)
        D: Pipe diameter (m)
        max_iter: Maximum iterations
        tol: Tolerance for convergence
        
        Returns:
        Friction factor (dimensionless)
        """
        if Re < 2300:
            return PipeFlow.friction_factor_laminar(Re)
        
        relative_roughness = epsilon / D
        f = 0.01  # Initial guess
        
        for _ in range(max_iter):
            f_new = (-2 * math.log10(relative_roughness / 3.7 + 2.51 / (Re * math.sqrt(f)))) ** (-2)
            if abs(f_new - f) < tol:
                return f_new
            f = f_new
        
        return f
    
    @staticmethod
    def friction_factor_laminar(Re):
        """
        Calculate friction factor for laminar flow (Re < 2300).
        
        Parameters:
        Re: Reynolds number
        
        Returns:
        Friction factor (dimensionless)
        """
        if Re == 0:
            return float('inf')
        return 64 / Re
    
    @staticmethod
    def friction_factor_swamee_jain(Re, epsilon, D):
        """
        Calculate friction factor using Swamee-Jain equation (explicit).
        
        Parameters:
        Re: Reynolds number
        epsilon: Pipe roughness (m)
        D: Pipe diameter (m)
        
        Returns:
        Friction factor (dimensionless)
        """
        if Re < 2300:
            return PipeFlow.friction_factor_laminar(Re)
        
        relative_roughness = epsilon / D
        return 0.25 / (math.log10(relative_roughness / 3.7 + 5.74 / (Re ** 0.9))) ** 2
    
    @staticmethod
    def pressure_drop_darcy_weisbach(f, L, D, rho, v):
        """
        Calculate pressure drop using Darcy-Weisbach equation.
        
        Parameters:
        f: Friction factor
        L: Pipe length (m)
        D: Pipe diameter (m)
        rho: Density (kg/m³)
        v: Velocity (m/s)
        
        Returns:
        Pressure drop (Pa)
        """
        return f * (L / D) * (rho * v ** 2) / 2
    
    @staticmethod
    def pressure_drop_hagen_poiseuille(mu, L, Q, D):
        """
        Calculate pressure drop for laminar flow (Hagen-Poiseuille equation).
        
        Parameters:
        mu: Dynamic viscosity (Pa·s)
        L: Pipe length (m)
        Q: Volumetric flow rate (m³/s)
        D: Pipe diameter (m)
        
        Returns:
        Pressure drop (Pa)
        """
        return (128 * mu * L * Q) / (math.pi * D ** 4)
    
    @staticmethod
    def velocity_from_flow_rate(Q, D):
        """
        Calculate average velocity from volumetric flow rate.
        
        Parameters:
        Q: Volumetric flow rate (m³/s)
        D: Pipe diameter (m)
        
        Returns:
        Velocity (m/s)
        """
        return (4 * Q) / (math.pi * D ** 2)
    
    @staticmethod
    def flow_rate_from_velocity(v, D):
        """
        Calculate volumetric flow rate from velocity.
        
        Parameters:
        v: Velocity (m/s)
        D: Pipe diameter (m)
        
        Returns:
        Volumetric flow rate (m³/s)
        """
        return (math.pi * D ** 2 * v) / 4
    
    @staticmethod
    def head_loss(f, L, D, v, g=9.81):
        """
        Calculate head loss.
        
        Parameters:
        f: Friction factor
        L: Pipe length (m)
        D: Pipe diameter (m)
        v: Velocity (m/s)
        g: Gravitational acceleration (m/s²)
        
        Returns:
        Head loss (m)
        """
        return f * (L / D) * (v ** 2) / (2 * g)
    
    @staticmethod
    def equivalent_length_minor_losses(K, D, f):
        """
        Calculate equivalent length for minor losses.
        
        Parameters:
        K: Loss coefficient
        D: Pipe diameter (m)
        f: Friction factor
        
        Returns:
        Equivalent length (m)
        """
        return (K * D) / f


class PumpCalculations:
    """
    Class for pump calculations
    """
    
    @staticmethod
    def pump_head(P_out, P_in, rho, g=9.81, z_out=0, z_in=0, v_out=0, v_in=0):
        """
        Calculate pump head using Bernoulli equation.
        
        Parameters:
        P_out: Outlet pressure (Pa)
        P_in: Inlet pressure (Pa)
        rho: Density (kg/m³)
        g: Gravitational acceleration (m/s²)
        z_out: Outlet elevation (m)
        z_in: Inlet elevation (m)
        v_out: Outlet velocity (m/s)
        v_in: Inlet velocity (m/s)
        
        Returns:
        Pump head (m)
        """
        pressure_head = (P_out - P_in) / (rho * g)
        elevation_head = z_out - z_in
        velocity_head = (v_out ** 2 - v_in ** 2) / (2 * g)
        return pressure_head + elevation_head + velocity_head
    
    @staticmethod
    def pump_power(Q, H, rho, g=9.81, eta=1.0):
        """
        Calculate pump power.
        
        Parameters:
        Q: Volumetric flow rate (m³/s)
        H: Pump head (m)
        rho: Density (kg/m³)
        g: Gravitational acceleration (m/s²)
        eta: Pump efficiency (0-1)
        
        Returns:
        Pump power (W)
        """
        return (Q * rho * g * H) / eta
    
    @staticmethod
    def npsh_available(P_atm, P_vapor, rho, g=9.81, z_suction=0, h_friction=0):
        """
        Calculate Net Positive Suction Head available.
        
        Parameters:
        P_atm: Atmospheric pressure (Pa)
        P_vapor: Vapor pressure of fluid (Pa)
        rho: Density (kg/m³)
        g: Gravitational acceleration (m/s²)
        z_suction: Suction elevation (positive if above pump) (m)
        h_friction: Friction head loss in suction line (m)
        
        Returns:
        NPSH available (m)
        """
        return ((P_atm - P_vapor) / (rho * g)) + z_suction - h_friction
    
    @staticmethod
    def affinity_law_flow_rate(Q1, Q2, N1, N2):
        """
        Affinity law: Flow rate is proportional to speed.
        
        Parameters:
        Q1: Flow rate at speed N1 (m³/s)
        Q2: Flow rate at speed N2 (m³/s)
        N1: Speed 1 (rpm)
        N2: Speed 2 (rpm)
        
        Returns:
        True if relationship is maintained
        """
        return abs(Q2 / Q1 - N2 / N1) < 1e-6
    
    @staticmethod
    def affinity_law_head(H1, H2, N1, N2):
        """
        Affinity law: Head is proportional to speed squared.
        
        Parameters:
        H1: Head at speed N1 (m)
        H2: Head at speed N2 (m)
        N1: Speed 1 (rpm)
        N2: Speed 2 (rpm)
        
        Returns:
        Head at speed N2 (m)
        """
        return H1 * (N2 / N1) ** 2
    
    @staticmethod
    def affinity_law_power(P1, P2, N1, N2):
        """
        Affinity law: Power is proportional to speed cubed.
        
        Parameters:
        P1: Power at speed N1 (W)
        P2: Power at speed N2 (W)
        N1: Speed 1 (rpm)
        N2: Speed 2 (rpm)
        
        Returns:
        Power at speed N2 (W)
        """
        return P1 * (N2 / N1) ** 3


class FlowMeasurement:
    """
    Class for flow measurement device calculations
    """
    
    @staticmethod
    def orifice_plate_flow_rate(C_d, A_o, rho, P1, P2):
        """
        Calculate flow rate through an orifice plate.
        
        Parameters:
        C_d: Discharge coefficient (typically 0.6-0.7)
        A_o: Orifice area (m²)
        rho: Density (kg/m³)
        P1: Upstream pressure (Pa)
        P2: Downstream pressure (Pa)
        
        Returns:
        Volumetric flow rate (m³/s)
        """
        delta_P = P1 - P2
        return C_d * A_o * math.sqrt((2 * delta_P) / rho)
    
    @staticmethod
    def venturi_flow_rate(C_v, A_t, rho, P1, P2):
        """
        Calculate flow rate through a Venturi meter.
        
        Parameters:
        C_v: Venturi coefficient (typically 0.95-0.98)
        A_t: Throat area (m²)
        rho: Density (kg/m³)
        P1: Upstream pressure (Pa)
        P2: Throat pressure (Pa)
        
        Returns:
        Volumetric flow rate (m³/s)
        """
        delta_P = P1 - P2
        return C_v * A_t * math.sqrt((2 * delta_P) / rho)
    
    @staticmethod
    def pitot_tube_velocity(P_static, P_total, rho):
        """
        Calculate velocity from Pitot tube readings.
        
        Parameters:
        P_static: Static pressure (Pa)
        P_total: Total/stagnation pressure (Pa)
        rho: Density (kg/m³)
        
        Returns:
        Velocity (m/s)
        """
        return math.sqrt((2 * (P_total - P_static)) / rho)
    
    @staticmethod
    def rotameter_flow_rate(C, A, rho_float, rho_fluid, V_float, g=9.81):
        """
        Calculate flow rate for rotameter (variable area flowmeter).
        
        Parameters:
        C: Discharge coefficient
        A: Annular area between float and tube (m²)
        rho_float: Density of float (kg/m³)
        rho_fluid: Density of fluid (kg/m³)
        V_float: Volume of float (m³)
        g: Gravitational acceleration (m/s²)
        
        Returns:
        Volumetric flow rate (m³/s)
        """
        return C * A * math.sqrt((2 * V_float * (rho_float - rho_fluid) * g) / (rho_fluid * A))


class FluidStatics:
    """
    Class for fluid statics calculations
    """
    
    @staticmethod
    def hydrostatic_pressure(rho, h, g=9.81, P_atm=101325):
        """
        Calculate hydrostatic pressure.
        
        Parameters:
        rho: Density (kg/m³)
        h: Depth/height (m)
        g: Gravitational acceleration (m/s²)
        P_atm: Atmospheric pressure (Pa), default 101325 Pa
        
        Returns:
        Pressure (Pa)
        """
        return P_atm + rho * g * h
    
    @staticmethod
    def manometer_pressure_difference(rho_manometer, rho_fluid, h, g=9.81):
        """
        Calculate pressure difference from manometer reading.
        
        Parameters:
        rho_manometer: Density of manometer fluid (kg/m³)
        rho_fluid: Density of fluid being measured (kg/m³)
        h: Height difference in manometer (m)
        g: Gravitational acceleration (m/s²)
        
        Returns:
        Pressure difference (Pa)
        """
        return (rho_manometer - rho_fluid) * g * h
    
    @staticmethod
    def buoyant_force(rho_fluid, V_displaced, g=9.81):
        """
        Calculate buoyant force (Archimedes' principle).
        
        Parameters:
        rho_fluid: Density of fluid (kg/m³)
        V_displaced: Volume of displaced fluid (m³)
        g: Gravitational acceleration (m/s²)
        
        Returns:
        Buoyant force (N)
        """
        return rho_fluid * V_displaced * g
    
    @staticmethod
    def center_of_pressure_rectangle(h, b, theta):
        """
        Calculate center of pressure for rectangular surface.
        
        Parameters:
        h: Depth to top of surface (m)
        b: Width of surface (m)
        theta: Angle of surface from horizontal (radians)
        
        Returns:
        Depth to center of pressure (m)
        """
        return h + (b / (2 * math.sin(theta)))
    
    @staticmethod
    def force_on_submerged_surface(rho, g, A, h_c, theta):
        """
        Calculate force on submerged plane surface.
        
        Parameters:
        rho: Density (kg/m³)
        g: Gravitational acceleration (m/s²)
        A: Area of surface (m²)
        h_c: Depth to centroid (m)
        theta: Angle of surface from horizontal (radians)
        
        Returns:
        Force (N)
        """
        return rho * g * A * h_c * math.sin(theta)


class DimensionalAnalysis:
    """
    Class for dimensionless numbers used in fluid mechanics
    """
    
    @staticmethod
    def reynolds_number(rho, v, L, mu):
        """
        Calculate Reynolds number.
        
        Parameters:
        rho: Density (kg/m³)
        v: Velocity (m/s)
        L: Characteristic length (m)
        mu: Dynamic viscosity (Pa·s)
        
        Returns:
        Reynolds number (dimensionless)
        """
        return (rho * v * L) / mu
    
    @staticmethod
    def froude_number(v, L, g=9.81):
        """
        Calculate Froude number.
        
        Parameters:
        v: Velocity (m/s)
        L: Characteristic length (m)
        g: Gravitational acceleration (m/s²)
        
        Returns:
        Froude number (dimensionless)
        """
        return v / math.sqrt(g * L)
    
    @staticmethod
    def weber_number(rho, v, L, sigma):
        """
        Calculate Weber number.
        
        Parameters:
        rho: Density (kg/m³)
        v: Velocity (m/s)
        L: Characteristic length (m)
        sigma: Surface tension (N/m)
        
        Returns:
        Weber number (dimensionless)
        """
        return (rho * v ** 2 * L) / sigma
    
    @staticmethod
    def mach_number(v, c):
        """
        Calculate Mach number.
        
        Parameters:
        v: Velocity (m/s)
        c: Speed of sound (m/s)
        
        Returns:
        Mach number (dimensionless)
        """
        return v / c
    
    @staticmethod
    def speed_of_sound_gas(gamma, R, T, MW):
        """
        Calculate speed of sound in ideal gas.
        
        Parameters:
        gamma: Specific heat ratio (Cp/Cv)
        R: Universal gas constant (J/(mol·K))
        T: Temperature (K)
        MW: Molecular weight (kg/kmol)
        
        Returns:
        Speed of sound (m/s)
        """
        return math.sqrt((gamma * R * T) / MW)
    
    @staticmethod
    def euler_number(P1, P2, rho, v):
        """
        Calculate Euler number.
        
        Parameters:
        P1: Pressure 1 (Pa)
        P2: Pressure 2 (Pa)
        rho: Density (kg/m³)
        v: Velocity (m/s)
        
        Returns:
        Euler number (dimensionless)
        """
        return (P1 - P2) / (rho * v ** 2)
    
    @staticmethod
    def strouhal_number(f, L, v):
        """
        Calculate Strouhal number.
        
        Parameters:
        f: Frequency (Hz)
        L: Characteristic length (m)
        v: Velocity (m/s)
        
        Returns:
        Strouhal number (dimensionless)
        """
        return (f * L) / v
    
    @staticmethod
    def prandtl_number(Cp, mu, k):
        """
        Calculate Prandtl number.
        
        Parameters:
        Cp: Specific heat at constant pressure (J/(kg·K))
        mu: Dynamic viscosity (Pa·s)
        k: Thermal conductivity (W/(m·K))
        
        Returns:
        Prandtl number (dimensionless)
        """
        return (Cp * mu) / k


class TwoPhaseFlow:
    """
    Class for two-phase flow calculations
    """
    
    @staticmethod
    def void_fraction(alpha_G, rho_G, rho_L, x):
        """
        Calculate void fraction (homogeneous model).
        
        Parameters:
        alpha_G: Void fraction (to be calculated)
        rho_G: Gas density (kg/m³)
        rho_L: Liquid density (kg/m³)
        x: Quality (vapor mass fraction)
        
        Returns:
        Void fraction (dimensionless)
        """
        # Homogeneous model
        return 1 / (1 + ((1 - x) / x) * (rho_G / rho_L))
    
    @staticmethod
    def lockhart_martinelli_parameter(x, rho_G, rho_L, mu_G, mu_L):
        """
        Calculate Lockhart-Martinelli parameter.
        
        Parameters:
        x: Quality (vapor mass fraction)
        rho_G: Gas density (kg/m³)
        rho_L: Liquid density (kg/m³)
        mu_G: Gas viscosity (Pa·s)
        mu_L: Liquid viscosity (Pa·s)
        
        Returns:
        Lockhart-Martinelli parameter (dimensionless)
        """
        return ((1 - x) / x) ** 0.9 * (rho_G / rho_L) ** 0.5 * (mu_L / mu_G) ** 0.1
    
    @staticmethod
    def two_phase_density(rho_G, rho_L, alpha):
        """
        Calculate two-phase density.
        
        Parameters:
        rho_G: Gas density (kg/m³)
        rho_L: Liquid density (kg/m³)
        alpha: Void fraction
        
        Returns:
        Two-phase density (kg/m³)
        """
        return alpha * rho_G + (1 - alpha) * rho_L


# Convenience functions (maintaining backward compatibility with original code)
def viscosity(T, MW, sigma, epsilon):
    """
    Calculate the viscosity of a fluid using the Sutherland equation.
    """
    return FluidProperties.dynamic_viscosity_sutherland(T)

def density(T, MW, sigma, epsilon):
    """
    Calculate the density of a fluid using the ideal gas law.
    Note: This function signature doesn't match ideal gas law requirements.
    Consider using FluidProperties.density_ideal_gas(P, T, MW) instead.
    """
    return MW / (28.97 * 8.31446261815324 * T)

def compressibility(T, MW, sigma, epsilon):
    """
    Calculate the compressibility of a fluid using the ideal gas law.
    """
    return 1.0 / (28.97 * 8.31446261815324 * T)
