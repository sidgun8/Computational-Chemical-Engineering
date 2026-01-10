"""
Heat Transfer Formulas for Chemical Engineering
Comprehensive collection of classes for heat transfer calculations
"""

import math


class Conduction:
    """
    Class for conduction heat transfer calculations
    """
    
    @staticmethod
    def fourier_law_heat_flux(k, dT_dx):
        """
        Calculate heat flux using Fourier's law.
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        dT_dx: Temperature gradient (K/m)
        
        Returns:
        Heat flux (W/m²)
        """
        return -k * dT_dx
    
    @staticmethod
    def fourier_law_heat_rate(k, A, dT_dx):
        """
        Calculate heat transfer rate using Fourier's law.
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        A: Cross-sectional area (m²)
        dT_dx: Temperature gradient (K/m)
        
        Returns:
        Heat transfer rate (W)
        """
        return -k * A * dT_dx
    
    @staticmethod
    def plane_wall_heat_rate(k, A, T1, T2, L):
        """
        Calculate heat transfer rate through a plane wall.
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        A: Area (m²)
        T1: Temperature at surface 1 (K)
        T2: Temperature at surface 2 (K)
        L: Wall thickness (m)
        
        Returns:
        Heat transfer rate (W)
        """
        return (k * A * (T1 - T2)) / L
    
    @staticmethod
    def plane_wall_thermal_resistance(L, k, A):
        """
        Calculate thermal resistance for a plane wall.
        
        Parameters:
        L: Wall thickness (m)
        k: Thermal conductivity (W/(m·K))
        A: Area (m²)
        
        Returns:
        Thermal resistance (K/W)
        """
        return L / (k * A)
    
    @staticmethod
    def cylindrical_wall_heat_rate(k, L, T1, T2, r1, r2):
        """
        Calculate heat transfer rate through a cylindrical wall.
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        L: Length (m)
        T1: Temperature at inner surface (K)
        T2: Temperature at outer surface (K)
        r1: Inner radius (m)
        r2: Outer radius (m)
        
        Returns:
        Heat transfer rate (W)
        """
        return (2 * math.pi * k * L * (T1 - T2)) / math.log(r2 / r1)
    
    @staticmethod
    def cylindrical_wall_thermal_resistance(r1, r2, k, L):
        """
        Calculate thermal resistance for a cylindrical wall.
        
        Parameters:
        r1: Inner radius (m)
        r2: Outer radius (m)
        k: Thermal conductivity (W/(m·K))
        L: Length (m)
        
        Returns:
        Thermal resistance (K/W)
        """
        return math.log(r2 / r1) / (2 * math.pi * k * L)
    
    @staticmethod
    def spherical_wall_heat_rate(k, T1, T2, r1, r2):
        """
        Calculate heat transfer rate through a spherical wall.
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        T1: Temperature at inner surface (K)
        T2: Temperature at outer surface (K)
        r1: Inner radius (m)
        r2: Outer radius (m)
        
        Returns:
        Heat transfer rate (W)
        """
        return (4 * math.pi * k * (T1 - T2)) / ((1 / r1) - (1 / r2))
    
    @staticmethod
    def spherical_wall_thermal_resistance(r1, r2, k):
        """
        Calculate thermal resistance for a spherical wall.
        
        Parameters:
        r1: Inner radius (m)
        r2: Outer radius (m)
        k: Thermal conductivity (W/(m·K))
        
        Returns:
        Thermal resistance (K/W)
        """
        return ((1 / r1) - (1 / r2)) / (4 * math.pi * k)
    
    @staticmethod
    def composite_wall_heat_rate(T_hot, T_cold, R_total):
        """
        Calculate heat transfer rate through composite wall.
        
        Parameters:
        T_hot: Hot side temperature (K)
        T_cold: Cold side temperature (K)
        R_total: Total thermal resistance (K/W)
        
        Returns:
        Heat transfer rate (W)
        """
        return (T_hot - T_cold) / R_total
    
    @staticmethod
    def contact_resistance(R_c, A):
        """
        Calculate contact resistance per unit area.
        
        Parameters:
        R_c: Contact resistance (K/W)
        A: Contact area (m²)
        
        Returns:
        Contact resistance per unit area (m²·K/W)
        """
        return R_c * A


class Convection:
    """
    Class for convection heat transfer calculations
    """
    
    @staticmethod
    def newtons_law_cooling(h, A, T_s, T_inf):
        """
        Calculate heat transfer rate using Newton's law of cooling.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        A: Surface area (m²)
        T_s: Surface temperature (K)
        T_inf: Fluid bulk temperature (K)
        
        Returns:
        Heat transfer rate (W)
        """
        return h * A * (T_s - T_inf)
    
    @staticmethod
    def convective_thermal_resistance(h, A):
        """
        Calculate convective thermal resistance.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        A: Surface area (m²)
        
        Returns:
        Thermal resistance (K/W)
        """
        return 1 / (h * A)
    
    @staticmethod
    def nusselt_number(h, L, k):
        """
        Calculate Nusselt number.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        L: Characteristic length (m)
        k: Thermal conductivity (W/(m·K))
        
        Returns:
        Nusselt number (dimensionless)
        """
        return (h * L) / k
    
    @staticmethod
    def heat_transfer_coefficient_from_nu(Nu, k, L):
        """
        Calculate heat transfer coefficient from Nusselt number.
        
        Parameters:
        Nu: Nusselt number
        k: Thermal conductivity (W/(m·K))
        L: Characteristic length (m)
        
        Returns:
        Heat transfer coefficient (W/(m²·K))
        """
        return (Nu * k) / L
    
    @staticmethod
    def flat_plate_laminar_nusselt(Re, Pr, x_or_L='L'):
        """
        Calculate Nusselt number for laminar flow over flat plate.
        
        Parameters:
        Re: Reynolds number
        Pr: Prandtl number
        x_or_L: 'x' for local, 'L' for average (default 'L')
        
        Returns:
        Nusselt number (dimensionless)
        """
        if x_or_L == 'x':
            # Local Nusselt number
            return 0.332 * (Re ** 0.5) * (Pr ** (1/3))
        else:
            # Average Nusselt number
            return 0.664 * (Re ** 0.5) * (Pr ** (1/3))
    
    @staticmethod
    def flat_plate_turbulent_nusselt(Re, Pr, x_or_L='L'):
        """
        Calculate Nusselt number for turbulent flow over flat plate.
        
        Parameters:
        Re: Reynolds number
        Pr: Prandtl number
        x_or_L: 'x' for local, 'L' for average (default 'L')
        
        Returns:
        Nusselt number (dimensionless)
        """
        if x_or_L == 'x':
            # Local Nusselt number
            return 0.0296 * (Re ** 0.8) * (Pr ** (1/3))
        else:
            # Average Nusselt number (simplified)
            return 0.037 * (Re ** 0.8) * (Pr ** (1/3))
    
    @staticmethod
    def pipe_flow_dittus_boelter(Re, Pr, heating=True):
        """
        Calculate Nusselt number for turbulent flow in pipe (Dittus-Boelter).
        
        Parameters:
        Re: Reynolds number (should be > 10000)
        Pr: Prandtl number (0.7 < Pr < 160)
        heating: True for heating (T_s > T_b), False for cooling
        
        Returns:
        Nusselt number (dimensionless)
        """
        if heating:
            n = 0.4
        else:
            n = 0.3
        return 0.023 * (Re ** 0.8) * (Pr ** n)
    
    @staticmethod
    def pipe_flow_gnielinski(Re, Pr, f):
        """
        Calculate Nusselt number for turbulent flow in pipe (Gnielinski).
        More accurate than Dittus-Boelter for wider range.
        
        Parameters:
        Re: Reynolds number (3000 < Re < 5e6)
        Pr: Prandtl number (0.5 < Pr < 2000)
        f: Friction factor
        
        Returns:
        Nusselt number (dimensionless)
        """
        numerator = (f / 8) * (Re - 1000) * Pr
        denominator = 1 + 12.7 * math.sqrt(f / 8) * ((Pr ** (2/3)) - 1)
        return numerator / denominator
    
    @staticmethod
    def pipe_flow_laminar_constant_wall_temp(Re, Pr, D, L):
        """
        Calculate Nusselt number for laminar flow in pipe (constant wall temp).
        
        Parameters:
        Re: Reynolds number
        Pr: Prandtl number
        D: Diameter (m)
        L: Length (m)
        
        Returns:
        Nusselt number (dimensionless)
        """
        # Fully developed laminar flow, constant wall temperature
        if (L / D) / (Re * Pr) > 0.05:
            # Fully developed
            return 3.66
        else:
            # Entry region (simplified)
            return 3.66 + (0.0668 * (D / L) * Re * Pr) / (1 + 0.04 * ((D / L) * Re * Pr) ** (2/3))
    
    @staticmethod
    def natural_convection_vertical_plate(Pr, Gr):
        """
        Calculate Nusselt number for natural convection on vertical plate.
        
        Parameters:
        Pr: Prandtl number
        Gr: Grashof number
        
        Returns:
        Nusselt number (dimensionless)
        """
        Ra = Gr * Pr  # Rayleigh number
        if Ra < 1e9:
            # Laminar
            return 0.59 * (Ra ** 0.25)
        else:
            # Turbulent
            return 0.1 * (Ra ** (1/3))
    
    @staticmethod
    def natural_convection_horizontal_cylinder(Pr, Gr):
        """
        Calculate Nusselt number for natural convection on horizontal cylinder.
        
        Parameters:
        Pr: Prandtl number
        Gr: Grashof number
        
        Returns:
        Nusselt number (dimensionless)
        """
        Ra = Gr * Pr  # Rayleigh number
        if Ra < 1e12:
            return (0.6 + (0.387 * Ra ** (1/6)) / (1 + (0.559 / Pr) ** (9/16)) ** (8/27)) ** 2
        else:
            return 0.1 * (Ra ** (1/3))


class Radiation:
    """
    Class for radiation heat transfer calculations
    """
    
    @staticmethod
    def stefan_boltzmann_emissive_power(epsilon, T, sigma=5.67e-8):
        """
        Calculate emissive power using Stefan-Boltzmann law.
        
        Parameters:
        epsilon: Emissivity (0-1)
        T: Temperature (K)
        sigma: Stefan-Boltzmann constant (W/(m²·K⁴)), default 5.67e-8
        
        Returns:
        Emissive power (W/m²)
        """
        return epsilon * sigma * (T ** 4)
    
    @staticmethod
    def stefan_boltzmann_heat_rate(epsilon, A, T, sigma=5.67e-8):
        """
        Calculate heat transfer rate using Stefan-Boltzmann law.
        
        Parameters:
        epsilon: Emissivity (0-1)
        A: Surface area (m²)
        T: Temperature (K)
        sigma: Stefan-Boltzmann constant (W/(m²·K⁴)), default 5.67e-8
        
        Returns:
        Heat transfer rate (W)
        """
        return epsilon * sigma * A * (T ** 4)
    
    @staticmethod
    def radiation_heat_exchange(epsilon1, epsilon2, A1, T1, T2, F12=1.0, sigma=5.67e-8):
        """
        Calculate radiation heat exchange between two surfaces.
        
        Parameters:
        epsilon1: Emissivity of surface 1
        epsilon2: Emissivity of surface 2
        A1: Area of surface 1 (m²)
        T1: Temperature of surface 1 (K)
        T2: Temperature of surface 2 (K)
        F12: View factor from surface 1 to surface 2 (default 1.0)
        sigma: Stefan-Boltzmann constant (W/(m²·K⁴)), default 5.67e-8
        
        Returns:
        Heat transfer rate from surface 1 to surface 2 (W)
        """
        return sigma * A1 * F12 * (T1 ** 4 - T2 ** 4) / (1/epsilon1 + (A1/A1) * (1/epsilon2 - 1))
    
    @staticmethod
    def parallel_plates_radiation(epsilon1, epsilon2, A, T1, T2, sigma=5.67e-8):
        """
        Calculate radiation heat exchange between parallel plates.
        
        Parameters:
        epsilon1: Emissivity of plate 1
        epsilon2: Emissivity of plate 2
        A: Area (m²)
        T1: Temperature of plate 1 (K)
        T2: Temperature of plate 2 (K)
        sigma: Stefan-Boltzmann constant (W/(m²·K⁴)), default 5.67e-8
        
        Returns:
        Heat transfer rate (W)
        """
        return (sigma * A * (T1 ** 4 - T2 ** 4)) / (1/epsilon1 + 1/epsilon2 - 1)
    
    @staticmethod
    def concentric_spheres_radiation(epsilon1, epsilon2, r1, r2, T1, T2, sigma=5.67e-8):
        """
        Calculate radiation heat exchange between concentric spheres.
        
        Parameters:
        epsilon1: Emissivity of inner sphere
        epsilon2: Emissivity of outer sphere
        r1: Radius of inner sphere (m)
        r2: Radius of outer sphere (m)
        T1: Temperature of inner sphere (K)
        T2: Temperature of outer sphere (K)
        sigma: Stefan-Boltzmann constant (W/(m²·K⁴)), default 5.67e-8
        
        Returns:
        Heat transfer rate (W)
        """
        A1 = 4 * math.pi * r1 ** 2
        return (sigma * A1 * (T1 ** 4 - T2 ** 4)) / (1/epsilon1 + (r1/r2) ** 2 * (1/epsilon2 - 1))
    
    @staticmethod
    def wien_displacement_law(T):
        """
        Calculate wavelength at which blackbody radiation is maximum (Wien's law).
        
        Parameters:
        T: Temperature (K)
        
        Returns:
        Wavelength (m)
        """
        return 2.898e-3 / T  # b = 2.898e-3 m·K


class HeatExchangers:
    """
    Class for heat exchanger calculations
    """
    
    @staticmethod
    def log_mean_temperature_difference(T_hot_in, T_hot_out, T_cold_in, T_cold_out):
        """
        Calculate log mean temperature difference (LMTD).
        
        Parameters:
        T_hot_in: Hot fluid inlet temperature (K)
        T_hot_out: Hot fluid outlet temperature (K)
        T_cold_in: Cold fluid inlet temperature (K)
        T_cold_out: Cold fluid outlet temperature (K)
        
        Returns:
        LMTD (K)
        """
        delta_T1 = T_hot_in - T_cold_out
        delta_T2 = T_hot_out - T_cold_in
        
        if delta_T1 == delta_T2:
            return delta_T1
        
        if delta_T1 <= 0 or delta_T2 <= 0:
            raise ValueError("LMTD undefined: temperature crossover")
        
        return (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)
    
    @staticmethod
    def heat_exchanger_heat_rate(U, A, LMTD):
        """
        Calculate heat transfer rate in heat exchanger.
        
        Parameters:
        U: Overall heat transfer coefficient (W/(m²·K))
        A: Heat transfer area (m²)
        LMTD: Log mean temperature difference (K)
        
        Returns:
        Heat transfer rate (W)
        """
        return U * A * LMTD
    
    @staticmethod
    def overall_heat_transfer_coefficient(h1, h2, k_wall, L_wall, A1, A2, fouling1=0, fouling2=0):
        """
        Calculate overall heat transfer coefficient.
        
        Parameters:
        h1: Convective heat transfer coefficient side 1 (W/(m²·K))
        h2: Convective heat transfer coefficient side 2 (W/(m²·K))
        k_wall: Thermal conductivity of wall (W/(m·K))
        L_wall: Wall thickness (m)
        A1: Area on side 1 (m²)
        A2: Area on side 2 (m²)
        fouling1: Fouling factor side 1 (m²·K/W), default 0
        fouling2: Fouling factor side 2 (m²·K/W), default 0
        
        Returns:
        Overall heat transfer coefficient based on A1 (W/(m²·K))
        """
        R_total = 1/(h1*A1) + fouling1/A1 + L_wall/(k_wall*A1) + fouling2/A2 + 1/(h2*A2)
        return 1 / (R_total * A1)
    
    @staticmethod
    def effectiveness_ntu_method(C_min, C_max, NTU, flow_type='counterflow'):
        """
        Calculate heat exchanger effectiveness using NTU method.
        
        Parameters:
        C_min: Minimum heat capacity rate (W/K)
        C_max: Maximum heat capacity rate (W/K)
        NTU: Number of transfer units
        flow_type: 'counterflow', 'parallel', or 'crossflow' (default 'counterflow')
        
        Returns:
        Effectiveness (0-1)
        """
        C_r = C_min / C_max
        
        if flow_type == 'counterflow':
            if C_r == 1:
                return NTU / (1 + NTU)
            else:
                numerator = 1 - math.exp(-NTU * (1 - C_r))
                denominator = 1 - C_r * math.exp(-NTU * (1 - C_r))
                return numerator / denominator
        
        elif flow_type == 'parallel':
            numerator = 1 - math.exp(-NTU * (1 + C_r))
            denominator = 1 + C_r
            return numerator / denominator
        
        elif flow_type == 'crossflow':
            # Simplified crossflow (both fluids unmixed)
            return 1 - math.exp((1/C_r) * (NTU ** 0.22) * (math.exp(-C_r * NTU ** 0.78) - 1))
        
        else:
            raise ValueError("flow_type must be 'counterflow', 'parallel', or 'crossflow'")
    
    @staticmethod
    def number_of_transfer_units(U, A, C_min):
        """
        Calculate number of transfer units (NTU).
        
        Parameters:
        U: Overall heat transfer coefficient (W/(m²·K))
        A: Heat transfer area (m²)
        C_min: Minimum heat capacity rate (W/K)
        
        Returns:
        NTU (dimensionless)
        """
        return (U * A) / C_min
    
    @staticmethod
    def heat_capacity_rate(m_dot, Cp):
        """
        Calculate heat capacity rate.
        
        Parameters:
        m_dot: Mass flow rate (kg/s)
        Cp: Specific heat capacity (J/(kg·K))
        
        Returns:
        Heat capacity rate (W/K)
        """
        return m_dot * Cp
    
    @staticmethod
    def heat_rate_from_effectiveness(epsilon, C_min, T_hot_in, T_cold_in):
        """
        Calculate heat transfer rate from effectiveness.
        
        Parameters:
        epsilon: Effectiveness (0-1)
        C_min: Minimum heat capacity rate (W/K)
        T_hot_in: Hot fluid inlet temperature (K)
        T_cold_in: Cold fluid inlet temperature (K)
        
        Returns:
        Heat transfer rate (W)
        """
        return epsilon * C_min * (T_hot_in - T_cold_in)


class TransientHeatTransfer:
    """
    Class for transient/unsteady heat transfer calculations
    """
    
    @staticmethod
    def biot_number(h, L, k):
        """
        Calculate Biot number.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        L: Characteristic length (m)
        k: Thermal conductivity (W/(m·K))
        
        Returns:
        Biot number (dimensionless)
        """
        return (h * L) / k
    
    @staticmethod
    def fourier_number(alpha, t, L):
        """
        Calculate Fourier number.
        
        Parameters:
        alpha: Thermal diffusivity (m²/s)
        t: Time (s)
        L: Characteristic length (m)
        
        Returns:
        Fourier number (dimensionless)
        """
        return (alpha * t) / (L ** 2)
    
    @staticmethod
    def thermal_diffusivity(k, rho, Cp):
        """
        Calculate thermal diffusivity.
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        rho: Density (kg/m³)
        Cp: Specific heat capacity (J/(kg·K))
        
        Returns:
        Thermal diffusivity (m²/s)
        """
        return k / (rho * Cp)
    
    @staticmethod
    def lumped_capacitance_temperature(T_i, T_inf, h, A, rho, V, Cp, t):
        """
        Calculate temperature using lumped capacitance method (Bi < 0.1).
        
        Parameters:
        T_i: Initial temperature (K)
        T_inf: Fluid temperature (K)
        h: Convective heat transfer coefficient (W/(m²·K))
        A: Surface area (m²)
        rho: Density (kg/m³)
        V: Volume (m³)
        Cp: Specific heat capacity (J/(kg·K))
        t: Time (s)
        
        Returns:
        Temperature at time t (K)
        """
        tau = (rho * V * Cp) / (h * A)  # Time constant
        return T_inf + (T_i - T_inf) * math.exp(-t / tau)
    
    @staticmethod
    def semi_infinite_solid_temperature(T_i, T_s, x, alpha, t):
        """
        Calculate temperature in semi-infinite solid with constant surface temperature.
        
        Parameters:
        T_i: Initial temperature (K)
        T_s: Surface temperature (K)
        x: Distance from surface (m)
        alpha: Thermal diffusivity (m²/s)
        t: Time (s)
        
        Returns:
        Temperature at distance x and time t (K)
        
        Note: Requires scipy.special.erf or Python 3.11+ for math.erf
        """
        try:
            # Try using math.erf (Python 3.11+)
            erf_func = math.erf
        except AttributeError:
            # Fall back to scipy if available
            try:
                from scipy.special import erf as erf_func
            except ImportError:
                raise ImportError("This function requires scipy.special.erf or Python 3.11+")
        
        eta = x / (2 * math.sqrt(alpha * t))
        return T_s + (T_i - T_s) * erf_func(eta)


class Fins:
    """
    Class for extended surface (fins) heat transfer calculations
    """
    
    @staticmethod
    def fin_parameter(h, P, k, A_c):
        """
        Calculate fin parameter m.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        P: Perimeter (m)
        k: Thermal conductivity (W/(m·K))
        A_c: Cross-sectional area (m²)
        
        Returns:
        Fin parameter m (1/m)
        """
        return math.sqrt((h * P) / (k * A_c))
    
    @staticmethod
    def fin_efficiency_rectangular(m, L, h_fin_tip='insulated'):
        """
        Calculate efficiency for rectangular fin.
        
        Parameters:
        m: Fin parameter (1/m)
        L: Fin length (m)
        h_fin_tip: 'insulated' or 'convecting' (default 'insulated')
        
        Returns:
        Fin efficiency (0-1)
        """
        if h_fin_tip == 'insulated':
            return math.tanh(m * L) / (m * L)
        else:
            # Simplified for convecting tip
            return math.tanh(m * L) / (m * L)
    
    @staticmethod
    def fin_heat_rate_rectangular(h, P, k, A_c, T_b, T_inf, L, h_fin_tip='insulated'):
        """
        Calculate heat transfer rate from rectangular fin.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        P: Perimeter (m)
        k: Thermal conductivity (W/(m·K))
        A_c: Cross-sectional area (m²)
        T_b: Base temperature (K)
        T_inf: Fluid temperature (K)
        L: Fin length (m)
        h_fin_tip: 'insulated' or 'convecting' (default 'insulated')
        
        Returns:
        Heat transfer rate (W)
        """
        m = Fins.fin_parameter(h, P, k, A_c)
        if h_fin_tip == 'insulated':
            return math.sqrt(h * P * k * A_c) * (T_b - T_inf) * math.tanh(m * L)
        else:
            # Simplified for convecting tip
            return math.sqrt(h * P * k * A_c) * (T_b - T_inf) * math.tanh(m * L)
    
    @staticmethod
    def fin_effectiveness(h, P, k, A_c, A_b, L):
        """
        Calculate fin effectiveness.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        P: Perimeter (m)
        k: Thermal conductivity (W/(m·K))
        A_c: Cross-sectional area (m²)
        A_b: Base area (m²)
        L: Fin length (m)
        
        Returns:
        Fin effectiveness (dimensionless)
        """
        m = Fins.fin_parameter(h, P, k, A_c)
        return math.sqrt((h * P * k * A_c) / (h * A_b)) * math.tanh(m * L)


class DimensionlessNumbers:
    """
    Class for dimensionless numbers used in heat transfer
    """
    
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
    
    @staticmethod
    def grashof_number(g, beta, T_s, T_inf, L, nu):
        """
        Calculate Grashof number (natural convection).
        
        Parameters:
        g: Gravitational acceleration (m/s²)
        beta: Thermal expansion coefficient (1/K)
        T_s: Surface temperature (K)
        T_inf: Fluid bulk temperature (K)
        L: Characteristic length (m)
        nu: Kinematic viscosity (m²/s)
        
        Returns:
        Grashof number (dimensionless)
        """
        return (g * beta * abs(T_s - T_inf) * L ** 3) / (nu ** 2)
    
    @staticmethod
    def rayleigh_number(Pr, Gr):
        """
        Calculate Rayleigh number.
        
        Parameters:
        Pr: Prandtl number
        Gr: Grashof number
        
        Returns:
        Rayleigh number (dimensionless)
        """
        return Pr * Gr
    
    @staticmethod
    def nusselt_number(h, L, k):
        """
        Calculate Nusselt number.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        L: Characteristic length (m)
        k: Thermal conductivity (W/(m·K))
        
        Returns:
        Nusselt number (dimensionless)
        """
        return (h * L) / k
    
    @staticmethod
    def biot_number(h, L, k):
        """
        Calculate Biot number.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        L: Characteristic length (m)
        k: Thermal conductivity (W/(m·K))
        
        Returns:
        Biot number (dimensionless)
        """
        return (h * L) / k
    
    @staticmethod
    def fourier_number(alpha, t, L):
        """
        Calculate Fourier number.
        
        Parameters:
        alpha: Thermal diffusivity (m²/s)
        t: Time (s)
        L: Characteristic length (m)
        
        Returns:
        Fourier number (dimensionless)
        """
        return (alpha * t) / (L ** 2)
    
    @staticmethod
    def peclet_number(Re, Pr):
        """
        Calculate Peclet number.
        
        Parameters:
        Re: Reynolds number
        Pr: Prandtl number
        
        Returns:
        Peclet number (dimensionless)
        """
        return Re * Pr
    
    @staticmethod
    def stanton_number(Nu, Re, Pr):
        """
        Calculate Stanton number.
        
        Parameters:
        Nu: Nusselt number
        Re: Reynolds number
        Pr: Prandtl number
        
        Returns:
        Stanton number (dimensionless)
        """
        return Nu / (Re * Pr)

