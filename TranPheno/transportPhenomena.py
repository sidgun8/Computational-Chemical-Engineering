"""
Transport Phenomena Formulas for Chemical Engineering
Comprehensive collection of classes for momentum, heat, and mass transport calculations
"""

import math


class MomentumTransport:
    """
    Class for momentum transport calculations (fluid mechanics)
    """
    
    @staticmethod
    def newtonian_shear_stress(mu, du_dy):
        """
        Calculate Newtonian shear stress (Newton's law of viscosity).
        
        Parameters:
        mu: Dynamic viscosity (Pa·s)
        du_dy: Velocity gradient (1/s)
        
        Returns:
        Shear stress (Pa)
        """
        return mu * du_dy
    
    @staticmethod
    def momentum_flux(tau):
        """
        Calculate momentum flux (shear stress).
        
        Parameters:
        tau: Shear stress (Pa)
        
        Returns:
        Momentum flux (N/m² or Pa)
        """
        return tau
    
    @staticmethod
    def velocity_profile_laminar_flow(r, R, v_max):
        """
        Calculate velocity profile for laminar flow in a pipe (parabolic).
        
        Parameters:
        r: Radial position (m)
        R: Pipe radius (m)
        v_max: Maximum velocity at center (m/s)
        
        Returns:
        Velocity at radial position r (m/s)
        """
        return v_max * (1 - (r / R) ** 2)
    
    @staticmethod
    def average_velocity_from_max(v_max):
        """
        Calculate average velocity from maximum velocity (laminar flow).
        
        Parameters:
        v_max: Maximum velocity (m/s)
        
        Returns:
        Average velocity (m/s)
        """
        return v_max / 2
    
    @staticmethod
    def wall_shear_stress_laminar(mu, v_avg, R):
        """
        Calculate wall shear stress for laminar flow.
        
        Parameters:
        mu: Dynamic viscosity (Pa·s)
        v_avg: Average velocity (m/s)
        R: Pipe radius (m)
        
        Returns:
        Wall shear stress (Pa)
        """
        return 4 * mu * v_avg / R
    
    @staticmethod
    def boundary_layer_thickness_flat_plate(x, Re_x):
        """
        Calculate boundary layer thickness for flat plate (laminar).
        
        Parameters:
        x: Distance from leading edge (m)
        Re_x: Reynolds number at position x
        
        Returns:
        Boundary layer thickness (m)
        """
        if Re_x == 0:
            return float('inf')
        return 5.0 * x / math.sqrt(Re_x)
    
    @staticmethod
    def displacement_thickness_flat_plate(delta):
        """
        Calculate displacement thickness for flat plate (laminar).
        
        Parameters:
        delta: Boundary layer thickness (m)
        
        Returns:
        Displacement thickness (m)
        """
        return delta / 3
    
    @staticmethod
    def momentum_thickness_flat_plate(delta):
        """
        Calculate momentum thickness for flat plate (laminar).
        
        Parameters:
        delta: Boundary layer thickness (m)
        
        Returns:
        Momentum thickness (m)
        """
        return (2 * delta) / 15
    
    @staticmethod
    def drag_coefficient_flat_plate_laminar(Re_L):
        """
        Calculate drag coefficient for flat plate (laminar flow).
        
        Parameters:
        Re_L: Reynolds number based on plate length
        
        Returns:
        Drag coefficient (dimensionless)
        """
        if Re_L == 0:
            return float('inf')
        return 1.328 / math.sqrt(Re_L)
    
    @staticmethod
    def drag_force_flat_plate(C_D, rho, v_inf, A):
        """
        Calculate drag force on flat plate.
        
        Parameters:
        C_D: Drag coefficient
        rho: Density (kg/m³)
        v_inf: Free stream velocity (m/s)
        A: Plate area (m²)
        
        Returns:
        Drag force (N)
        """
        return 0.5 * C_D * rho * v_inf ** 2 * A
    
    @staticmethod
    def momentum_balance_control_volume(rho, v_in, v_out, A):
        """
        Calculate net momentum flux for control volume.
        
        Parameters:
        rho: Density (kg/m³)
        v_in: Inlet velocity (m/s)
        v_out: Outlet velocity (m/s)
        A: Cross-sectional area (m²)
        
        Returns:
        Net momentum flux (N)
        """
        return rho * A * (v_out ** 2 - v_in ** 2)


class HeatTransfer:
    """
    Class for heat transport calculations
    """
    
    @staticmethod
    def fourier_heat_flux(k, dT_dx):
        """
        Calculate heat flux using Fourier's law (conduction).
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        dT_dx: Temperature gradient (K/m)
        
        Returns:
        Heat flux (W/m²)
        """
        return -k * dT_dx
    
    @staticmethod
    def heat_conduction_plane_wall(k, A, T1, T2, L):
        """
        Calculate heat transfer rate through plane wall (1D steady-state).
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        A: Cross-sectional area (m²)
        T1: Temperature at surface 1 (K)
        T2: Temperature at surface 2 (K)
        L: Wall thickness (m)
        
        Returns:
        Heat transfer rate (W)
        """
        return k * A * (T1 - T2) / L
    
    @staticmethod
    def heat_conduction_cylinder(k, L, T1, T2, r1, r2):
        """
        Calculate heat transfer rate through cylindrical wall (radial conduction).
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        L: Length of cylinder (m)
        T1: Temperature at inner surface (K)
        T2: Temperature at outer surface (K)
        r1: Inner radius (m)
        r2: Outer radius (m)
        
        Returns:
        Heat transfer rate (W)
        """
        if r1 == 0 or r2 == 0:
            raise ValueError("Radius cannot be zero")
        return (2 * math.pi * k * L * (T1 - T2)) / math.log(r2 / r1)
    
    @staticmethod
    def heat_conduction_sphere(k, T1, T2, r1, r2):
        """
        Calculate heat transfer rate through spherical shell (radial conduction).
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        T1: Temperature at inner surface (K)
        T2: Temperature at outer surface (K)
        r1: Inner radius (m)
        r2: Outer radius (m)
        
        Returns:
        Heat transfer rate (W)
        """
        if r1 == 0 or r2 == 0:
            raise ValueError("Radius cannot be zero")
        return (4 * math.pi * k * (T1 - T2)) / ((1 / r1) - (1 / r2))
    
    @staticmethod
    def thermal_resistance_conduction(L, k, A):
        """
        Calculate thermal resistance for conduction.
        
        Parameters:
        L: Thickness (m)
        k: Thermal conductivity (W/(m·K))
        A: Cross-sectional area (m²)
        
        Returns:
        Thermal resistance (K/W)
        """
        return L / (k * A)
    
    @staticmethod
    def thermal_resistance_convection(h, A):
        """
        Calculate thermal resistance for convection.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        A: Surface area (m²)
        
        Returns:
        Thermal resistance (K/W)
        """
        return 1 / (h * A)
    
    @staticmethod
    def heat_transfer_convection(h, A, T_s, T_inf):
        """
        Calculate heat transfer rate using Newton's law of cooling (convection).
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        A: Surface area (m²)
        T_s: Surface temperature (K)
        T_inf: Bulk fluid temperature (K)
        
        Returns:
        Heat transfer rate (W)
        """
        return h * A * (T_s - T_inf)
    
    @staticmethod
    def nusselt_number_laminar_flat_plate(Re_x, Pr):
        """
        Calculate local Nusselt number for laminar flow over flat plate.
        
        Parameters:
        Re_x: Reynolds number at position x
        Pr: Prandtl number
        
        Returns:
        Local Nusselt number (dimensionless)
        """
        if Re_x == 0:
            return float('inf')
        return 0.332 * (Re_x ** 0.5) * (Pr ** (1/3))
    
    @staticmethod
    def nusselt_number_turbulent_flat_plate(Re_x, Pr):
        """
        Calculate local Nusselt number for turbulent flow over flat plate.
        
        Parameters:
        Re_x: Reynolds number at position x
        Pr: Prandtl number
        
        Returns:
        Local Nusselt number (dimensionless)
        """
        if Re_x == 0:
            return float('inf')
        return 0.0296 * (Re_x ** 0.8) * (Pr ** (1/3))
    
    @staticmethod
    def nusselt_number_fully_developed_pipe_laminar():
        """
        Calculate Nusselt number for fully developed laminar flow in pipe (constant wall temperature).
        
        Returns:
        Nusselt number = 3.66 (dimensionless)
        """
        return 3.66
    
    @staticmethod
    def nusselt_number_fully_developed_pipe_laminar_constant_flux():
        """
        Calculate Nusselt number for fully developed laminar flow in pipe (constant heat flux).
        
        Returns:
        Nusselt number = 4.36 (dimensionless)
        """
        return 4.36
    
    @staticmethod
    def nusselt_number_dittus_boelter(Re, Pr, n=0.4):
        """
        Calculate Nusselt number using Dittus-Boelter equation (turbulent pipe flow).
        
        Parameters:
        Re: Reynolds number
        Pr: Prandtl number
        n: Exponent, 0.4 for heating, 0.3 for cooling
        
        Returns:
        Nusselt number (dimensionless)
        """
        return 0.023 * (Re ** 0.8) * (Pr ** n)
    
    @staticmethod
    def heat_transfer_coefficient_from_nusselt(Nu, k, L):
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
    def thermal_boundary_layer_thickness(x, Re_x, Pr):
        """
        Calculate thermal boundary layer thickness for flat plate (laminar).
        
        Parameters:
        x: Distance from leading edge (m)
        Re_x: Reynolds number at position x
        Pr: Prandtl number
        
        Returns:
        Thermal boundary layer thickness (m)
        """
        if Re_x == 0:
            return float('inf')
        return 5.0 * x / (Re_x ** 0.5 * Pr ** (1/3))
    
    @staticmethod
    def stefan_boltzmann_law(epsilon, T):
        """
        Calculate radiative heat flux using Stefan-Boltzmann law.
        
        Parameters:
        epsilon: Emissivity (0-1)
        T: Absolute temperature (K)
        
        Returns:
        Radiative heat flux (W/m²)
        """
        sigma = 5.67e-8  # Stefan-Boltzmann constant (W/(m²·K⁴))
        return epsilon * sigma * T ** 4
    
    @staticmethod
    def net_radiative_heat_transfer(epsilon, A, T1, T2):
        """
        Calculate net radiative heat transfer between two surfaces (simplified).
        
        Parameters:
        epsilon: Emissivity (0-1)
        A: Surface area (m²)
        T1: Temperature of surface 1 (K)
        T2: Temperature of surrounding/enclosure (K)
        
        Returns:
        Net radiative heat transfer rate (W)
        """
        sigma = 5.67e-8  # Stefan-Boltzmann constant
        return epsilon * sigma * A * (T1 ** 4 - T2 ** 4)
    
    @staticmethod
    def fins_efficiency_straight_fin(L, h, k, P, A_c, m=None):
        """
        Calculate efficiency of straight fin with uniform cross-section.
        
        Parameters:
        L: Fin length (m)
        h: Convective heat transfer coefficient (W/(m²·K))
        k: Fin thermal conductivity (W/(m·K))
        P: Perimeter (m)
        A_c: Cross-sectional area (m²)
        m: Fin parameter (optional, calculated if not provided)
        
        Returns:
        Fin efficiency (dimensionless)
        """
        if m is None:
            m = math.sqrt((h * P) / (k * A_c))
        
        if m * L == 0:
            return 1.0
        
        return math.tanh(m * L) / (m * L)
    
    @staticmethod
    def biot_number(h, L, k):
        """
        Calculate Biot number for lumped capacitance method.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        L: Characteristic length (m)
        k: Thermal conductivity (W/(m·K))
        
        Returns:
        Biot number (dimensionless)
        """
        return (h * L) / k
    
    @staticmethod
    def lumped_capacitance_time_constant(rho, Cp, V, h, A):
        """
        Calculate time constant for lumped capacitance method.
        
        Parameters:
        rho: Density (kg/m³)
        Cp: Specific heat capacity (J/(kg·K))
        V: Volume (m³)
        h: Convective heat transfer coefficient (W/(m²·K))
        A: Surface area (m²)
        
        Returns:
        Time constant (s)
        """
        return (rho * Cp * V) / (h * A)
    
    @staticmethod
    def temperature_lumped_capacitance(T0, T_inf, t, tau):
        """
        Calculate temperature using lumped capacitance method.
        
        Parameters:
        T0: Initial temperature (K)
        T_inf: Ambient temperature (K)
        t: Time (s)
        tau: Time constant (s)
        
        Returns:
        Temperature at time t (K)
        """
        return T_inf + (T0 - T_inf) * math.exp(-t / tau)


class MassTransfer:
    """
    Class for mass transport calculations
    """
    
    @staticmethod
    def ficks_law_molar_flux(D_AB, dC_A_dx):
        """
        Calculate molar flux using Fick's law of diffusion.
        
        Parameters:
        D_AB: Diffusion coefficient (m²/s)
        dC_A_dx: Concentration gradient (mol/(m³·m) or mol/m⁴)
        
        Returns:
        Molar flux (mol/(m²·s))
        """
        return -D_AB * dC_A_dx
    
    @staticmethod
    def ficks_law_mass_flux(D_AB, rho, dw_A_dx):
        """
        Calculate mass flux using Fick's law (mass basis).
        
        Parameters:
        D_AB: Diffusion coefficient (m²/s)
        rho: Total density (kg/m³)
        dw_A_dx: Mass fraction gradient (1/m)
        
        Returns:
        Mass flux (kg/(m²·s))
        """
        return -D_AB * rho * dw_A_dx
    
    @staticmethod
    def diffusion_steady_state_plane_wall(D_AB, A, C_A1, C_A2, L):
        """
        Calculate molar diffusion rate through plane wall (1D steady-state).
        
        Parameters:
        D_AB: Diffusion coefficient (m²/s)
        A: Cross-sectional area (m²)
        C_A1: Concentration at surface 1 (mol/m³)
        C_A2: Concentration at surface 2 (mol/m³)
        L: Wall thickness (m)
        
        Returns:
        Molar diffusion rate (mol/s)
        """
        return D_AB * A * (C_A1 - C_A2) / L
    
    @staticmethod
    def diffusion_steady_state_cylinder(D_AB, L, C_A1, C_A2, r1, r2):
        """
        Calculate molar diffusion rate through cylindrical shell (radial diffusion).
        
        Parameters:
        D_AB: Diffusion coefficient (m²/s)
        L: Length of cylinder (m)
        C_A1: Concentration at inner surface (mol/m³)
        C_A2: Concentration at outer surface (mol/m³)
        r1: Inner radius (m)
        r2: Outer radius (m)
        
        Returns:
        Molar diffusion rate (mol/s)
        """
        if r1 == 0 or r2 == 0:
            raise ValueError("Radius cannot be zero")
        return (2 * math.pi * D_AB * L * (C_A1 - C_A2)) / math.log(r2 / r1)
    
    @staticmethod
    def mass_transfer_convection(k_c, A, C_As, C_A_inf):
        """
        Calculate mass transfer rate using convective mass transfer coefficient.
        
        Parameters:
        k_c: Mass transfer coefficient (m/s)
        A: Surface area (m²)
        C_As: Surface concentration (mol/m³)
        C_A_inf: Bulk concentration (mol/m³)
        
        Returns:
        Molar mass transfer rate (mol/s)
        """
        return k_c * A * (C_As - C_A_inf)
    
    @staticmethod
    def sherwood_number_laminar_flat_plate(Re_x, Sc):
        """
        Calculate local Sherwood number for laminar flow over flat plate.
        
        Parameters:
        Re_x: Reynolds number at position x
        Sc: Schmidt number
        
        Returns:
        Local Sherwood number (dimensionless)
        """
        if Re_x == 0:
            return float('inf')
        return 0.332 * (Re_x ** 0.5) * (Sc ** (1/3))
    
    @staticmethod
    def sherwood_number_turbulent_flat_plate(Re_x, Sc):
        """
        Calculate local Sherwood number for turbulent flow over flat plate.
        
        Parameters:
        Re_x: Reynolds number at position x
        Sc: Schmidt number
        
        Returns:
        Local Sherwood number (dimensionless)
        """
        if Re_x == 0:
            return float('inf')
        return 0.0296 * (Re_x ** 0.8) * (Sc ** (1/3))
    
    @staticmethod
    def sherwood_number_fully_developed_pipe_laminar():
        """
        Calculate Sherwood number for fully developed laminar flow in pipe (constant wall concentration).
        
        Returns:
        Sherwood number = 3.66 (dimensionless)
        """
        return 3.66
    
    @staticmethod
    def sherwood_number_dittus_boelter_mass(Re, Sc, n=0.4):
        """
        Calculate Sherwood number using Dittus-Boelter type equation (turbulent pipe flow).
        
        Parameters:
        Re: Reynolds number
        Sc: Schmidt number
        n: Exponent, 0.4 for increasing concentration, 0.3 for decreasing
        
        Returns:
        Sherwood number (dimensionless)
        """
        return 0.023 * (Re ** 0.8) * (Sc ** n)
    
    @staticmethod
    def mass_transfer_coefficient_from_sherwood(Sh, D_AB, L):
        """
        Calculate mass transfer coefficient from Sherwood number.
        
        Parameters:
        Sh: Sherwood number
        D_AB: Diffusion coefficient (m²/s)
        L: Characteristic length (m)
        
        Returns:
        Mass transfer coefficient (m/s)
        """
        return (Sh * D_AB) / L
    
    @staticmethod
    def concentration_boundary_layer_thickness(x, Re_x, Sc):
        """
        Calculate concentration boundary layer thickness for flat plate (laminar).
        
        Parameters:
        x: Distance from leading edge (m)
        Re_x: Reynolds number at position x
        Sc: Schmidt number
        
        Returns:
        Concentration boundary layer thickness (m)
        """
        if Re_x == 0:
            return float('inf')
        return 5.0 * x / (Re_x ** 0.5 * Sc ** (1/3))
    
    @staticmethod
    def diffusion_coefficient_gas_chapman_enkog(T, P, M_A, M_B, sigma_AB, omega_D):
        """
        Estimate binary diffusion coefficient for gases using Chapman-Enskog theory (simplified).
        Note: This is a simplified form; full Chapman-Enskog requires collision integrals.
        
        Parameters:
        T: Temperature (K)
        P: Pressure (Pa)
        M_A: Molecular weight of species A (kg/mol)
        M_B: Molecular weight of species B (kg/mol)
        sigma_AB: Average collision diameter (m)
        omega_D: Collision integral (dimensionless, ~1 for simplification)
        
        Returns:
        Diffusion coefficient (m²/s)
        """
        k_B = 1.380649e-23  # Boltzmann constant (J/K)
        M_AB = 2 / ((1 / M_A) + (1 / M_B))  # Reduced molecular weight
        return (3 / (16 * math.pi)) * math.sqrt((k_B ** 3 * T ** 3) / (M_AB * P * sigma_AB ** 2 * omega_D))
    
    @staticmethod
    def diffusion_coefficient_gas_wilke_chang(T, mu_B, V_A, phi_B, M_B):
        """
        Estimate binary diffusion coefficient in liquids using Wilke-Chang equation.
        
        Parameters:
        T: Temperature (K)
        mu_B: Viscosity of solvent B (Pa·s)
        V_A: Molar volume of solute A at normal boiling point (m³/mol)
        phi_B: Association parameter for solvent (1.0 for unassociated solvents, 2.6 for water)
        M_B: Molecular weight of solvent B (kg/mol)
        
        Returns:
        Diffusion coefficient (m²/s)
        """
        # Wilke-Chang constant
        constant = 7.4e-12  # (m²/s) for SI units
        return (constant * (phi_B * M_B) ** 0.5 * T) / (mu_B * V_A ** 0.6)
    
    @staticmethod
    def stefan_maxwell_equation_simplified(x_A, x_B, D_AB, N_B=0):
        """
        Simplified Stefan-Maxwell equation for binary mixture (equimolar counter-diffusion).
        
        Parameters:
        x_A: Mole fraction of A
        x_B: Mole fraction of B
        D_AB: Binary diffusion coefficient (m²/s)
        N_B: Molar flux of B (mol/(m²·s)), default 0 for equimolar counter-diffusion
        
        Returns:
        Concentration gradient term (simplified)
        """
        # This is a simplified representation; full Stefan-Maxwell is more complex
        return (x_A * N_B - x_B) / D_AB


class DimensionlessNumbers:
    """
    Class for dimensionless numbers used in transport phenomena
    """
    
    @staticmethod
    def reynolds_number(rho, v, L, mu):
        """
        Calculate Reynolds number (momentum transport).
        
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
    def prandtl_number(Cp, mu, k):
        """
        Calculate Prandtl number (momentum to thermal diffusivity ratio).
        
        Parameters:
        Cp: Specific heat at constant pressure (J/(kg·K))
        mu: Dynamic viscosity (Pa·s)
        k: Thermal conductivity (W/(m·K))
        
        Returns:
        Prandtl number (dimensionless)
        """
        return (Cp * mu) / k
    
    @staticmethod
    def schmidt_number(mu, rho, D_AB):
        """
        Calculate Schmidt number (momentum to mass diffusivity ratio).
        
        Parameters:
        mu: Dynamic viscosity (Pa·s)
        rho: Density (kg/m³)
        D_AB: Diffusion coefficient (m²/s)
        
        Returns:
        Schmidt number (dimensionless)
        """
        return mu / (rho * D_AB)
    
    @staticmethod
    def lewis_number(k, rho, Cp, D_AB):
        """
        Calculate Lewis number (thermal to mass diffusivity ratio).
        
        Parameters:
        k: Thermal conductivity (W/(m·K))
        rho: Density (kg/m³)
        Cp: Specific heat capacity (J/(kg·K))
        D_AB: Diffusion coefficient (m²/s)
        
        Returns:
        Lewis number (dimensionless)
        """
        alpha = k / (rho * Cp)  # Thermal diffusivity
        return alpha / D_AB
    
    @staticmethod
    def nusselt_number(h, L, k):
        """
        Calculate Nusselt number (convective to conductive heat transfer).
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        L: Characteristic length (m)
        k: Thermal conductivity (W/(m·K))
        
        Returns:
        Nusselt number (dimensionless)
        """
        return (h * L) / k
    
    @staticmethod
    def sherwood_number(k_c, L, D_AB):
        """
        Calculate Sherwood number (convective to diffusive mass transfer).
        
        Parameters:
        k_c: Mass transfer coefficient (m/s)
        L: Characteristic length (m)
        D_AB: Diffusion coefficient (m²/s)
        
        Returns:
        Sherwood number (dimensionless)
        """
        return (k_c * L) / D_AB
    
    @staticmethod
    def peclet_number_heat(Re, Pr):
        """
        Calculate Peclet number for heat transfer.
        
        Parameters:
        Re: Reynolds number
        Pr: Prandtl number
        
        Returns:
        Peclet number (dimensionless)
        """
        return Re * Pr
    
    @staticmethod
    def peclet_number_mass(Re, Sc):
        """
        Calculate Peclet number for mass transfer.
        
        Parameters:
        Re: Reynolds number
        Sc: Schmidt number
        
        Returns:
        Peclet number (dimensionless)
        """
        return Re * Sc
    
    @staticmethod
    def biot_number_heat(h, L, k):
        """
        Calculate Biot number for heat transfer.
        
        Parameters:
        h: Convective heat transfer coefficient (W/(m²·K))
        L: Characteristic length (m)
        k: Thermal conductivity (W/(m·K))
        
        Returns:
        Biot number (dimensionless)
        """
        return (h * L) / k
    
    @staticmethod
    def stanton_number_heat(Nu, Re, Pr):
        """
        Calculate Stanton number for heat transfer.
        
        Parameters:
        Nu: Nusselt number
        Re: Reynolds number
        Pr: Prandtl number
        
        Returns:
        Stanton number (dimensionless)
        """
        return Nu / (Re * Pr)
    
    @staticmethod
    def stanton_number_mass(Sh, Re, Sc):
        """
        Calculate Stanton number for mass transfer.
        
        Parameters:
        Sh: Sherwood number
        Re: Reynolds number
        Sc: Schmidt number
        
        Returns:
        Stanton number (dimensionless)
        """
        return Sh / (Re * Sc)
    
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
    def kinematic_viscosity(mu, rho):
        """
        Calculate kinematic viscosity (momentum diffusivity).
        
        Parameters:
        mu: Dynamic viscosity (Pa·s)
        rho: Density (kg/m³)
        
        Returns:
        Kinematic viscosity (m²/s)
        """
        return mu / rho


class TransportAnalogy:
    """
    Class for transport analogies (Reynolds, Chilton-Colburn)
    """
    
    @staticmethod
    def reynolds_analogy(C_f, St_H=None, St_M=None):
        """
        Reynolds analogy: f/2 = St_H = St_M (for Pr = Sc = 1).
        
        Parameters:
        C_f: Skin friction coefficient (f/2)
        St_H: Stanton number for heat transfer (optional)
        St_M: Stanton number for mass transfer (optional)
        
        Returns:
        Dictionary with calculated values
        """
        result = {'friction_coefficient': C_f}
        if St_H is None:
            result['Stanton_heat'] = C_f
        else:
            result['Stanton_heat'] = St_H
            
        if St_M is None:
            result['Stanton_mass'] = C_f
        else:
            result['Stanton_mass'] = St_M
            
        return result
    
    @staticmethod
    def chilton_colburn_analogy(C_f, Pr=None, Sc=None):
        """
        Chilton-Colburn analogy: j_H = j_M = f/2 (valid for 0.6 < Pr < 60, 0.6 < Sc < 3000).
        
        Parameters:
        C_f: Skin friction coefficient (f/2)
        Pr: Prandtl number (optional, for heat transfer)
        Sc: Schmidt number (optional, for mass transfer)
        
        Returns:
        Dictionary with j-factors
        """
        result = {'friction_coefficient': C_f}
        
        if Pr is not None:
            # j_H = St_H * Pr^(2/3) = f/2
            result['j_H'] = C_f
            result['Stanton_heat'] = C_f / (Pr ** (2/3))
        
        if Sc is not None:
            # j_M = St_M * Sc^(2/3) = f/2
            result['j_M'] = C_f
            result['Stanton_mass'] = C_f / (Sc ** (2/3))
            
        return result
    
    @staticmethod
    def heat_mass_analogy(Sh, Nu, Sc, Pr):
        """
        Analogy between heat and mass transfer coefficients.
        
        Parameters:
        Sh: Sherwood number
        Nu: Nusselt number
        Sc: Schmidt number
        Pr: Prandtl number
        
        Returns:
        Ratio Sh/Nu and Sc/Pr for comparison
        """
        return {
            'Sh_Nu_ratio': Sh / Nu if Nu != 0 else float('inf'),
            'Sc_Pr_ratio': Sc / Pr if Pr != 0 else float('inf'),
            'analogy_valid': abs((Sh / Nu) - (Sc / Pr)) < 0.1 if Nu != 0 and Pr != 0 else False
        }


class BoundaryLayers:
    """
    Class for boundary layer calculations across transport phenomena
    """
    
    @staticmethod
    def momentum_boundary_layer_ratio(Pr):
        """
        Calculate ratio of thermal to momentum boundary layer thickness (Pr >> 1).
        
        Parameters:
        Pr: Prandtl number
        
        Returns:
        Ratio delta_T / delta_M
        """
        return Pr ** (-1/3)
    
    @staticmethod
    def concentration_boundary_layer_ratio(Sc):
        """
        Calculate ratio of concentration to momentum boundary layer thickness (Sc >> 1).
        
        Parameters:
        Sc: Schmidt number
        
        Returns:
        Ratio delta_C / delta_M
        """
        return Sc ** (-1/3)
    
    @staticmethod
    def blasius_solution_shear_stress(tau_w, rho, v_inf, x):
        """
        Calculate wall shear stress using Blasius solution (laminar flat plate).
        
        Parameters:
        tau_w: Wall shear stress (Pa)
        rho: Density (kg/m³)
        v_inf: Free stream velocity (m/s)
        x: Distance from leading edge (m)
        
        Returns:
        Local skin friction coefficient
        """
        if x == 0:
            return float('inf')
        return tau_w / (0.5 * rho * v_inf ** 2)
