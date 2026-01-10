"""
Mass Transfer Formulas for Chemical Engineering
Comprehensive collection of classes for mass transfer calculations
"""

import math


class Diffusion:
    """
    Class for diffusion calculations
    """
    
    @staticmethod
    def ficks_first_law_flux(D_AB, dC_dx):
        """
        Calculate molar flux using Fick's first law.
        
        Parameters:
        D_AB: Diffusion coefficient (m²/s)
        dC_dx: Concentration gradient (mol/(m³·m))
        
        Returns:
        Molar flux (mol/(m²·s))
        """
        return -D_AB * dC_dx
    
    @staticmethod
    def ficks_second_law_concentration(C0, x, D_AB, t):
        """
        Calculate concentration profile using Fick's second law (1D, infinite medium).
        
        Parameters:
        C0: Initial concentration (mol/m³)
        x: Position (m)
        D_AB: Diffusion coefficient (m²/s)
        t: Time (s)
        
        Returns:
        Concentration at position x and time t (mol/m³)
        """
        if t == 0:
            return C0
        return C0 * math.erfc(x / (2 * math.sqrt(D_AB * t)))
    
    @staticmethod
    def diffusion_coefficient_gas_fuller(T, P, M_A, M_B, v_A, v_B):
        """
        Calculate binary diffusion coefficient in gases using Fuller-Schettler-Giddings correlation.
        
        Parameters:
        T: Temperature (K)
        P: Pressure (Pa)
        M_A: Molecular weight of A (kg/kmol)
        M_B: Molecular weight of B (kg/kmol)
        v_A: Diffusion volume of A (cm³/mol)
        v_B: Diffusion volume of B (cm³/mol)
        
        Returns:
        Diffusion coefficient (m²/s)
        """
        M_AB = 2 / ((1 / M_A) + (1 / M_B))
        v_AB = (v_A ** (1/3) + v_B ** (1/3)) ** 3
        # Convert pressure from Pa to atm for correlation
        P_atm = P / 101325
        # Fuller-Schettler-Giddings equation
        D_AB = (0.00143 * T ** 1.75) / (P_atm * math.sqrt(M_AB) * v_AB)
        # Convert from cm²/s to m²/s
        return D_AB * 1e-4
    
    @staticmethod
    def diffusion_coefficient_gas_wilke_lee(T, P, M_A, M_B, sigma_AB, epsilon_AB_k):
        """
        Calculate binary diffusion coefficient using Wilke-Lee correlation.
        
        Parameters:
        T: Temperature (K)
        P: Pressure (Pa)
        M_A: Molecular weight of A (kg/kmol)
        M_B: Molecular weight of B (kg/kmol)
        sigma_AB: Collision diameter (Å)
        epsilon_AB_k: Lennard-Jones parameter (K)
        
        Returns:
        Diffusion coefficient (m²/s)
        """
        M_AB = 2 / ((1 / M_A) + (1 / M_B))
        T_star = T / epsilon_AB_k
        omega_D = 1.06036 / (T_star ** 0.15610) + 0.19300 / math.exp(0.47635 * T_star) + \
                  1.03587 / math.exp(1.52996 * T_star) + 1.76474 / math.exp(3.89411 * T_star)
        # Convert pressure from Pa to atm
        P_atm = P / 101325
        # Convert sigma from Å to m
        sigma_m = sigma_AB * 1e-10
        D_AB = (3.03 - 0.98 / math.sqrt(M_AB)) * 1e-3 * (T ** 1.5) / \
               (P_atm * math.sqrt(M_AB) * sigma_m ** 2 * omega_D)
        return D_AB
    
    @staticmethod
    def diffusion_coefficient_liquid_wilke_chang(T, mu_B, M_B, V_A, phi=2.6):
        """
        Calculate diffusion coefficient in liquids using Wilke-Chang correlation.
        
        Parameters:
        T: Temperature (K)
        mu_B: Viscosity of solvent (Pa·s)
        M_B: Molecular weight of solvent (kg/kmol)
        V_A: Molar volume of solute at normal boiling point (cm³/mol)
        phi: Association parameter (2.6 for water, 1.9 for methanol, 1.0 for unassociated)
        
        Returns:
        Diffusion coefficient (m²/s)
        """
        # Wilke-Chang equation
        D_AB = 7.4e-8 * (phi * M_B) ** 0.5 * T / (mu_B * (V_A ** 0.6))
        # Convert from cm²/s to m²/s
        return D_AB * 1e-4
    
    @staticmethod
    def diffusion_coefficient_liquid_stokes_einstein(k_B, T, mu, r):
        """
        Calculate diffusion coefficient using Stokes-Einstein equation (for large molecules).
        
        Parameters:
        k_B: Boltzmann constant (1.380649e-23 J/K)
        T: Temperature (K)
        mu: Viscosity (Pa·s)
        r: Radius of diffusing particle (m)
        
        Returns:
        Diffusion coefficient (m²/s)
        """
        return (k_B * T) / (6 * math.pi * mu * r)
    
    @staticmethod
    def effective_diffusion_porous(epsilon, tau, D_AB):
        """
        Calculate effective diffusion coefficient in porous media.
        
        Parameters:
        epsilon: Porosity (void fraction)
        tau: Tortuosity factor
        D_AB: Bulk diffusion coefficient (m²/s)
        
        Returns:
        Effective diffusion coefficient (m²/s)
        """
        return (epsilon * D_AB) / tau


class MassTransferCoefficients:
    """
    Class for mass transfer coefficient calculations
    """
    
    @staticmethod
    def mass_transfer_flux(k_c, C_bulk, C_surface):
        """
        Calculate mass transfer flux.
        
        Parameters:
        k_c: Mass transfer coefficient (m/s)
        C_bulk: Bulk concentration (mol/m³)
        C_surface: Surface concentration (mol/m³)
        
        Returns:
        Molar flux (mol/(m²·s))
        """
        return k_c * (C_bulk - C_surface)
    
    @staticmethod
    def overall_mass_transfer_coefficient_liquid(k_L, k_G, H, m):
        """
        Calculate overall mass transfer coefficient (liquid phase basis).
        
        Parameters:
        k_L: Liquid phase mass transfer coefficient (m/s)
        k_G: Gas phase mass transfer coefficient (mol/(m²·s·Pa))
        H: Henry's law constant (Pa·m³/mol)
        m: Partition coefficient (dimensionless)
        
        Returns:
        Overall mass transfer coefficient (m/s)
        """
        return 1 / ((1 / k_L) + (1 / (H * k_G)))
    
    @staticmethod
    def overall_mass_transfer_coefficient_gas(k_L, k_G, H):
        """
        Calculate overall mass transfer coefficient (gas phase basis).
        
        Parameters:
        k_L: Liquid phase mass transfer coefficient (m/s)
        k_G: Gas phase mass transfer coefficient (mol/(m²·s·Pa))
        H: Henry's law constant (Pa·m³/mol)
        
        Returns:
        Overall mass transfer coefficient (mol/(m²·s·Pa))
        """
        return 1 / ((1 / k_G) + (H / k_L))
    
    @staticmethod
    def mass_transfer_coefficient_flat_plate_laminar(Re, Sc, D_AB, L):
        """
        Calculate average mass transfer coefficient for flat plate (laminar flow).
        
        Parameters:
        Re: Reynolds number
        Sc: Schmidt number
        D_AB: Diffusion coefficient (m²/s)
        L: Length of plate (m)
        
        Returns:
        Average mass transfer coefficient (m/s)
        """
        Sh = 0.664 * (Re ** 0.5) * (Sc ** (1/3))
        return (Sh * D_AB) / L
    
    @staticmethod
    def mass_transfer_coefficient_flat_plate_turbulent(Re, Sc, D_AB, L):
        """
        Calculate average mass transfer coefficient for flat plate (turbulent flow).
        
        Parameters:
        Re: Reynolds number
        Sc: Schmidt number
        D_AB: Diffusion coefficient (m²/s)
        L: Length of plate (m)
        
        Returns:
        Average mass transfer coefficient (m/s)
        """
        Sh = 0.037 * (Re ** 0.8) * (Sc ** (1/3))
        return (Sh * D_AB) / L
    
    @staticmethod
    def mass_transfer_coefficient_cylinder(Re, Sc, D_AB, D):
        """
        Calculate mass transfer coefficient for flow over cylinder.
        
        Parameters:
        Re: Reynolds number
        Sc: Schmidt number
        D_AB: Diffusion coefficient (m²/s)
        D: Diameter (m)
        
        Returns:
        Mass transfer coefficient (m/s)
        """
        Sh = 0.3 + (0.62 * Re ** 0.5 * Sc ** (1/3)) / \
            (1 + (0.4 / Sc) ** (2/3)) ** 0.25 * \
            (1 + (Re / 282000) ** (5/8)) ** (4/5)
        return (Sh * D_AB) / D
    
    @staticmethod
    def mass_transfer_coefficient_sphere(Re, Sc, D_AB, D):
        """
        Calculate mass transfer coefficient for flow over sphere (Frossling correlation).
        
        Parameters:
        Re: Reynolds number
        Sc: Schmidt number
        D_AB: Diffusion coefficient (m²/s)
        D: Diameter (m)
        
        Returns:
        Mass transfer coefficient (m/s)
        """
        Sh = 2 + 0.6 * (Re ** 0.5) * (Sc ** (1/3))
        return (Sh * D_AB) / D
    
    @staticmethod
    def mass_transfer_coefficient_packed_bed(Re, Sc, D_AB, D_p, epsilon):
        """
        Calculate mass transfer coefficient in packed bed.
        
        Parameters:
        Re: Reynolds number (based on particle diameter)
        Sc: Schmidt number
        D_AB: Diffusion coefficient (m²/s)
        D_p: Particle diameter (m)
        epsilon: Bed porosity
        
        Returns:
        Mass transfer coefficient (m/s)
        """
        j_D = 0.91 * (Re ** -0.51) * (epsilon ** -1)  # For Re < 50
        if Re >= 50:
            j_D = 0.61 * (Re ** -0.41) * (epsilon ** -1)
        Sh = j_D * Re * (Sc ** (1/3))
        return (Sh * D_AB) / D_p
    
    @staticmethod
    def mass_transfer_coefficient_inside_tube_laminar(Re, Sc, D_AB, D, L):
        """
        Calculate mass transfer coefficient inside tube (laminar flow, Leveque solution).
        
        Parameters:
        Re: Reynolds number
        Sc: Schmidt number
        D_AB: Diffusion coefficient (m²/s)
        D: Tube diameter (m)
        L: Tube length (m)
        
        Returns:
        Mass transfer coefficient (m/s)
        """
        Sh = 3.66 + (0.0668 * (D / L) * Re * Sc) / \
            (1 + 0.04 * ((D / L) * Re * Sc) ** (2/3))
        return (Sh * D_AB) / D
    
    @staticmethod
    def mass_transfer_coefficient_inside_tube_turbulent(Re, Sc, D_AB, D):
        """
        Calculate mass transfer coefficient inside tube (turbulent flow, Dittus-Boelter type).
        
        Parameters:
        Re: Reynolds number
        Sc: Schmidt number
        D_AB: Diffusion coefficient (m²/s)
        D: Tube diameter (m)
        
        Returns:
        Mass transfer coefficient (m/s)
        """
        Sh = 0.023 * (Re ** 0.8) * (Sc ** 0.33)
        return (Sh * D_AB) / D


class DimensionlessNumbers:
    """
    Class for dimensionless numbers in mass transfer
    """
    
    @staticmethod
    def schmidt_number(mu, rho, D_AB):
        """
        Calculate Schmidt number.
        
        Parameters:
        mu: Dynamic viscosity (Pa·s)
        rho: Density (kg/m³)
        D_AB: Diffusion coefficient (m²/s)
        
        Returns:
        Schmidt number (dimensionless)
        """
        return mu / (rho * D_AB)
    
    @staticmethod
    def sherwood_number(k_c, L, D_AB):
        """
        Calculate Sherwood number.
        
        Parameters:
        k_c: Mass transfer coefficient (m/s)
        L: Characteristic length (m)
        D_AB: Diffusion coefficient (m²/s)
        
        Returns:
        Sherwood number (dimensionless)
        """
        return (k_c * L) / D_AB
    
    @staticmethod
    def lewis_number(alpha, D_AB):
        """
        Calculate Lewis number.
        
        Parameters:
        alpha: Thermal diffusivity (m²/s)
        D_AB: Mass diffusivity (m²/s)
        
        Returns:
        Lewis number (dimensionless)
        """
        return alpha / D_AB
    
    @staticmethod
    def peclet_number_mass(v, L, D_AB):
        """
        Calculate Peclet number for mass transfer.
        
        Parameters:
        v: Velocity (m/s)
        L: Characteristic length (m)
        D_AB: Diffusion coefficient (m²/s)
        
        Returns:
        Peclet number (dimensionless)
        """
        return (v * L) / D_AB
    
    @staticmethod
    def stanton_number_mass(k_c, v):
        """
        Calculate Stanton number for mass transfer.
        
        Parameters:
        k_c: Mass transfer coefficient (m/s)
        v: Velocity (m/s)
        
        Returns:
        Stanton number (dimensionless)
        """
        return k_c / v
    
    @staticmethod
    def j_factor_mass(Sh, Re, Sc):
        """
        Calculate j-factor for mass transfer.
        
        Parameters:
        Sh: Sherwood number
        Re: Reynolds number
        Sc: Schmidt number
        
        Returns:
        j-factor (dimensionless)
        """
        return Sh / (Re * (Sc ** (1/3)))


class AbsorptionStripping:
    """
    Class for absorption and stripping calculations
    """
    
    @staticmethod
    def henrys_law_constant(P, x, y):
        """
        Calculate Henry's law constant from equilibrium data.
        
        Parameters:
        P: Total pressure (Pa)
        x: Liquid phase mole fraction
        y: Gas phase mole fraction
        
        Returns:
        Henry's law constant (Pa)
        """
        if x == 0:
            return float('inf')
        return (P * y) / x
    
    @staticmethod
    def raoults_law_vapor_pressure(x, P_sat):
        """
        Calculate partial pressure using Raoult's law.
        
        Parameters:
        x: Liquid mole fraction
        P_sat: Saturation pressure (Pa)
        
        Returns:
        Partial pressure (Pa)
        """
        return x * P_sat
    
    @staticmethod
    def number_of_transfer_units_absorption(y_in, y_out, y_star_in, y_star_out):
        """
        Calculate number of transfer units (NTU) for absorption (gas phase).
        
        Parameters:
        y_in: Inlet gas mole fraction
        y_out: Outlet gas mole fraction
        y_star_in: Equilibrium gas mole fraction at inlet liquid composition
        y_star_out: Equilibrium gas mole fraction at outlet liquid composition
        
        Returns:
        Number of transfer units (dimensionless)
        """
        delta_y_1 = y_in - y_star_in
        delta_y_2 = y_out - y_star_out
        if delta_y_1 == delta_y_2:
            return (y_in - y_out) / delta_y_1
        delta_y_lm = (delta_y_1 - delta_y_2) / math.log(delta_y_1 / delta_y_2)
        return (y_in - y_out) / delta_y_lm
    
    @staticmethod
    def height_of_transfer_unit(K_y, G, a):
        """
        Calculate height of transfer unit (HTU) for gas phase.
        
        Parameters:
        K_y: Overall gas phase mass transfer coefficient (mol/(m²·s))
        G: Gas molar flux (mol/(m²·s))
        a: Interfacial area per unit volume (m²/m³)
        
        Returns:
        Height of transfer unit (m)
        """
        return G / (K_y * a)
    
    @staticmethod
    def tower_height(NTU, HTU):
        """
        Calculate packed tower height.
        
        Parameters:
        NTU: Number of transfer units
        HTU: Height of transfer unit (m)
        
        Returns:
        Tower height (m)
        """
        return NTU * HTU
    
    @staticmethod
    def minimum_liquid_gas_ratio(y_in, y_out, x_in, x_out_eq):
        """
        Calculate minimum liquid-to-gas ratio for absorption.
        
        Parameters:
        y_in: Inlet gas mole fraction
        y_out: Outlet gas mole fraction
        x_in: Inlet liquid mole fraction
        x_out_eq: Equilibrium liquid mole fraction at outlet gas composition
        
        Returns:
        Minimum L/G ratio (dimensionless)
        """
        return (y_in - y_out) / (x_out_eq - x_in)
    
    @staticmethod
    def operating_line_absorption(L, G, x_in, y_in, x):
        """
        Calculate operating line for absorption (gas phase composition).
        
        Parameters:
        L: Liquid molar flow rate (mol/s)
        G: Gas molar flow rate (mol/s)
        x_in: Inlet liquid mole fraction
        y_in: Inlet gas mole fraction
        x: Liquid mole fraction
        
        Returns:
        Gas mole fraction (dimensionless)
        """
        return y_in + (L / G) * (x - x_in)


class Distillation:
    """
    Class for distillation calculations
    """
    
    @staticmethod
    def relative_volatility(P_A_sat, P_B_sat):
        """
        Calculate relative volatility.
        
        Parameters:
        P_A_sat: Saturation pressure of component A (Pa)
        P_B_sat: Saturation pressure of component B (Pa)
        
        Returns:
        Relative volatility (dimensionless)
        """
        return P_A_sat / P_B_sat
    
    @staticmethod
    def equilibrium_relation_ideal(alpha, x):
        """
        Calculate equilibrium vapor composition (ideal solution).
        
        Parameters:
        alpha: Relative volatility
        x: Liquid mole fraction
        
        Returns:
        Vapor mole fraction (dimensionless)
        """
        return (alpha * x) / (1 + (alpha - 1) * x)
    
    @staticmethod
    def minimum_reflux_ratio(x_D, x_F, x_W, alpha, q):
        """
        Calculate minimum reflux ratio using Underwood method (simplified).
        
        Parameters:
        x_D: Distillate composition
        x_F: Feed composition
        x_W: Bottoms composition
        alpha: Relative volatility
        q: Feed quality (1 = saturated liquid, 0 = saturated vapor)
        
        Returns:
        Minimum reflux ratio (dimensionless)
        """
        # Simplified Fenske-Underwood approach
        R_min = (x_D / (1 - x_D) - alpha * x_F / (1 - x_F)) / (alpha - 1)
        return max(0, R_min)
    
    @staticmethod
    def fenske_equation(alpha, x_D, x_W, N_min):
        """
        Calculate minimum number of stages using Fenske equation.
        
        Parameters:
        alpha: Relative volatility
        x_D: Distillate composition
        x_W: Bottoms composition
        N_min: Minimum number of stages (to be calculated)
        
        Returns:
        Minimum number of stages (dimensionless)
        """
        return math.log((x_D / (1 - x_D)) * ((1 - x_W) / x_W)) / math.log(alpha)
    
    @staticmethod
    def mccabe_thiele_stages(x_D, x_F, x_W, R, alpha, q):
        """
        Estimate number of stages using McCabe-Thiele method (simplified).
        
        Parameters:
        x_D: Distillate composition
        x_F: Feed composition
        x_W: Bottoms composition
        R: Reflux ratio
        alpha: Relative volatility
        q: Feed quality
        
        Returns:
        Estimated number of stages (dimensionless)
        """
        # Simplified calculation - full McCabe-Thiele requires graphical solution
        N_min = Distillation.fenske_equation(alpha, x_D, x_W, None)
        # Gilliland correlation approximation
        R_min = Distillation.minimum_reflux_ratio(x_D, x_F, x_W, alpha, q)
        if R == R_min:
            return N_min
        X = (R - R_min) / (R + 1)
        Y = 1 - math.exp((1 + 54.4 * X) / (11 + 117.2 * X) * (X - 1) / (X ** 0.5))
        N = (N_min + Y) / (1 - Y)
        return N
    
    @staticmethod
    def operating_line_rectifying(R, x_D, x):
        """
        Calculate rectifying section operating line.
        
        Parameters:
        R: Reflux ratio
        x_D: Distillate composition
        x: Liquid composition
        
        Returns:
        Vapor composition (dimensionless)
        """
        return (R / (R + 1)) * x + (x_D / (R + 1))
    
    @staticmethod
    def operating_line_stripping(L_bar, V_bar, x_W, x):
        """
        Calculate stripping section operating line.
        
        Parameters:
        L_bar: Liquid flow rate in stripping section (mol/s)
        V_bar: Vapor flow rate in stripping section (mol/s)
        x_W: Bottoms composition
        x: Liquid composition
        
        Returns:
        Vapor composition (dimensionless)
        """
        return (L_bar / V_bar) * x - ((L_bar / V_bar - 1) * x_W)


class Extraction:
    """
    Class for liquid-liquid extraction calculations
    """
    
    @staticmethod
    def distribution_coefficient(C_extract, C_raffinate):
        """
        Calculate distribution coefficient.
        
        Parameters:
        C_extract: Concentration in extract phase (mol/m³)
        C_raffinate: Concentration in raffinate phase (mol/m³)
        
        Returns:
        Distribution coefficient (dimensionless)
        """
        if C_raffinate == 0:
            return float('inf')
        return C_extract / C_raffinate
    
    @staticmethod
    def selectivity(beta_A, beta_B):
        """
        Calculate selectivity.
        
        Parameters:
        beta_A: Distribution coefficient of A
        beta_B: Distribution coefficient of B
        
        Returns:
        Selectivity (dimensionless)
        """
        if beta_B == 0:
            return float('inf')
        return beta_A / beta_B
    
    @staticmethod
    def minimum_solvent_ratio(x_F, x_R, y_E, y_S):
        """
        Calculate minimum solvent-to-feed ratio.
        
        Parameters:
        x_F: Feed composition
        x_R: Raffinate composition
        y_E: Extract composition
        y_S: Solvent composition
        
        Returns:
        Minimum S/F ratio (dimensionless)
        """
        if y_E == y_S:
            return float('inf')
        return (x_F - x_R) / (y_E - y_S)
    
    @staticmethod
    def number_of_stages_extraction(x_F, x_N, y_1, y_S, beta, S_F):
        """
        Calculate number of stages for countercurrent extraction (Kremser equation).
        
        Parameters:
        x_F: Feed composition
        x_N: Final raffinate composition
        y_1: Final extract composition
        y_S: Solvent composition
        beta: Distribution coefficient
        S_F: Solvent-to-feed ratio
        
        Returns:
        Number of stages (dimensionless)
        """
        A = beta * S_F  # Extraction factor
        if A == 1:
            return (x_F - x_N) / (y_1 - y_S)
        return math.log((x_F - y_S / beta) / (x_N - y_S / beta)) / math.log(A)


class Drying:
    """
    Class for drying calculations
    """
    
    @staticmethod
    def moisture_content_wet_basis(m_water, m_total):
        """
        Calculate moisture content on wet basis.
        
        Parameters:
        m_water: Mass of water (kg)
        m_total: Total mass (kg)
        
        Returns:
        Moisture content (kg water/kg wet material)
        """
        return m_water / m_total
    
    @staticmethod
    def moisture_content_dry_basis(m_water, m_dry):
        """
        Calculate moisture content on dry basis.
        
        Parameters:
        m_water: Mass of water (kg)
        m_dry: Mass of dry solid (kg)
        
        Returns:
        Moisture content (kg water/kg dry solid)
        """
        return m_water / m_dry
    
    @staticmethod
    def drying_rate_constant_period(k, A, X, X_eq):
        """
        Calculate drying rate in constant rate period.
        
        Parameters:
        k: Mass transfer coefficient (kg/(m²·s))
        A: Surface area (m²)
        X: Moisture content (kg water/kg dry solid)
        X_eq: Equilibrium moisture content (kg water/kg dry solid)
        
        Returns:
        Drying rate (kg/s)
        """
        return k * A * (X - X_eq)
    
    @staticmethod
    def time_constant_rate_period(m_solid, A, k, X_initial, X_critical):
        """
        Calculate time for constant rate drying period.
        
        Parameters:
        m_solid: Mass of dry solid (kg)
        A: Surface area (m²)
        k: Mass transfer coefficient (kg/(m²·s))
        X_initial: Initial moisture content (kg water/kg dry solid)
        X_critical: Critical moisture content (kg water/kg dry solid)
        
        Returns:
        Time (s)
        """
        return (m_solid / (A * k)) * (X_initial - X_critical)
    
    @staticmethod
    def time_falling_rate_period(m_solid, A, k, X_critical, X_final, X_eq):
        """
        Calculate time for falling rate drying period (linear approximation).
        
        Parameters:
        m_solid: Mass of dry solid (kg)
        A: Surface area (m²)
        k: Mass transfer coefficient (kg/(m²·s))
        X_critical: Critical moisture content (kg water/kg dry solid)
        X_final: Final moisture content (kg water/kg dry solid)
        X_eq: Equilibrium moisture content (kg water/kg dry solid)
        
        Returns:
        Time (s)
        """
        return (m_solid / (A * k)) * (X_critical - X_eq) * \
               math.log((X_critical - X_eq) / (X_final - X_eq))


class MembraneSeparation:
    """
    Class for membrane separation calculations
    """
    
    @staticmethod
    def flux_solution_diffusion(P_perm, delta_P, delta_pi, L_m):
        """
        Calculate flux through membrane using solution-diffusion model.
        
        Parameters:
        P_perm: Permeability (mol·m/(m²·s·Pa))
        delta_P: Pressure difference (Pa)
        delta_pi: Osmotic pressure difference (Pa)
        L_m: Membrane thickness (m)
        
        Returns:
        Flux (mol/(m²·s))
        """
        return (P_perm / L_m) * (delta_P - delta_pi)
    
    @staticmethod
    def rejection_coefficient(C_feed, C_permeate):
        """
        Calculate rejection coefficient.
        
        Parameters:
        C_feed: Feed concentration (mol/m³)
        C_permeate: Permeate concentration (mol/m³)
        
        Returns:
        Rejection coefficient (dimensionless)
        """
        if C_feed == 0:
            return 0
        return 1 - (C_permeate / C_feed)
    
    @staticmethod
    def osmotic_pressure(C, T, R=8.314):
        """
        Calculate osmotic pressure using van't Hoff equation.
        
        Parameters:
        C: Molar concentration (mol/m³)
        T: Temperature (K)
        R: Gas constant (J/(mol·K))
        
        Returns:
        Osmotic pressure (Pa)
        """
        return C * R * T
    
    @staticmethod
    def selectivity_membrane(P_A, P_B):
        """
        Calculate membrane selectivity.
        
        Parameters:
        P_A: Permeability of component A (mol·m/(m²·s·Pa))
        P_B: Permeability of component B (mol·m/(m²·s·Pa))
        
        Returns:
        Selectivity (dimensionless)
        """
        if P_B == 0:
            return float('inf')
        return P_A / P_B


class MassTransferWithReaction:
    """
    Class for mass transfer with chemical reaction
    """
    
    @staticmethod
    def hatta_number(k_reaction, D_AB, k_L):
        """
        Calculate Hatta number (ratio of reaction rate to mass transfer rate).
        
        Parameters:
        k_reaction: Reaction rate constant (1/s)
        D_AB: Diffusion coefficient (m²/s)
        k_L: Liquid phase mass transfer coefficient (m/s)
        
        Returns:
        Hatta number (dimensionless)
        """
        return math.sqrt(k_reaction * D_AB) / k_L
    
    @staticmethod
    def enhancement_factor_instantaneous(Ha, E_i):
        """
        Calculate enhancement factor for instantaneous reaction.
        
        Parameters:
        Ha: Hatta number
        E_i: Instantaneous enhancement factor
        
        Returns:
        Enhancement factor (dimensionless)
        """
        if Ha < 0.3:
            return 1 + Ha
        elif Ha > 3:
            return E_i
        else:
            # Intermediate regime
            return 1 + (E_i - 1) * (1 - math.exp(-Ha))
    
    @staticmethod
    def thiele_modulus(k_reaction, D_eff, C_surface, r_p):
        """
        Calculate Thiele modulus for porous catalyst.
        
        Parameters:
        k_reaction: Reaction rate constant (1/s)
        D_eff: Effective diffusion coefficient (m²/s)
        C_surface: Surface concentration (mol/m³)
        r_p: Particle radius (m)
        
        Returns:
        Thiele modulus (dimensionless)
        """
        return r_p * math.sqrt(k_reaction / D_eff)
    
    @staticmethod
    def effectiveness_factor_porous(phi, n=1):
        """
        Calculate effectiveness factor for porous catalyst.
        
        Parameters:
        phi: Thiele modulus
        n: Reaction order (default 1)
        
        Returns:
        Effectiveness factor (dimensionless)
        """
        if phi < 0.1:
            return 1.0
        elif n == 1:
            return (3 / phi) * ((1 / math.tanh(phi)) - (1 / phi))
        else:
            # Approximation for other reaction orders
            return 1 / (1 + phi ** 2 / (2 * n + 1))
    
    @staticmethod
    def weisz_modulus(phi, eta):
        """
        Calculate Weisz modulus.
        
        Parameters:
        phi: Thiele modulus
        eta: Effectiveness factor
        
        Returns:
        Weisz modulus (dimensionless)
        """
        return phi ** 2 * eta

