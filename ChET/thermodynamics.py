"""
Chemical Engineering Thermodynamics Formulas
Comprehensive collection of classes for thermodynamics calculations
"""

import math


class EquationsOfState:
    """
    Class for equations of state calculations
    """
    
    @staticmethod
    def ideal_gas_pressure(n, T, V, R=8.314462618):
        """
        Calculate pressure using ideal gas law.
        
        Parameters:
        n: Number of moles (mol)
        T: Temperature (K)
        V: Volume (m³)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Pressure (Pa)
        """
        return (n * R * T) / V
    
    @staticmethod
    def ideal_gas_volume(n, T, P, R=8.314462618):
        """
        Calculate volume using ideal gas law.
        
        Parameters:
        n: Number of moles (mol)
        T: Temperature (K)
        P: Pressure (Pa)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Volume (m³)
        """
        return (n * R * T) / P
    
    @staticmethod
    def van_der_waals_pressure(n, T, V, a, b, R=8.314462618):
        """
        Calculate pressure using van der Waals equation.
        
        Parameters:
        n: Number of moles (mol)
        T: Temperature (K)
        V: Volume (m³)
        a: van der Waals parameter a (Pa·m⁶/mol²)
        b: van der Waals parameter b (m³/mol)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Pressure (Pa)
        """
        return (n * R * T) / (V - n * b) - (a * n ** 2) / (V ** 2)
    
    @staticmethod
    def compressibility_factor_van_der_waals(P, T, V, n, R=8.314462618):
        """
        Calculate compressibility factor using van der Waals equation.
        
        Parameters:
        P: Pressure (Pa)
        T: Temperature (K)
        V: Volume (m³)
        n: Number of moles (mol)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Compressibility factor Z (dimensionless)
        """
        return (P * V) / (n * R * T)
    
    @staticmethod
    def reduced_temperature(T, T_c):
        """
        Calculate reduced temperature.
        
        Parameters:
        T: Temperature (K)
        T_c: Critical temperature (K)
        
        Returns:
        Reduced temperature (dimensionless)
        """
        return T / T_c
    
    @staticmethod
    def reduced_pressure(P, P_c):
        """
        Calculate reduced pressure.
        
        Parameters:
        P: Pressure (Pa)
        P_c: Critical pressure (Pa)
        
        Returns:
        Reduced pressure (dimensionless)
        """
        return P / P_c
    
    @staticmethod
    def reduced_volume(V, V_c):
        """
        Calculate reduced volume.
        
        Parameters:
        V: Volume (m³)
        V_c: Critical volume (m³)
        
        Returns:
        Reduced volume (dimensionless)
        """
        return V / V_c


class ThermodynamicProperties:
    """
    Class for thermodynamic property calculations
    """
    
    @staticmethod
    def internal_energy_ideal_gas(Cv, T, T_ref=298.15, U_ref=0):
        """
        Calculate internal energy for ideal gas.
        
        Parameters:
        Cv: Heat capacity at constant volume (J/(mol·K))
        T: Temperature (K)
        T_ref: Reference temperature (K), default 298.15 K
        U_ref: Reference internal energy at T_ref (J/mol)
        
        Returns:
        Internal energy (J/mol)
        """
        return U_ref + Cv * (T - T_ref)
    
    @staticmethod
    def enthalpy_ideal_gas(Cp, T, T_ref=298.15, H_ref=0):
        """
        Calculate enthalpy for ideal gas.
        
        Parameters:
        Cp: Heat capacity at constant pressure (J/(mol·K))
        T: Temperature (K)
        T_ref: Reference temperature (K), default 298.15 K
        H_ref: Reference enthalpy at T_ref (J/mol)
        
        Returns:
        Enthalpy (J/mol)
        """
        return H_ref + Cp * (T - T_ref)
    
    @staticmethod
    def entropy_ideal_gas(Cp, T, T_ref=298.15, P=101325, P_ref=101325, S_ref=0, R=8.314462618):
        """
        Calculate entropy for ideal gas (isobaric).
        
        Parameters:
        Cp: Heat capacity at constant pressure (J/(mol·K))
        T: Temperature (K)
        T_ref: Reference temperature (K), default 298.15 K
        P: Pressure (Pa)
        P_ref: Reference pressure (Pa), default 101325 Pa
        S_ref: Reference entropy at T_ref, P_ref (J/(mol·K))
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Entropy (J/(mol·K))
        """
        return S_ref + Cp * math.log(T / T_ref) - R * math.log(P / P_ref)
    
    @staticmethod
    def entropy_ideal_gas_constant_volume(Cv, T, T_ref=298.15, V=1, V_ref=1, S_ref=0, R=8.314462618):
        """
        Calculate entropy for ideal gas at constant volume.
        
        Parameters:
        Cv: Heat capacity at constant volume (J/(mol·K))
        T: Temperature (K)
        T_ref: Reference temperature (K), default 298.15 K
        V: Volume (m³)
        V_ref: Reference volume (m³)
        S_ref: Reference entropy at T_ref, V_ref (J/(mol·K))
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Entropy (J/(mol·K))
        """
        return S_ref + Cv * math.log(T / T_ref) + R * math.log(V / V_ref)
    
    @staticmethod
    def gibbs_free_energy(H, T, S):
        """
        Calculate Gibbs free energy.
        
        Parameters:
        H: Enthalpy (J/mol)
        T: Temperature (K)
        S: Entropy (J/(mol·K))
        
        Returns:
        Gibbs free energy (J/mol)
        """
        return H - T * S
    
    @staticmethod
    def helmholtz_free_energy(U, T, S):
        """
        Calculate Helmholtz free energy.
        
        Parameters:
        U: Internal energy (J/mol)
        T: Temperature (K)
        S: Entropy (J/(mol·K))
        
        Returns:
        Helmholtz free energy (J/mol)
        """
        return U - T * S
    
    @staticmethod
    def cp_cv_relationship(Cp, Cv, R=8.314462618):
        """
        Relationship between Cp and Cv for ideal gas.
        
        Parameters:
        Cp: Heat capacity at constant pressure (J/(mol·K))
        Cv: Heat capacity at constant volume (J/(mol·K))
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Difference Cp - Cv (should equal R for ideal gas) (J/(mol·K))
        """
        return Cp - Cv
    
    @staticmethod
    def gamma_ratio(Cp, Cv):
        """
        Calculate specific heat ratio (gamma).
        
        Parameters:
        Cp: Heat capacity at constant pressure (J/(mol·K))
        Cv: Heat capacity at constant volume (J/(mol·K))
        
        Returns:
        Specific heat ratio γ = Cp/Cv (dimensionless)
        """
        return Cp / Cv


class PhaseEquilibria:
    """
    Class for phase equilibrium calculations
    """
    
    @staticmethod
    def antoine_equation(T, A, B, C):
        """
        Calculate vapor pressure using Antoine equation.
        
        Parameters:
        T: Temperature (K)
        A: Antoine parameter A (dimensionless)
        B: Antoine parameter B (K)
        C: Antoine parameter C (K)
        
        Returns:
        Vapor pressure (Pa)
        Note: If T is in °C, use C=0. For K, C is typically negative
        """
        return 10 ** (A - B / (T + C)) * 100  # Convert from bar to Pa (assuming standard Antoine)
    
    @staticmethod
    def clausius_clapeyron(T, T_ref, P_ref, delta_H_vap, R=8.314462618):
        """
        Calculate vapor pressure using Clausius-Clapeyron equation.
        
        Parameters:
        T: Temperature (K)
        T_ref: Reference temperature (K)
        P_ref: Reference vapor pressure at T_ref (Pa)
        delta_H_vap: Enthalpy of vaporization (J/mol)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Vapor pressure at temperature T (Pa)
        """
        return P_ref * math.exp((delta_H_vap / R) * ((1 / T_ref) - (1 / T)))
    
    @staticmethod
    def clausius_clapeyron_temperature(P, T_ref, P_ref, delta_H_vap, R=8.314462618):
        """
        Calculate temperature from vapor pressure using Clausius-Clapeyron equation.
        
        Parameters:
        P: Vapor pressure (Pa)
        T_ref: Reference temperature (K)
        P_ref: Reference vapor pressure at T_ref (Pa)
        delta_H_vap: Enthalpy of vaporization (J/mol)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Temperature (K)
        """
        return 1 / ((1 / T_ref) - (R / delta_H_vap) * math.log(P / P_ref))
    
    @staticmethod
    def raoults_law_vapor_pressure(x_i, P_i_sat):
        """
        Calculate partial vapor pressure using Raoult's law.
        
        Parameters:
        x_i: Mole fraction of component i in liquid phase
        P_i_sat: Saturation vapor pressure of pure component i (Pa)
        
        Returns:
        Partial vapor pressure of component i (Pa)
        """
        return x_i * P_i_sat
    
    @staticmethod
    def raoults_law_binary_total_pressure(x_A, P_A_sat, P_B_sat):
        """
        Calculate total vapor pressure for binary mixture using Raoult's law.
        
        Parameters:
        x_A: Mole fraction of component A in liquid phase
        P_A_sat: Saturation vapor pressure of pure A (Pa)
        P_B_sat: Saturation vapor pressure of pure B (Pa)
        
        Returns:
        Total vapor pressure (Pa)
        """
        return x_A * P_A_sat + (1 - x_A) * P_B_sat
    
    @staticmethod
    def raoults_law_vapor_composition(x_A, P_A_sat, P_B_sat):
        """
        Calculate vapor phase mole fraction for binary mixture using Raoult's law.
        
        Parameters:
        x_A: Mole fraction of component A in liquid phase
        P_A_sat: Saturation vapor pressure of pure A (Pa)
        P_B_sat: Saturation vapor pressure of pure B (Pa)
        
        Returns:
        Mole fraction of A in vapor phase
        """
        P_total = PhaseEquilibria.raoults_law_binary_total_pressure(x_A, P_A_sat, P_B_sat)
        return (x_A * P_A_sat) / P_total
    
    @staticmethod
    def henrys_law(x_i, H_i):
        """
        Calculate partial pressure using Henry's law.
        
        Parameters:
        x_i: Mole fraction of component i in liquid phase
        H_i: Henry's law constant for component i (Pa)
        
        Returns:
        Partial pressure of component i (Pa)
        """
        return H_i * x_i
    
    @staticmethod
    def relative_volatility(alpha_AB, P_A_sat, P_B_sat):
        """
        Calculate relative volatility.
        
        Parameters:
        alpha_AB: Relative volatility of A to B (to be calculated)
        P_A_sat: Saturation vapor pressure of A (Pa)
        P_B_sat: Saturation vapor pressure of B (Pa)
        
        Returns:
        Relative volatility α_AB (dimensionless)
        """
        return P_A_sat / P_B_sat


class PropertyRelations:
    """
    Class for thermodynamic property relationships
    """
    
    @staticmethod
    def heat_capacity_polynomial(T, a, b, c, d):
        """
        Calculate heat capacity using polynomial correlation: Cp = a + b*T + c*T² + d*T³.
        
        Parameters:
        T: Temperature (K)
        a: Constant coefficient (J/(mol·K))
        b: Linear coefficient (J/(mol·K²))
        c: Quadratic coefficient (J/(mol·K³))
        d: Cubic coefficient (J/(mol·K⁴))
        
        Returns:
        Heat capacity at constant pressure (J/(mol·K))
        """
        return a + b * T + c * T ** 2 + d * T ** 3
    
    @staticmethod
    def heat_capacity_average(Cp_T1, Cp_T2):
        """
        Calculate average heat capacity.
        
        Parameters:
        Cp_T1: Heat capacity at temperature T1 (J/(mol·K))
        Cp_T2: Heat capacity at temperature T2 (J/(mol·K))
        
        Returns:
        Average heat capacity (J/(mol·K))
        """
        return (Cp_T1 + Cp_T2) / 2
    
    @staticmethod
    def joule_thomson_coefficient(mu_JT, Cp, V, T, alpha):
        """
        Calculate Joule-Thomson coefficient.
        
        Parameters:
        mu_JT: Joule-Thomson coefficient (to be calculated) (K/Pa)
        Cp: Heat capacity at constant pressure (J/(mol·K))
        V: Molar volume (m³/mol)
        T: Temperature (K)
        alpha: Coefficient of thermal expansion (1/K)
        
        Returns:
        Joule-Thomson coefficient (K/Pa)
        """
        return (1 / Cp) * (T * alpha - V)
    
    @staticmethod
    def thermal_expansion_coefficient(alpha, rho, T_ref, rho_ref):
        """
        Estimate thermal expansion coefficient from density data.
        
        Parameters:
        alpha: Thermal expansion coefficient (to be calculated) (1/K)
        rho: Density at temperature T (kg/m³)
        rho_ref: Reference density at T_ref (kg/m³)
        T_ref: Reference temperature (K)
        T: Temperature (K)
        
        Note: This is an approximate method. Typically alpha = (1/V)(dV/dT)_P
        """
        # Approximate: alpha ≈ -(1/rho)(drho/dT)
        # For liquids, often: rho ≈ rho_ref * (1 - alpha * (T - T_ref))
        pass  # Implementation depends on data availability


class ChemicalReactions:
    """
    Class for chemical reaction thermodynamics
    """
    
    @staticmethod
    def gibbs_free_energy_reaction(delta_G_f_products, nu_products, delta_G_f_reactants, nu_reactants):
        """
        Calculate Gibbs free energy of reaction.
        
        Parameters:
        delta_G_f_products: List of standard Gibbs free energies of formation for products (J/mol)
        nu_products: List of stoichiometric coefficients for products
        delta_G_f_reactants: List of standard Gibbs free energies of formation for reactants (J/mol)
        nu_reactants: List of stoichiometric coefficients for reactants
        
        Returns:
        Gibbs free energy of reaction (J/mol)
        """
        G_products = sum(nu * delta_G for nu, delta_G in zip(nu_products, delta_G_f_products))
        G_reactants = sum(nu * delta_G for nu, delta_G in zip(nu_reactants, delta_G_f_reactants))
        return G_products - G_reactants
    
    @staticmethod
    def enthalpy_reaction(delta_H_f_products, nu_products, delta_H_f_reactants, nu_reactants):
        """
        Calculate enthalpy of reaction.
        
        Parameters:
        delta_H_f_products: List of standard enthalpies of formation for products (J/mol)
        nu_products: List of stoichiometric coefficients for products
        delta_H_f_reactants: List of standard enthalpies of formation for reactants (J/mol)
        nu_reactants: List of stoichiometric coefficients for reactants
        
        Returns:
        Enthalpy of reaction (J/mol)
        """
        H_products = sum(nu * delta_H for nu, delta_H in zip(nu_products, delta_H_f_products))
        H_reactants = sum(nu * delta_H for nu, delta_H in zip(nu_reactants, delta_H_f_reactants))
        return H_products - H_reactants
    
    @staticmethod
    def entropy_reaction(S_products, nu_products, S_reactants, nu_reactants):
        """
        Calculate entropy of reaction.
        
        Parameters:
        S_products: List of standard entropies for products (J/(mol·K))
        nu_products: List of stoichiometric coefficients for products
        S_reactants: List of standard entropies for reactants (J/(mol·K))
        nu_reactants: List of stoichiometric coefficients for reactants
        
        Returns:
        Entropy of reaction (J/(mol·K))
        """
        S_prod_sum = sum(nu * S for nu, S in zip(nu_products, S_products))
        S_react_sum = sum(nu * S for nu, S in zip(nu_reactants, S_reactants))
        return S_prod_sum - S_react_sum
    
    @staticmethod
    def equilibrium_constant_delta_G(delta_G_rxn, T, R=8.314462618):
        """
        Calculate equilibrium constant from Gibbs free energy of reaction.
        
        Parameters:
        delta_G_rxn: Gibbs free energy of reaction (J/mol)
        T: Temperature (K)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Equilibrium constant K (dimensionless)
        """
        return math.exp(-delta_G_rxn / (R * T))
    
    @staticmethod
    def van_t_hoff_equation(K1, K2, T1, T2, delta_H_rxn, R=8.314462618):
        """
        Calculate equilibrium constant at different temperature using van't Hoff equation.
        
        Parameters:
        K1: Equilibrium constant at temperature T1
        K2: Equilibrium constant at temperature T2 (to be calculated)
        T1: Temperature 1 (K)
        T2: Temperature 2 (K)
        delta_H_rxn: Enthalpy of reaction (J/mol)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Equilibrium constant K2 at temperature T2
        """
        return K1 * math.exp((delta_H_rxn / R) * ((1 / T1) - (1 / T2)))
    
    @staticmethod
    def reaction_quotient(activities_products, nu_products, activities_reactants, nu_reactants):
        """
        Calculate reaction quotient Q.
        
        Parameters:
        activities_products: List of activities for products
        nu_products: List of stoichiometric coefficients for products
        activities_reactants: List of activities for reactants
        nu_reactants: List of stoichiometric coefficients for reactants
        
        Returns:
        Reaction quotient Q (dimensionless)
        """
        Q_products = math.prod(a ** nu for a, nu in zip(activities_products, nu_products))
        Q_reactants = math.prod(a ** nu for a, nu in zip(activities_reactants, nu_reactants))
        return Q_products / Q_reactants
    
    @staticmethod
    def gibbs_free_energy_change(delta_G_rxn_standard, Q, T, R=8.314462618):
        """
        Calculate Gibbs free energy change for non-standard conditions.
        
        Parameters:
        delta_G_rxn_standard: Standard Gibbs free energy of reaction (J/mol)
        Q: Reaction quotient
        T: Temperature (K)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Gibbs free energy change (J/mol)
        """
        return delta_G_rxn_standard + R * T * math.log(Q)


class WorkAndHeat:
    """
    Class for work and heat calculations
    """
    
    @staticmethod
    def work_isothermal_ideal_gas(n, T, V1, V2, R=8.314462618):
        """
        Calculate work for isothermal reversible expansion of ideal gas.
        
        Parameters:
        n: Number of moles (mol)
        T: Temperature (K)
        V1: Initial volume (m³)
        V2: Final volume (m³)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Work (J) - negative for expansion, positive for compression
        """
        return -n * R * T * math.log(V2 / V1)
    
    @staticmethod
    def work_isobaric(P, V1, V2):
        """
        Calculate work for isobaric process.
        
        Parameters:
        P: Pressure (Pa)
        V1: Initial volume (m³)
        V2: Final volume (m³)
        
        Returns:
        Work (J) - negative for expansion, positive for compression
        """
        return -P * (V2 - V1)
    
    @staticmethod
    def work_adiabatic_ideal_gas(Cv, T1, T2):
        """
        Calculate work for adiabatic process of ideal gas.
        
        Parameters:
        Cv: Heat capacity at constant volume (J/(mol·K))
        T1: Initial temperature (K)
        T2: Final temperature (K)
        
        Returns:
        Work (J/mol)
        """
        return Cv * (T2 - T1)
    
    @staticmethod
    def work_polytropic(P1, V1, P2, V2, n_poly):
        """
        Calculate work for polytropic process.
        
        Parameters:
        P1: Initial pressure (Pa)
        V1: Initial volume (m³)
        P2: Final pressure (Pa)
        V2: Final volume (m³)
        n_poly: Polytropic exponent
        
        Returns:
        Work (J)
        """
        if n_poly == 1:
            # Isothermal process
            return -P1 * V1 * math.log(V2 / V1)
        else:
            return -(P2 * V2 - P1 * V1) / (1 - n_poly)
    
    @staticmethod
    def heat_constant_pressure(Cp, n, T1, T2):
        """
        Calculate heat transfer at constant pressure.
        
        Parameters:
        Cp: Heat capacity at constant pressure (J/(mol·K))
        n: Number of moles (mol)
        T1: Initial temperature (K)
        T2: Final temperature (K)
        
        Returns:
        Heat (J) - positive for heat added, negative for heat removed
        """
        return n * Cp * (T2 - T1)
    
    @staticmethod
    def heat_constant_volume(Cv, n, T1, T2):
        """
        Calculate heat transfer at constant volume.
        
        Parameters:
        Cv: Heat capacity at constant volume (J/(mol·K))
        n: Number of moles (mol)
        T1: Initial temperature (K)
        T2: Final temperature (K)
        
        Returns:
        Heat (J) - positive for heat added, negative for heat removed
        """
        return n * Cv * (T2 - T1)
    
    @staticmethod
    def carnot_efficiency(T_hot, T_cold):
        """
        Calculate Carnot cycle efficiency.
        
        Parameters:
        T_hot: Temperature of hot reservoir (K)
        T_cold: Temperature of cold reservoir (K)
        
        Returns:
        Efficiency (dimensionless, 0-1)
        """
        if T_hot <= T_cold:
            return 0
        return 1 - (T_cold / T_hot)
    
    @staticmethod
    def cop_refrigerator(T_cold, T_hot):
        """
        Calculate coefficient of performance for refrigerator.
        
        Parameters:
        T_cold: Temperature of cold reservoir (K)
        T_hot: Temperature of hot reservoir (K)
        
        Returns:
        COP (dimensionless)
        """
        if T_hot <= T_cold:
            return 0
        return T_cold / (T_hot - T_cold)
    
    @staticmethod
    def cop_heat_pump(T_hot, T_cold):
        """
        Calculate coefficient of performance for heat pump.
        
        Parameters:
        T_hot: Temperature of hot reservoir (K)
        T_cold: Temperature of cold reservoir (K)
        
        Returns:
        COP (dimensionless)
        """
        if T_hot <= T_cold:
            return 0
        return T_hot / (T_hot - T_cold)


class Compressibility:
    """
    Class for compressibility factor calculations
    """
    
    @staticmethod
    def compressibility_factor(P, V, n, T, R=8.314462618):
        """
        Calculate compressibility factor.
        
        Parameters:
        P: Pressure (Pa)
        V: Volume (m³)
        n: Number of moles (mol)
        T: Temperature (K)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Compressibility factor Z (dimensionless)
        """
        return (P * V) / (n * R * T)
    
    @staticmethod
    def acentric_factor(P_vap, P_c, T, T_c):
        """
        Calculate acentric factor from vapor pressure data.
        
        Parameters:
        P_vap: Vapor pressure at T = 0.7*T_c (Pa)
        P_c: Critical pressure (Pa)
        T: Temperature (should be 0.7*T_c) (K)
        T_c: Critical temperature (K)
        
        Returns:
        Acentric factor ω (dimensionless)
        """
        if abs(T - 0.7 * T_c) > 1e-3:
            print("Warning: T should be approximately 0.7*T_c for accurate acentric factor")
        return -math.log10(P_vap / P_c) - 1.0


class Mixing:
    """
    Class for mixing and solution thermodynamics
    """
    
    @staticmethod
    def ideal_solution_entropy(n_i, x_i, R=8.314462618):
        """
        Calculate entropy of mixing for ideal solution.
        
        Parameters:
        n_i: List of moles of each component (mol)
        x_i: List of mole fractions
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Entropy of mixing (J/K)
        """
        n_total = sum(n_i)
        return -n_total * R * sum(x * math.log(x) for x in x_i if x > 0)
    
    @staticmethod
    def ideal_solution_gibbs_free_energy(T, n_i, x_i, R=8.314462618):
        """
        Calculate Gibbs free energy of mixing for ideal solution.
        
        Parameters:
        T: Temperature (K)
        n_i: List of moles of each component (mol)
        x_i: List of mole fractions
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Gibbs free energy of mixing (J)
        """
        n_total = sum(n_i)
        return n_total * R * T * sum(x * math.log(x) for x in x_i if x > 0)
    
    @staticmethod
    def activity_coefficient_margules_2component(x_A, A12, A21):
        """
        Calculate activity coefficients using Margules equation (2 components).
        
        Parameters:
        x_A: Mole fraction of component A
        A12: Margules parameter A12
        A21: Margules parameter A21
        
        Returns:
        Tuple (gamma_A, gamma_B) - activity coefficients
        """
        x_B = 1 - x_A
        ln_gamma_A = x_B ** 2 * (A12 + 2 * (A21 - A12) * x_A)
        ln_gamma_B = x_A ** 2 * (A21 + 2 * (A12 - A21) * x_B)
        return (math.exp(ln_gamma_A), math.exp(ln_gamma_B))
    
    @staticmethod
    def fugacity_coefficient(Z, P, P_ref=101325):
        """
        Estimate fugacity coefficient from compressibility factor.
        
        Parameters:
        Z: Compressibility factor
        P: Pressure (Pa)
        P_ref: Reference pressure (Pa), default 101325 Pa
        
        Returns:
        Fugacity coefficient (dimensionless)
        """
        # Simplified approximation: phi = exp((Z-1)*P/P_ref)
        # More accurate methods require integration
        return math.exp((Z - 1) * P / P_ref)


class HeatEngines:
    """
    Class for heat engine and refrigeration cycle calculations
    """
    
    @staticmethod
    def otto_cycle_efficiency(r_compression, gamma):
        """
        Calculate efficiency of Otto cycle (spark-ignition engine).
        
        Parameters:
        r_compression: Compression ratio (V_max/V_min)
        gamma: Specific heat ratio (Cp/Cv)
        
        Returns:
        Efficiency (dimensionless, 0-1)
        """
        return 1 - (1 / (r_compression ** (gamma - 1)))
    
    @staticmethod
    def diesel_cycle_efficiency(r_compression, r_cutoff, gamma):
        """
        Calculate efficiency of Diesel cycle.
        
        Parameters:
        r_compression: Compression ratio (V_max/V_min)
        r_cutoff: Cutoff ratio (V3/V2)
        gamma: Specific heat ratio (Cp/Cv)
        
        Returns:
        Efficiency (dimensionless, 0-1)
        """
        return 1 - (1 / (r_compression ** (gamma - 1))) * ((r_cutoff ** gamma - 1) / (gamma * (r_cutoff - 1)))
    
    @staticmethod
    def brayton_cycle_efficiency(r_pressure, gamma):
        """
        Calculate efficiency of Brayton cycle (gas turbine).
        
        Parameters:
        r_pressure: Pressure ratio (P_max/P_min)
        gamma: Specific heat ratio (Cp/Cv)
        
        Returns:
        Efficiency (dimensionless, 0-1)
        """
        return 1 - (1 / (r_pressure ** ((gamma - 1) / gamma)))


class ThermodynamicCycles:
    """
    Class for thermodynamic cycle analysis
    """
    
    @staticmethod
    def rankine_cycle_work_pump(P1, P2, rho_water=1000, eta_pump=1.0):
        """
        Calculate pump work in Rankine cycle.
        
        Parameters:
        P1: Low pressure (Pa)
        P2: High pressure (Pa)
        rho_water: Density of water (kg/m³)
        eta_pump: Pump efficiency (0-1)
        
        Returns:
        Specific work (J/kg) - work per unit mass
        """
        return (P2 - P1) / (rho_water * eta_pump)
    
    @staticmethod
    def rankine_cycle_work_turbine(h_in, h_out, eta_turbine=1.0):
        """
        Calculate turbine work in Rankine cycle.
        
        Parameters:
        h_in: Enthalpy at turbine inlet (J/kg)
        h_out: Enthalpy at turbine outlet (J/kg)
        eta_turbine: Turbine efficiency (0-1)
        
        Returns:
        Specific work (J/kg) - work per unit mass
        """
        return eta_turbine * (h_in - h_out)
    
    @staticmethod
    def rankine_cycle_efficiency(W_turbine, W_pump, Q_boiler):
        """
        Calculate Rankine cycle efficiency.
        
        Parameters:
        W_turbine: Turbine work (J/kg)
        W_pump: Pump work (J/kg)
        Q_boiler: Heat added in boiler (J/kg)
        
        Returns:
        Efficiency (dimensionless, 0-1)
        """
        if Q_boiler == 0:
            return 0
        return (W_turbine - W_pump) / Q_boiler
