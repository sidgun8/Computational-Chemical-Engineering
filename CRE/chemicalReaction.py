"""
Chemical Reaction Engineering Formulas
Comprehensive collection of classes for chemical reaction engineering calculations
"""

import math


class ReactionKinetics:
    """
    Class for reaction rate laws and kinetics
    """
    
    @staticmethod
    def arrhenius_equation(A, E_a, T, R=8.314462618):
        """
        Calculate reaction rate constant using Arrhenius equation.
        
        Parameters:
        A: Pre-exponential factor (frequency factor)
        E_a: Activation energy (J/mol)
        T: Temperature (K)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Rate constant k (units depend on reaction order)
        """
        return A * math.exp(-E_a / (R * T))
    
    @staticmethod
    def zero_order_rate(r, k, C=None):
        """
        Zero-order reaction rate: r = k
        
        Parameters:
        r: Reaction rate (mol/(L·s) or mol/(m³·s))
        k: Rate constant (mol/(L·s) or mol/(m³·s))
        C: Concentration (not used for zero-order, included for consistency)
        
        Returns:
        Reaction rate (mol/(L·s) or mol/(m³·s))
        """
        return k
    
    @staticmethod
    def first_order_rate(k, C_A):
        """
        First-order reaction rate: r = k*C_A
        
        Parameters:
        k: Rate constant (1/s)
        C_A: Concentration of A (mol/L or mol/m³)
        
        Returns:
        Reaction rate (mol/(L·s) or mol/(m³·s))
        """
        return k * C_A
    
    @staticmethod
    def second_order_rate(k, C_A, C_B=None):
        """
        Second-order reaction rate.
        
        If C_B is None: r = k*C_A² (A + A -> products)
        If C_B is provided: r = k*C_A*C_B (A + B -> products)
        
        Parameters:
        k: Rate constant (L/(mol·s) or m³/(mol·s))
        C_A: Concentration of A (mol/L or mol/m³)
        C_B: Concentration of B (mol/L or mol/m³), optional
        
        Returns:
        Reaction rate (mol/(L·s) or mol/(m³·s))
        """
        if C_B is None:
            return k * C_A ** 2
        else:
            return k * C_A * C_B
    
    @staticmethod
    def n_order_rate(k, C_A, n):
        """
        n-th order reaction rate: r = k*C_A^n
        
        Parameters:
        k: Rate constant (units depend on n)
        C_A: Concentration of A (mol/L or mol/m³)
        n: Reaction order (dimensionless)
        
        Returns:
        Reaction rate (mol/(L·s) or mol/(m³·s))
        """
        return k * (C_A ** n)
    
    @staticmethod
    def langmuir_hinshelwood_rate(k, K_A, K_B, C_A, C_B, P_A, P_B=None):
        """
        Langmuir-Hinshelwood rate expression for surface reactions.
        Simplified version: r = k*θ_A*θ_B where θ_i = K_i*P_i/(1 + ΣK_i*P_i)
        
        Parameters:
        k: Rate constant
        K_A: Adsorption equilibrium constant for A
        K_B: Adsorption equilibrium constant for B
        C_A: Surface concentration of A (or use P_A)
        C_B: Surface concentration of B (or use P_B)
        P_A: Partial pressure of A (Pa or atm)
        P_B: Partial pressure of B (Pa or atm), optional
        
        Returns:
        Reaction rate
        """
        if P_B is None:
            # Single reactant case
            theta_A = K_A * P_A / (1 + K_A * P_A)
            return k * theta_A
        else:
            # Two reactant case
            denominator = 1 + K_A * P_A + K_B * P_B
            theta_A = K_A * P_A / denominator
            theta_B = K_B * P_B / denominator
            return k * theta_A * theta_B
    
    @staticmethod
    def michaelis_menten_rate(V_max, K_M, C_S):
        """
        Michaelis-Menten kinetics for enzyme-catalyzed reactions.
        r = (V_max * C_S) / (K_M + C_S)
        
        Parameters:
        V_max: Maximum reaction rate (mol/(L·s))
        K_M: Michaelis constant (mol/L)
        C_S: Substrate concentration (mol/L)
        
        Returns:
        Reaction rate (mol/(L·s))
        """
        return (V_max * C_S) / (K_M + C_S)
    
    @staticmethod
    def temperature_dependence_k2(k1, E_a, T1, T2, R=8.314462618):
        """
        Calculate rate constant at T2 given k1 at T1.
        k2 = k1 * exp((E_a/R) * (1/T1 - 1/T2))
        
        Parameters:
        k1: Rate constant at T1
        E_a: Activation energy (J/mol)
        T1: Temperature 1 (K)
        T2: Temperature 2 (K)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Rate constant at T2
        """
        return k1 * math.exp((E_a / R) * (1/T1 - 1/T2))


class ReactorDesign:
    """
    Class for reactor design equations
    """
    
    @staticmethod
    def batch_reactor_time_zero_order(C_A0, C_A, k):
        """
        Time required for zero-order reaction in batch reactor.
        t = (C_A0 - C_A) / k
        
        Parameters:
        C_A0: Initial concentration (mol/L or mol/m³)
        C_A: Final concentration (mol/L or mol/m³)
        k: Rate constant (mol/(L·s) or mol/(m³·s))
        
        Returns:
        Time (s)
        """
        return (C_A0 - C_A) / k
    
    @staticmethod
    def batch_reactor_time_first_order(C_A0, C_A, k):
        """
        Time required for first-order reaction in batch reactor.
        t = (1/k) * ln(C_A0/C_A)
        
        Parameters:
        C_A0: Initial concentration (mol/L or mol/m³)
        C_A: Final concentration (mol/L or mol/m³)
        k: Rate constant (1/s)
        
        Returns:
        Time (s)
        """
        if C_A <= 0:
            return float('inf')
        return (1 / k) * math.log(C_A0 / C_A)
    
    @staticmethod
    def batch_reactor_time_second_order(C_A0, C_A, k):
        """
        Time required for second-order reaction (A + A -> products) in batch reactor.
        t = (1/k) * (1/C_A - 1/C_A0)
        
        Parameters:
        C_A0: Initial concentration (mol/L or mol/m³)
        C_A: Final concentration (mol/L or mol/m³)
        k: Rate constant (L/(mol·s) or m³/(mol·s))
        
        Returns:
        Time (s)
        """
        if C_A <= 0:
            return float('inf')
        return (1 / k) * (1 / C_A - 1 / C_A0)
    
    @staticmethod
    def batch_reactor_time_n_order(C_A0, C_A, k, n):
        """
        Time required for n-th order reaction in batch reactor.
        For n ≠ 1: t = (1/(k*(n-1))) * (C_A^(1-n) - C_A0^(1-n))
        
        Parameters:
        C_A0: Initial concentration (mol/L or mol/m³)
        C_A: Final concentration (mol/L or mol/m³)
        k: Rate constant (units depend on n)
        n: Reaction order
        
        Returns:
        Time (s)
        """
        if n == 1:
            return ReactorDesign.batch_reactor_time_first_order(C_A0, C_A, k)
        if C_A <= 0:
            return float('inf')
        return (1 / (k * (n - 1))) * (C_A ** (1 - n) - C_A0 ** (1 - n))
    
    @staticmethod
    def cstr_volume_zero_order(F_A0, X, k):
        """
        Volume required for zero-order reaction in CSTR.
        V = (F_A0 * X) / k
        
        Parameters:
        F_A0: Inlet molar flow rate of A (mol/s)
        X: Conversion
        k: Rate constant (mol/(L·s) or mol/(m³·s))
        
        Returns:
        Reactor volume (L or m³)
        """
        return (F_A0 * X) / k
    
    @staticmethod
    def cstr_volume_first_order(F_A0, X, k, C_A0, v0):
        """
        Volume required for first-order reaction in CSTR.
        V = (F_A0 * X) / (k * C_A * v0) where C_A = C_A0*(1-X)
        V = (v0 * X) / (k * (1 - X))
        
        Parameters:
        F_A0: Inlet molar flow rate of A (mol/s)
        X: Conversion
        k: Rate constant (1/s)
        C_A0: Inlet concentration of A (mol/L or mol/m³)
        v0: Volumetric flow rate (L/s or m³/s)
        
        Returns:
        Reactor volume (L or m³)
        """
        return (v0 * X) / (k * (1 - X))
    
    @staticmethod
    def cstr_volume_second_order(F_A0, X, k, C_A0, v0):
        """
        Volume required for second-order reaction (A + A -> products) in CSTR.
        V = (v0 * X) / (k * C_A0 * (1 - X)²)
        
        Parameters:
        F_A0: Inlet molar flow rate of A (mol/s)
        X: Conversion
        k: Rate constant (L/(mol·s) or m³/(mol·s))
        C_A0: Inlet concentration of A (mol/L or mol/m³)
        v0: Volumetric flow rate (L/s or m³/s)
        
        Returns:
        Reactor volume (L or m³)
        """
        return (v0 * X) / (k * C_A0 * (1 - X) ** 2)
    
    @staticmethod
    def pfr_volume_zero_order(F_A0, X, k):
        """
        Volume required for zero-order reaction in PFR.
        V = (F_A0 * X) / k
        
        Parameters:
        F_A0: Inlet molar flow rate of A (mol/s)
        X: Conversion
        k: Rate constant (mol/(L·s) or mol/(m³·s))
        
        Returns:
        Reactor volume (L or m³)
        """
        return (F_A0 * X) / k
    
    @staticmethod
    def pfr_volume_first_order(F_A0, X, k, v0):
        """
        Volume required for first-order reaction in PFR.
        V = (v0 / k) * ln(1 / (1 - X))
        
        Parameters:
        F_A0: Inlet molar flow rate of A (mol/s)
        X: Conversion
        k: Rate constant (1/s)
        v0: Volumetric flow rate (L/s or m³/s)
        
        Returns:
        Reactor volume (L or m³)
        """
        if X >= 1:
            return float('inf')
        return (v0 / k) * math.log(1 / (1 - X))
    
    @staticmethod
    def pfr_volume_second_order(F_A0, X, k, C_A0, v0):
        """
        Volume required for second-order reaction (A + A -> products) in PFR.
        V = (v0 / (k * C_A0)) * (X / (1 - X))
        
        Parameters:
        F_A0: Inlet molar flow rate of A (mol/s)
        X: Conversion
        k: Rate constant (L/(mol·s) or m³/(mol·s))
        C_A0: Inlet concentration of A (mol/L or mol/m³)
        v0: Volumetric flow rate (L/s or m³/s)
        
        Returns:
        Reactor volume (L or m³)
        """
        if X >= 1:
            return float('inf')
        return (v0 / (k * C_A0)) * (X / (1 - X))
    
    @staticmethod
    def space_time(V, v0):
        """
        Calculate space time (mean residence time).
        tau = V / v0
        
        Parameters:
        V: Reactor volume (L or m³)
        v0: Volumetric flow rate (L/s or m³/s)
        
        Returns:
        Space time (s)
        """
        return V / v0
    
    @staticmethod
    def space_velocity(v0, V):
        """
        Calculate space velocity.
        SV = v0 / V
        
        Parameters:
        v0: Volumetric flow rate (L/s or m³/s)
        V: Reactor volume (L or m³)
        
        Returns:
        Space velocity (1/s)
        """
        return v0 / V
    
    @staticmethod
    def conversion_from_conc(C_A0, C_A):
        """
        Calculate conversion from concentrations.
        X = (C_A0 - C_A) / C_A0
        
        Parameters:
        C_A0: Initial/inlet concentration (mol/L or mol/m³)
        C_A: Final/outlet concentration (mol/L or mol/m³)
        
        Returns:
        Conversion (0 to 1)
        """
        if C_A0 == 0:
            return 0
        return (C_A0 - C_A) / C_A0
    
    @staticmethod
    def conversion_from_molar_flow(F_A0, F_A):
        """
        Calculate conversion from molar flow rates.
        X = (F_A0 - F_A) / F_A0
        
        Parameters:
        F_A0: Inlet molar flow rate (mol/s)
        F_A: Outlet molar flow rate (mol/s)
        
        Returns:
        Conversion (0 to 1)
        """
        if F_A0 == 0:
            return 0
        return (F_A0 - F_A) / F_A0


class ReactionEquilibrium:
    """
    Class for chemical equilibrium calculations
    """
    
    @staticmethod
    def equilibrium_constant_from_gibbs(delta_G, T, R=8.314462618):
        """
        Calculate equilibrium constant from Gibbs free energy change.
        K = exp(-ΔG° / (R*T))
        
        Parameters:
        delta_G: Standard Gibbs free energy change (J/mol)
        T: Temperature (K)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Equilibrium constant K (dimensionless)
        """
        return math.exp(-delta_G / (R * T))
    
    @staticmethod
    def van_t_hoff_equation(K1, delta_H, T1, T2, R=8.314462618):
        """
        Calculate equilibrium constant at T2 given K1 at T1.
        ln(K2/K1) = (ΔH°/R) * (1/T1 - 1/T2)
        
        Parameters:
        K1: Equilibrium constant at T1
        delta_H: Enthalpy change (J/mol)
        T1: Temperature 1 (K)
        T2: Temperature 2 (K)
        R: Universal gas constant (J/(mol·K))
        
        Returns:
        Equilibrium constant at T2
        """
        return K1 * math.exp((delta_H / R) * (1/T1 - 1/T2))
    
    @staticmethod
    def conversion_at_equilibrium(K, nu_A=1, nu_B=1, nu_C=1, nu_D=1, 
                                  C_A0=1, C_B0=0, C_C0=0, C_D0=0, 
                                  epsilon=0, pressure_ratio=1):
        """
        Calculate conversion at equilibrium for gas-phase reaction.
        For reaction: aA + bB <-> cC + dD
        K = (C_C^c * C_D^d) / (C_A^a * C_B^b)
        
        This is a simplified version. For exact solution, solve numerically.
        
        Parameters:
        K: Equilibrium constant
        nu_A, nu_B, nu_C, nu_D: Stoichiometric coefficients
        C_A0, C_B0, C_C0, C_D0: Initial concentrations (mol/L)
        epsilon: Change in number of moles per mole of A reacted
        pressure_ratio: Pressure ratio P/P0
        
        Returns:
        Equilibrium conversion (0 to 1) - approximation
        """
        # Simplified calculation - for exact solution, use numerical methods
        # This returns an approximation
        if K < 1e-10:
            return 0
        # For simple case: A <-> B, X_eq = K/(1+K)
        if nu_A == 1 and nu_B == 1 and nu_C == 0 and nu_D == 0:
            return K / (1 + K)
        # Default approximation
        return min(0.99, K / (1 + K))


class SelectivityAndYield:
    """
    Class for selectivity and yield calculations
    """
    
    @staticmethod
    def selectivity_instantaneous(F_R_out, F_S_out, F_R_in=0, F_S_in=0):
        """
        Calculate instantaneous selectivity.
        S = (dF_R/dF_A) / (dF_S/dF_A)
        For parallel reactions, S = r_R / r_S
        
        Parameters:
        F_R_out: Molar flow rate of desired product R (mol/s)
        F_S_out: Molar flow rate of undesired product S (mol/s)
        F_R_in: Inlet molar flow rate of R (mol/s), default 0
        F_S_in: Inlet molar flow rate of S (mol/s), default 0
        
        Returns:
        Selectivity (dimensionless)
        """
        delta_F_S = F_S_out - F_S_in
        if delta_F_S == 0:
            return float('inf') if (F_R_out - F_R_in) > 0 else 0
        return (F_R_out - F_R_in) / delta_F_S
    
    @staticmethod
    def yield_reaction(F_R_out, F_A0, nu_R=1, nu_A=1, F_R_in=0):
        """
        Calculate yield of product R.
        Y_R = (F_R_out - F_R_in) / (F_A0 * (nu_R/nu_A))
        
        Parameters:
        F_R_out: Outlet molar flow rate of product R (mol/s)
        F_A0: Inlet molar flow rate of reactant A (mol/s)
        nu_R: Stoichiometric coefficient of R
        nu_A: Stoichiometric coefficient of A
        F_R_in: Inlet molar flow rate of R (mol/s), default 0
        
        Returns:
        Yield (dimensionless, 0 to 1)
        """
        if F_A0 == 0:
            return 0
        return ((F_R_out - F_R_in) / F_A0) * (nu_A / nu_R)
    
    @staticmethod
    def yield_from_conversion_and_selectivity(X, S):
        """
        Calculate yield from conversion and selectivity.
        Y = X * S
        
        Parameters:
        X: Conversion
        S: Selectivity
        
        Returns:
        Yield (dimensionless)
        """
        return X * S


class CatalystEffectiveness:
    """
    Class for catalyst effectiveness factor calculations
    """
    
    @staticmethod
    def thiele_modulus_sphere(k, D_eff, R, C_A_s, n=1):
        """
        Calculate Thiele modulus for spherical catalyst particle.
        phi = R * sqrt((k*C_A_s^(n-1)) / D_eff)
        
        Parameters:
        k: Rate constant (units depend on reaction order)
        D_eff: Effective diffusivity (m²/s)
        R: Particle radius (m)
        C_A_s: Surface concentration (mol/m³)
        n: Reaction order
        
        Returns:
        Thiele modulus (dimensionless)
        """
        if n == 1:
            return R * math.sqrt(k / D_eff)
        else:
            return R * math.sqrt((k * (C_A_s ** (n - 1))) / D_eff)
    
    @staticmethod
    def thiele_modulus_slab(k, D_eff, L, C_A_s, n=1):
        """
        Calculate Thiele modulus for slab catalyst particle.
        phi = L * sqrt((k*C_A_s^(n-1)) / D_eff)
        
        Parameters:
        k: Rate constant (units depend on reaction order)
        D_eff: Effective diffusivity (m²/s)
        L: Half-thickness of slab (m)
        C_A_s: Surface concentration (mol/m³)
        n: Reaction order
        
        Returns:
        Thiele modulus (dimensionless)
        """
        if n == 1:
            return L * math.sqrt(k / D_eff)
        else:
            return L * math.sqrt((k * (C_A_s ** (n - 1))) / D_eff)
    
    @staticmethod
    def effectiveness_factor_sphere(phi):
        """
        Calculate effectiveness factor for spherical catalyst particle.
        eta = (3/phi) * ((1/tanh(phi)) - (1/phi))
        
        Parameters:
        phi: Thiele modulus
        
        Returns:
        Effectiveness factor (0 to 1)
        """
        if phi == 0:
            return 1.0
        if phi < 0.01:
            # For small phi, eta ≈ 1
            return 1.0
        return (3 / phi) * ((1 / math.tanh(phi)) - (1 / phi))
    
    @staticmethod
    def effectiveness_factor_slab(phi):
        """
        Calculate effectiveness factor for slab catalyst particle.
        eta = tanh(phi) / phi
        
        Parameters:
        phi: Thiele modulus
        
        Returns:
        Effectiveness factor (0 to 1)
        """
        if phi == 0:
            return 1.0
        if phi < 0.01:
            return 1.0
        return math.tanh(phi) / phi
    
    @staticmethod
    def weisz_modulus(phi, eta):
        """
        Calculate Weisz modulus.
        Psi = phi² * eta
        
        Parameters:
        phi: Thiele modulus
        eta: Effectiveness factor
        
        Returns:
        Weisz modulus (dimensionless)
        """
        return phi ** 2 * eta


class MultipleReactions:
    """
    Class for multiple reaction systems
    """
    
    @staticmethod
    def parallel_reactions_selectivity(k1, k2, alpha, beta, C_A):
        """
        Calculate selectivity for parallel reactions.
        A -> R (desired), rate = k1 * C_A^alpha
        A -> S (undesired), rate = k2 * C_A^beta
        S = k1 * C_A^(alpha-beta) / k2
        
        Parameters:
        k1: Rate constant for desired reaction
        k2: Rate constant for undesired reaction
        alpha: Order of desired reaction
        beta: Order of undesired reaction
        C_A: Concentration of A
        
        Returns:
        Selectivity S = r_R / r_S
        """
        if k2 == 0 or C_A <= 0:
            return float('inf')
        return (k1 / k2) * (C_A ** (alpha - beta))
    
    @staticmethod
    def series_reactions_concentration_C(C_A0, k1, k2, t):
        """
        Calculate concentration of intermediate C for series reactions.
        A -> B -> C
        C_B = (k1 * C_A0 / (k2 - k1)) * (exp(-k1*t) - exp(-k2*t))
        
        Parameters:
        C_A0: Initial concentration of A (mol/L)
        k1: Rate constant for A -> B (1/s)
        k2: Rate constant for B -> C (1/s)
        t: Time (s)
        
        Returns:
        Concentration of B (mol/L)
        """
        if k1 == k2:
            return k1 * C_A0 * t * math.exp(-k1 * t)
        return (k1 * C_A0 / (k2 - k1)) * (math.exp(-k1 * t) - math.exp(-k2 * t))
    
    @staticmethod
    def optimum_time_max_intermediate(C_A0, k1, k2):
        """
        Calculate time to maximize intermediate concentration in series reactions.
        t_opt = ln(k2/k1) / (k2 - k1)
        
        Parameters:
        C_A0: Initial concentration of A (mol/L)
        k1: Rate constant for A -> B (1/s)
        k2: Rate constant for B -> C (1/s)
        
        Returns:
        Optimal time (s)
        """
        if k1 == k2 or k2 <= 0 or k1 <= 0:
            return 0
        return math.log(k2 / k1) / (k2 - k1)


class NonIdealReactors:
    """
    Class for non-ideal reactor behavior (RTD, dispersion)
    """
    
    @staticmethod
    def peclet_number(u, L, D_ax):
        """
        Calculate Peclet number for axial dispersion.
        Pe = u*L / D_ax
        
        Parameters:
        u: Superficial velocity (m/s)
        L: Reactor length (m)
        D_ax: Axial dispersion coefficient (m²/s)
        
        Returns:
        Peclet number (dimensionless)
        """
        if D_ax == 0:
            return float('inf')
        return (u * L) / D_ax
    
    @staticmethod
    def dispersion_model_conversion(Pe, Da, X_ideal):
        """
        Approximate conversion with dispersion model.
        Simplified calculation - for exact solution, solve numerically.
        
        Parameters:
        Pe: Peclet number
        Da: Damköhler number
        X_ideal: Conversion for ideal PFR
        
        Returns:
        Conversion with dispersion (approximation)
        """
        if Pe > 100:
            return X_ideal  # Approaches PFR
        # Approximation: reduced conversion due to dispersion
        return X_ideal * (1 - 1/Pe)
    
    @staticmethod
    def damkohler_number(k, tau, C_A0, n=1):
        """
        Calculate Damköhler number.
        Da = k * tau * C_A0^(n-1)
        
        Parameters:
        k: Rate constant
        tau: Space time (s)
        C_A0: Initial concentration (mol/L)
        n: Reaction order
        
        Returns:
        Damköhler number (dimensionless)
        """
        if n == 1:
            return k * tau
        else:
            return k * tau * (C_A0 ** (n - 1))
    
    @staticmethod
    def residence_time_distribution_cstr(t, tau):
        """
        RTD function for CSTR.
        E(t) = (1/tau) * exp(-t/tau)
        
        Parameters:
        t: Time (s)
        tau: Mean residence time (s)
        
        Returns:
        E(t) (1/s)
        """
        if tau <= 0:
            return 0
        return (1 / tau) * math.exp(-t / tau)
    
    @staticmethod
    def residence_time_distribution_pfr(t, tau):
        """
        RTD function for ideal PFR (Dirac delta function).
        This is an approximation: E(t) = delta(t - tau)
        Returns 0 for t != tau, infinity for t = tau.
        
        Parameters:
        t: Time (s)
        tau: Mean residence time (s)
        
        Returns:
        E(t) approximation (1/s) - returns large value if t ≈ tau
        """
        if abs(t - tau) < 1e-6:
            return 1e6  # Approximation for delta function
        return 0


class HeatEffects:
    """
    Class for heat effects in chemical reactors
    """
    
    @staticmethod
    def adiabatic_temperature_rise(X, delta_H_rxn, F_A0, C_p_total):
        """
        Calculate adiabatic temperature rise.
        ΔT_ad = (X * F_A0 * (-ΔH_rxn)) / (Σ F_i0 * C_pi)
        
        Parameters:
        X: Conversion
        delta_H_rxn: Heat of reaction (J/mol) - negative for exothermic
        F_A0: Inlet molar flow rate of A (mol/s)
        C_p_total: Total heat capacity flow rate (J/(s·K))
        
        Returns:
        Temperature rise (K)
        """
        if C_p_total == 0:
            return float('inf')
        return (X * F_A0 * (-delta_H_rxn)) / C_p_total
    
    @staticmethod
    def energy_balance_cstr(T, T0, X, delta_H_rxn, F_A0, C_p_total, Q):
        """
        Energy balance for CSTR.
        Σ F_i0 * C_pi * (T - T0) = X * F_A0 * (-ΔH_rxn) + Q
        
        Parameters:
        T: Outlet temperature (K)
        T0: Inlet temperature (K)
        X: Conversion
        delta_H_rxn: Heat of reaction (J/mol)
        F_A0: Inlet molar flow rate of A (mol/s)
        C_p_total: Total heat capacity flow rate (J/(s·K))
        Q: Heat transfer rate (W, positive for heating)
        
        Returns:
        Energy balance residual (should be 0 at steady state)
        """
        return C_p_total * (T - T0) - X * F_A0 * (-delta_H_rxn) - Q
    
    @staticmethod
    def heat_removal_rate(U, A, T_reactor, T_coolant):
        """
        Calculate heat removal rate.
        Q = U * A * (T_reactor - T_coolant)
        
        Parameters:
        U: Overall heat transfer coefficient (W/(m²·K))
        A: Heat transfer area (m²)
        T_reactor: Reactor temperature (K)
        T_coolant: Coolant temperature (K)
        
        Returns:
        Heat transfer rate (W)
        """
        return U * A * (T_reactor - T_coolant)


class ReactorSizing:
    """
    Class for reactor sizing and scale-up
    """
    
    @staticmethod
    def number_of_cstrs_in_series(X_total, k, tau_single):
        """
        Calculate number of CSTRs in series to achieve conversion.
        For first-order reaction: X = 1 - 1/(1 + k*tau)^N
        
        Parameters:
        X_total: Desired total conversion
        k: Rate constant (1/s)
        tau_single: Space time per reactor (s)
        
        Returns:
        Number of CSTRs required (approximate)
        """
        if k * tau_single <= 0:
            return float('inf')
        if X_total >= 1:
            return float('inf')
        # Solve: X = 1 - 1/(1 + k*tau)^N
        # N = ln(1/(1-X)) / ln(1 + k*tau)
        return math.ceil(math.log(1 / (1 - X_total)) / math.log(1 + k * tau_single))
    
    @staticmethod
    def reactor_volume_from_conversion_pfr(F_A0, X, k, C_A0, n=1, v0=1):
        """
        General PFR volume calculation.
        V = F_A0 * ∫[0 to X] (dX / (-r_A))
        
        This is a simplified version for specific orders.
        For complex rate laws, use numerical integration.
        
        Parameters:
        F_A0: Inlet molar flow rate (mol/s)
        X: Conversion
        k: Rate constant
        C_A0: Inlet concentration (mol/L)
        n: Reaction order
        v0: Volumetric flow rate (L/s)
        
        Returns:
        Reactor volume (L)
        """
        if n == 1:
            return ReactorDesign.pfr_volume_first_order(F_A0, X, k, v0)
        elif n == 2:
            return ReactorDesign.pfr_volume_second_order(F_A0, X, k, C_A0, v0)
        elif n == 0:
            return ReactorDesign.pfr_volume_zero_order(F_A0, X, k)
        else:
            # For other orders, approximate or use numerical integration
            # This is a rough approximation
            return (F_A0 * X) / (k * (C_A0 ** n) * ((1 - X) ** n))
