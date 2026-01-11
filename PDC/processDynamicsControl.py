"""
Process Dynamics and Control Formulas for Chemical Engineering
Comprehensive collection of classes for process control calculations
"""

import math
import numpy as np
from typing import Tuple, List


class FirstOrderSystems:
    """
    Class for first-order system calculations
    """
    
    @staticmethod
    def time_constant_tau(R, C=None, tau=None):
        """
        Calculate time constant for first-order system.
        
        Parameters:
        R: Resistance (varies with system)
        C: Capacitance (varies with system)
        tau: Time constant (s) - if provided, returns directly
        
        Returns:
        Time constant (s)
        """
        if tau is not None:
            return tau
        if C is None:
            raise ValueError("Either tau or both R and C must be provided")
        return R * C
    
    @staticmethod
    def step_response_first_order(K, tau, t, y0=0):
        """
        Calculate step response of first-order system.
        y(t) = K*(1 - exp(-t/tau)) + y0
        
        Parameters:
        K: Process gain
        tau: Time constant (s)
        t: Time (s)
        y0: Initial condition
        
        Returns:
        Response value at time t
        """
        return K * (1 - math.exp(-t / tau)) + y0
    
    @staticmethod
    def ramp_response_first_order(K, tau, A, t, y0=0):
        """
        Calculate ramp response of first-order system.
        
        Parameters:
        K: Process gain
        tau: Time constant (s)
        A: Ramp slope
        t: Time (s)
        y0: Initial condition
        
        Returns:
        Response value at time t
        """
        return K * A * (t - tau * (1 - math.exp(-t / tau))) + y0
    
    @staticmethod
    def settling_time_first_order(tau, percentage=2):
        """
        Calculate settling time for first-order system.
        
        Parameters:
        tau: Time constant (s)
        percentage: Settling percentage (default 2% for ±2%)
        
        Returns:
        Settling time (s)
        """
        # For ±2%: exp(-t/tau) = 0.02, so t = -tau*ln(0.02) ≈ 3.91*tau
        # For ±5%: t ≈ 2.99*tau
        if percentage == 2:
            return 3.91 * tau
        elif percentage == 5:
            return 2.99 * tau
        else:
            epsilon = percentage / 100
            return -tau * math.log(epsilon)
    
    @staticmethod
    def time_to_reach_percentage(K, tau, percentage, y0=0):
        """
        Calculate time to reach a certain percentage of final value.
        
        Parameters:
        K: Process gain
        tau: Time constant (s)
        percentage: Percentage of final value (0-100)
        y0: Initial condition
        
        Returns:
        Time to reach percentage (s)
        """
        p = percentage / 100
        final_value = K + y0
        target = y0 + p * K
        # y(t) = K*(1 - exp(-t/tau)) + y0
        # target - y0 = K*(1 - exp(-t/tau))
        # (target - y0)/K = 1 - exp(-t/tau)
        # exp(-t/tau) = 1 - (target - y0)/K = 1 - p
        # t = -tau*ln(1 - p)
        if p >= 1:
            return float('inf')
        return -tau * math.log(1 - p)
    
    @staticmethod
    def transfer_function_first_order(K, tau):
        """
        Get transfer function parameters for first-order system.
        G(s) = K/(tau*s + 1)
        
        Parameters:
        K: Process gain
        tau: Time constant (s)
        
        Returns:
        Tuple (gain, time_constant)
        """
        return (K, tau)


class SecondOrderSystems:
    """
    Class for second-order system calculations
    """
    
    @staticmethod
    def natural_frequency_omega_n(tau, zeta=None):
        """
        Calculate natural frequency.
        
        Parameters:
        tau: Time constant (s) or period parameter
        zeta: Damping ratio (optional)
        
        Returns:
        Natural frequency (rad/s)
        """
        if zeta is None:
            # Assuming tau is the period: tau = 2*pi/omega_n
            return 2 * math.pi / tau
        return 1 / tau  # For standard form: tau = 1/omega_n
    
    @staticmethod
    def damping_ratio_zeta(tau1, tau2):
        """
        Calculate damping ratio from two time constants.
        
        Parameters:
        tau1: First time constant (s)
        tau2: Second time constant (s)
        
        Returns:
        Damping ratio (dimensionless)
        """
        # For second-order: zeta = (tau1 + tau2)/(2*sqrt(tau1*tau2))
        return (tau1 + tau2) / (2 * math.sqrt(tau1 * tau2))
    
    @staticmethod
    def step_response_second_order(K, omega_n, zeta, t, y0=0):
        """
        Calculate step response of second-order system.
        
        Parameters:
        K: Process gain
        omega_n: Natural frequency (rad/s)
        zeta: Damping ratio
        t: Time (s)
        y0: Initial condition
        
        Returns:
        Response value at time t
        """
        if zeta < 1:  # Underdamped
            omega_d = omega_n * math.sqrt(1 - zeta ** 2)
            response = K * (1 - (math.exp(-zeta * omega_n * t) / 
                                math.sqrt(1 - zeta ** 2)) * 
                           math.sin(omega_d * t + math.atan2(
                               math.sqrt(1 - zeta ** 2), zeta))) + y0
        elif zeta == 1:  # Critically damped
            response = K * (1 - (1 + omega_n * t) * math.exp(-omega_n * t)) + y0
        else:  # Overdamped
            s1 = -omega_n * (zeta + math.sqrt(zeta ** 2 - 1))
            s2 = -omega_n * (zeta - math.sqrt(zeta ** 2 - 1))
            A1 = omega_n / (2 * math.sqrt(zeta ** 2 - 1))
            A2 = -A1
            response = K * (1 + A1 * math.exp(s1 * t) + A2 * math.exp(s2 * t)) + y0
        
        return response
    
    @staticmethod
    def peak_time_underdamped(omega_n, zeta):
        """
        Calculate peak time for underdamped system.
        
        Parameters:
        omega_n: Natural frequency (rad/s)
        zeta: Damping ratio (must be < 1)
        
        Returns:
        Peak time (s)
        """
        if zeta >= 1:
            raise ValueError("Peak time only defined for underdamped systems (zeta < 1)")
        omega_d = omega_n * math.sqrt(1 - zeta ** 2)
        return math.pi / omega_d
    
    @staticmethod
    def overshoot_percentage(zeta):
        """
        Calculate percentage overshoot for underdamped system.
        
        Parameters:
        zeta: Damping ratio (must be < 1)
        
        Returns:
        Percentage overshoot (%)
        """
        if zeta >= 1:
            return 0.0
        return 100 * math.exp(-zeta * math.pi / math.sqrt(1 - zeta ** 2))
    
    @staticmethod
    def settling_time_second_order(omega_n, zeta, percentage=2):
        """
        Calculate settling time for second-order system (±2% or ±5%).
        
        Parameters:
        omega_n: Natural frequency (rad/s)
        zeta: Damping ratio
        percentage: Settling percentage (2 or 5)
        
        Returns:
        Settling time (s)
        """
        if percentage == 2:
            tolerance = 0.02
        elif percentage == 5:
            tolerance = 0.05
        else:
            tolerance = percentage / 100
        
        # Approximate formula: ts = 4/(zeta*omega_n) for 2% criterion
        if zeta > 0:
            return 4 / (zeta * omega_n)
        else:
            return float('inf')
    
    @staticmethod
    def rise_time_underdamped(omega_n, zeta, definition='10-90'):
        """
        Calculate rise time for underdamped system.
        
        Parameters:
        omega_n: Natural frequency (rad/s)
        zeta: Damping ratio (must be < 1)
        definition: '10-90' or '0-100' (default '10-90')
        
        Returns:
        Rise time (s)
        """
        if zeta >= 1:
            raise ValueError("Rise time calculation for underdamped systems only")
        
        omega_d = omega_n * math.sqrt(1 - zeta ** 2)
        
        if definition == '10-90':
            # Approximation for 10-90% rise time
            phi = math.atan2(math.sqrt(1 - zeta ** 2), zeta)
            t10 = (math.pi - phi) / omega_d
            t90 = (math.pi + phi) / omega_d
            return t90 - t10
        else:  # 0-100%
            phi = math.atan2(math.sqrt(1 - zeta ** 2), zeta)
            return (math.pi - phi) / omega_d


class PIDController:
    """
    Class for PID controller calculations
    """
    
    @staticmethod
    def pid_output(Kc, e, e_prev, e_sum, dt, tau_i=None, tau_d=None):
        """
        Calculate PID controller output (position form).
        u(t) = Kc*[e(t) + (1/tau_i)*∫e dt + tau_d*de/dt]
        
        Parameters:
        Kc: Controller gain
        e: Current error
        e_prev: Previous error
        e_sum: Integral of error (sum)
        dt: Time step (s)
        tau_i: Integral time constant (s)
        tau_d: Derivative time constant (s)
        
        Returns:
        Controller output and updated error integral
        """
        # Proportional term
        P = Kc * e
        
        # Integral term
        I = 0
        if tau_i is not None and tau_i > 0:
            e_sum += e * dt
            I = Kc * e_sum / tau_i
        
        # Derivative term
        D = 0
        if tau_d is not None and tau_d > 0:
            de_dt = (e - e_prev) / dt if dt > 0 else 0
            D = Kc * tau_d * de_dt
        
        u = P + I + D
        
        return u, e_sum
    
    @staticmethod
    def pid_velocity_form(Kc, e, e_prev, e_prev2, dt, tau_i=None, tau_d=None, u_prev=0):
        """
        Calculate PID controller output (velocity/incremental form).
        
        Parameters:
        Kc: Controller gain
        e: Current error
        e_prev: Previous error
        e_prev2: Error two steps ago
        dt: Time step (s)
        tau_i: Integral time constant (s)
        tau_d: Derivative time constant (s)
        u_prev: Previous controller output
        
        Returns:
        Controller output (incremental change added to previous)
        """
        # Proportional term
        P = Kc * (e - e_prev)
        
        # Integral term
        I = 0
        if tau_i is not None and tau_i > 0:
            I = Kc * e * dt / tau_i
        
        # Derivative term
        D = 0
        if tau_d is not None and tau_d > 0:
            if dt > 0:
                D = Kc * tau_d * (e - 2*e_prev + e_prev2) / dt
        
        delta_u = P + I + D
        return u_prev + delta_u
    
    @staticmethod
    def parallel_form_to_isa_form(Kp, Ki, Kd):
        """
        Convert parallel PID form to ISA (interacting) form.
        
        Parameters:
        Kp: Proportional gain
        Ki: Integral gain (1/s)
        Kd: Derivative gain (s)
        
        Returns:
        Tuple (Kc, tau_i, tau_d) in ISA form
        """
        Kc = Kp
        tau_i = Kp / Ki if Ki > 0 else float('inf')
        tau_d = Kd / Kp if Kp > 0 else 0
        return (Kc, tau_i, tau_d)
    
    @staticmethod
    def isa_form_to_parallel_form(Kc, tau_i, tau_d):
        """
        Convert ISA PID form to parallel form.
        
        Parameters:
        Kc: Controller gain
        tau_i: Integral time constant (s)
        tau_d: Derivative time constant (s)
        
        Returns:
        Tuple (Kp, Ki, Kd) in parallel form
        """
        Kp = Kc
        Ki = Kc / tau_i if tau_i > 0 else float('inf')
        Kd = Kc * tau_d
        return (Kp, Ki, Kd)


class ZieglerNicholsTuning:
    """
    Class for Ziegler-Nichols tuning methods
    """
    
    @staticmethod
    def ultimate_gain_method(Ku, Pu):
        """
        Calculate PID parameters using ultimate gain method.
        
        Parameters:
        Ku: Ultimate gain
        Pu: Ultimate period (s)
        
        Returns:
        Dictionary with P, PI, and PID controller parameters
        (Kc, tau_i, tau_d)
        """
        # P controller
        Kc_p = 0.5 * Ku
        tau_i_p = float('inf')
        tau_d_p = 0
        
        # PI controller
        Kc_pi = 0.45 * Ku
        tau_i_pi = Pu / 1.2
        tau_d_pi = 0
        
        # PID controller
        Kc_pid = 0.6 * Ku
        tau_i_pid = Pu / 2
        tau_d_pid = Pu / 8
        
        return {
            'P': (Kc_p, tau_i_p, tau_d_p),
            'PI': (Kc_pi, tau_i_pi, tau_d_pi),
            'PID': (Kc_pid, tau_i_pid, tau_d_pid)
        }
    
    @staticmethod
    def reaction_curve_method(K, tau, theta):
        """
        Calculate PID parameters using reaction curve (step response) method.
        Process modeled as: G(s) = K*exp(-theta*s)/(tau*s + 1)
        
        Parameters:
        K: Process gain
        tau: Time constant (s)
        theta: Dead time (s)
        
        Returns:
        Dictionary with P, PI, and PID controller parameters
        (Kc, tau_i, tau_d)
        """
        # P controller
        Kc_p = tau / (K * theta)
        tau_i_p = float('inf')
        tau_d_p = 0
        
        # PI controller
        Kc_pi = 0.9 * tau / (K * theta)
        tau_i_pi = 3.33 * theta
        tau_d_pi = 0
        
        # PID controller
        Kc_pid = 1.2 * tau / (K * theta)
        tau_i_pid = 2 * theta
        tau_d_pid = 0.5 * theta
        
        return {
            'P': (Kc_p, tau_i_p, tau_d_p),
            'PI': (Kc_pi, tau_i_pi, tau_d_pi),
            'PID': (Kc_pid, tau_i_pid, tau_d_pid)
        }
    
    @staticmethod
    def tyreus_luyben_tuning(Ku, Pu):
        """
        Calculate PID parameters using Tyreus-Luyben tuning (more conservative).
        
        Parameters:
        Ku: Ultimate gain
        Pu: Ultimate period (s)
        
        Returns:
        Tuple (Kc, tau_i, tau_d) for PID controller
        """
        Kc = Ku / 2.2
        tau_i = 2.2 * Pu
        tau_d = Pu / 6.3
        return (Kc, tau_i, tau_d)


class FrequencyResponse:
    """
    Class for frequency response analysis
    """
    
    @staticmethod
    def magnitude_first_order(K, tau, omega):
        """
        Calculate magnitude of first-order transfer function at frequency.
        |G(jω)| = K/sqrt((tau*ω)^2 + 1)
        
        Parameters:
        K: Process gain
        tau: Time constant (s)
        omega: Frequency (rad/s)
        
        Returns:
        Magnitude (absolute value)
        """
        return K / math.sqrt((tau * omega) ** 2 + 1)
    
    @staticmethod
    def phase_first_order(tau, omega):
        """
        Calculate phase angle of first-order transfer function.
        ∠G(jω) = -arctan(tau*ω)
        
        Parameters:
        tau: Time constant (s)
        omega: Frequency (rad/s)
        
        Returns:
        Phase angle (radians)
        """
        return -math.atan(tau * omega)
    
    @staticmethod
    def magnitude_second_order(K, omega_n, zeta, omega):
        """
        Calculate magnitude of second-order transfer function.
        |G(jω)| = K/sqrt((1 - (ω/ω_n)^2)^2 + (2*ζ*ω/ω_n)^2)
        
        Parameters:
        K: Process gain
        omega_n: Natural frequency (rad/s)
        zeta: Damping ratio
        omega: Frequency (rad/s)
        
        Returns:
        Magnitude (absolute value)
        """
        r = omega / omega_n
        return K / math.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)
    
    @staticmethod
    def phase_second_order(omega_n, zeta, omega):
        """
        Calculate phase angle of second-order transfer function.
        
        Parameters:
        omega_n: Natural frequency (rad/s)
        zeta: Damping ratio
        omega: Frequency (rad/s)
        
        Returns:
        Phase angle (radians)
        """
        r = omega / omega_n
        if r < 1:
            return -math.atan2(2 * zeta * r, 1 - r ** 2)
        else:
            return -math.pi - math.atan2(2 * zeta * r, r ** 2 - 1)
    
    @staticmethod
    def dead_time_phase(theta, omega):
        """
        Calculate phase contribution from dead time.
        ∠exp(-j*θ*ω) = -θ*ω
        
        Parameters:
        theta: Dead time (s)
        omega: Frequency (rad/s)
        
        Returns:
        Phase angle (radians)
        """
        return -theta * omega
    
    @staticmethod
    def gain_margin(omega_cg, Kc):
        """
        Calculate gain margin.
        
        Parameters:
        omega_cg: Phase crossover frequency (rad/s)
        Kc: Controller gain at phase crossover
        
        Returns:
        Gain margin (dB if Kc in linear scale)
        """
        # GM = 1/Kc (linear) or 20*log10(1/Kc) (dB)
        return 20 * math.log10(1 / Kc) if Kc > 0 else float('inf')
    
    @staticmethod
    def phase_margin(omega_cp, phase_at_cp):
        """
        Calculate phase margin.
        
        Parameters:
        omega_cp: Gain crossover frequency (rad/s)
        phase_at_cp: Phase at gain crossover (radians)
        
        Returns:
        Phase margin (degrees)
        """
        return 180 + math.degrees(phase_at_cp)


class StabilityAnalysis:
    """
    Class for stability analysis methods
    """
    
    @staticmethod
    def routh_hurwitz_coefficients(a):
        """
        Calculate Routh-Hurwitz table coefficients.
        Returns first two rows.
        
        Parameters:
        a: Coefficients of characteristic polynomial [a_n, a_n-1, ..., a_0]
           where a_n*s^n + a_n-1*s^(n-1) + ... + a_0 = 0
        
        Returns:
        First two rows of Routh table
        """
        n = len(a) - 1  # Order of polynomial
        if n < 1:
            raise ValueError("Polynomial must be at least order 1")
        
        # First row: a_n, a_n-2, a_n-4, ...
        row1 = []
        # Second row: a_n-1, a_n-3, a_n-5, ...
        row2 = []
        
        for i in range(n + 1):
            if i % 2 == 0:
                row1.append(a[n - i])
            else:
                row2.append(a[n - i])
        
        # Pad shorter row with zeros
        max_len = max(len(row1), len(row2))
        row1.extend([0] * (max_len - len(row1)))
        row2.extend([0] * (max_len - len(row2)))
        
        return [row1, row2]
    
    @staticmethod
    def stability_criterion_routh(first_row, second_row):
        """
        Check stability using Routh-Hurwitz criterion (sign changes).
        
        Parameters:
        first_row: First row of Routh table
        second_row: Second row of Routh table
        
        Returns:
        True if stable (all elements of first column have same sign)
        """
        # Check first column for sign changes
        signs = []
        for row in [first_row, second_row]:
            if len(row) > 0 and row[0] != 0:
                signs.append(1 if row[0] > 0 else -1)
        
        # All signs must be the same for stability
        return len(set(signs)) == 1 if signs else False
    
    @staticmethod
    def nyquist_criterion(open_loop_poles_rhp):
        """
        Nyquist stability criterion (simplified check).
        System is stable if N = P - Z = 0, where:
        N = number of encirclements of -1
        P = number of open-loop poles in RHP
        Z = number of closed-loop poles in RHP (should be 0 for stability)
        
        Parameters:
        open_loop_poles_rhp: Number of open-loop poles in right half plane
        
        Returns:
        True if closed-loop system can be stable (requires N = P)
        """
        # Simplified: if P = 0, system is stable if no encirclements
        # Full analysis requires Nyquist plot construction
        return open_loop_poles_rhp == 0


class ProcessModels:
    """
    Class for common process model calculations
    """
    
    @staticmethod
    def cstr_steady_state(C_A0, V, F, k):
        """
        Calculate steady-state concentration in CSTR.
        C_A = C_A0 / (1 + k*V/F)
        
        Parameters:
        C_A0: Inlet concentration (mol/L)
        V: Reactor volume (L)
        F: Volumetric flow rate (L/s)
        k: Reaction rate constant (1/s)
        
        Returns:
        Outlet concentration (mol/L)
        """
        tau = V / F  # Residence time
        return C_A0 / (1 + k * tau)
    
    @staticmethod
    def cstr_time_constant(V, F):
        """
        Calculate time constant for CSTR.
        
        Parameters:
        V: Reactor volume (L)
        F: Volumetric flow rate (L/s)
        
        Returns:
        Time constant tau = V/F (s)
        """
        return V / F
    
    @staticmethod
    def heat_exchanger_gain(Q, F, rho, Cp, T_in, T_out, T_sp):
        """
        Estimate process gain for heat exchanger control.
        
        Parameters:
        Q: Heat duty (W)
        F: Flow rate (kg/s)
        rho: Density (kg/m³) - optional for some models
        Cp: Heat capacity (J/(kg·K))
        T_in: Inlet temperature (K)
        T_out: Outlet temperature (K)
        T_sp: Set point temperature (K)
        
        Returns:
        Process gain estimate
        """
        # Simplified: K ≈ ΔT_out / ΔQ
        # For constant flow: dT_out/dQ ≈ 1/(F*Cp)
        return 1 / (F * Cp)
    
    @staticmethod
    def mixing_tank_gain(V, F):
        """
        Calculate process gain for mixing tank (concentration control).
        
        Parameters:
        V: Tank volume (L)
        F: Flow rate (L/s)
        
        Returns:
        Process gain (dimensionless, approximately 1 for perfect mixing)
        """
        return 1.0  # Perfect mixing assumption
    
    @staticmethod
    def dead_time_approximation(L, v):
        """
        Calculate dead time from transport delay.
        
        Parameters:
        L: Length/distance (m)
        v: Velocity (m/s)
        
        Returns:
        Dead time theta = L/v (s)
        """
        if v <= 0:
            return float('inf')
        return L / v


class ClosedLoopAnalysis:
    """
    Class for closed-loop system analysis
    """
    
    @staticmethod
    def closed_loop_transfer_function(Kc, Kp, tau_p, tau_c=None, tau_d=None):
        """
        Calculate closed-loop transfer function for first-order process with P/PI/PID.
        Simplified: assumes unity feedback, first-order process
        
        Parameters:
        Kc: Controller gain
        Kp: Process gain
        tau_p: Process time constant (s)
        tau_c: Controller integral time (s), None for P control
        tau_d: Controller derivative time (s), None for no derivative action
        
        Returns:
        Tuple (K_cl, tau_cl) for closed-loop first-order approximation
        """
        # For P control: G_cl = Kc*Kp / (1 + Kc*Kp + tau_p*s)
        K_cl = (Kc * Kp) / (1 + Kc * Kp)
        tau_cl = tau_p / (1 + Kc * Kp)
        return (K_cl, tau_cl)
    
    @staticmethod
    def offset_proportional_control(Kc, Kp, setpoint, disturbance):
        """
        Calculate steady-state offset for proportional control.
        
        Parameters:
        Kc: Controller gain
        Kp: Process gain
        setpoint: Set point value
        disturbance: Disturbance magnitude
        
        Returns:
        Steady-state offset
        """
        # Offset = disturbance / (1 + Kc*Kp)
        return disturbance / (1 + Kc * Kp)
    
    @staticmethod
    def offset_integral_action(setpoint, disturbance):
        """
        Offset with integral action (should be zero at steady state).
        
        Parameters:
        setpoint: Set point value
        disturbance: Disturbance magnitude
        
        Returns:
        Steady-state offset (should be 0 for PI/PID with integral action)
        """
        return 0.0  # Integral action eliminates offset
    
    @staticmethod
    def sensitivity_function_complementary(Kc, Gp):
        """
        Calculate sensitivity and complementary sensitivity (simplified).
        S = 1/(1 + Kc*Gp)
        T = Kc*Gp/(1 + Kc*Gp)
        
        Parameters:
        Kc: Controller gain
        Gp: Process transfer function magnitude at frequency of interest
        
        Returns:
        Tuple (S, T) - sensitivity and complementary sensitivity
        """
        denominator = 1 + Kc * Gp
        S = 1 / denominator
        T = (Kc * Gp) / denominator
        return (S, T)


class DisturbanceRejection:
    """
    Class for disturbance rejection analysis
    """
    
    @staticmethod
    def load_disturbance_response(Kc, Kp, Kd, tau_p, disturbance_magnitude):
        """
        Calculate steady-state response to load disturbance.
        
        Parameters:
        Kc: Controller gain
        Kp: Process gain
        Kd: Disturbance gain
        tau_p: Process time constant (s)
        disturbance_magnitude: Magnitude of step disturbance
        
        Returns:
        Steady-state output change
        """
        # For P control: y_ss = Kd*d/(1 + Kc*Kp)
        return (Kd * disturbance_magnitude) / (1 + Kc * Kp)
    
    @staticmethod
    def setpoint_tracking_error(Kc, Kp, setpoint_change):
        """
        Calculate setpoint tracking error.
        
        Parameters:
        Kc: Controller gain
        Kp: Process gain
        setpoint_change: Change in setpoint
        
        Returns:
        Steady-state error
        """
        # Error = setpoint / (1 + Kc*Kp)
        return setpoint_change / (1 + Kc * Kp)


# Convenience functions for backward compatibility
def first_order_step_response(K, tau, t, y0=0):
    """Convenience function for first-order step response."""
    return FirstOrderSystems.step_response_first_order(K, tau, t, y0)

def second_order_overshoot(zeta):
    """Convenience function for overshoot calculation."""
    return SecondOrderSystems.overshoot_percentage(zeta)

def pid_tuning_zn(Ku, Pu):
    """Convenience function for Ziegler-Nichols tuning."""
    return ZieglerNicholsTuning.ultimate_gain_method(Ku, Pu)
