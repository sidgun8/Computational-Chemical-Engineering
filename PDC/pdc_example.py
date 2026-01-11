"""
Example usage of Process Dynamics and Control formulas
Demonstrates various calculations for control system design
"""

import numpy as np
import matplotlib.pyplot as plt
from processDynamicsControl import (
    FirstOrderSystems,
    SecondOrderSystems,
    PIDController,
    ZieglerNicholsTuning,
    FrequencyResponse,
    StabilityAnalysis,
    ProcessModels,
    ClosedLoopAnalysis
)


def example_first_order_system():
    """Example: First-order system response analysis"""
    print("=" * 60)
    print("FIRST-ORDER SYSTEM ANALYSIS")
    print("=" * 60)
    
    # System parameters
    K = 2.0  # Process gain
    tau = 5.0  # Time constant (s)
    
    # Calculate step response at various times
    times = np.linspace(0, 20, 100)
    responses = [FirstOrderSystems.step_response_first_order(K, tau, t) 
                 for t in times]
    
    # Calculate settling time (±2%)
    ts = FirstOrderSystems.settling_time_first_order(tau, percentage=2)
    print(f"\nProcess Gain (K): {K}")
    print(f"Time Constant (τ): {tau} s")
    print(f"Settling Time (±2%): {ts:.2f} s")
    
    # Calculate time to reach 63.2% (one time constant)
    t_632 = FirstOrderSystems.time_to_reach_percentage(K, tau, 63.2)
    print(f"Time to reach 63.2% of final value: {t_632:.2f} s")
    
    # Plot step response
    plt.figure(figsize=(10, 6))
    plt.plot(times, responses, 'b-', linewidth=2, label='Step Response')
    plt.axhline(y=K * 0.98, color='r', linestyle='--', label='±2% Band')
    plt.axhline(y=K * 1.02, color='r', linestyle='--')
    plt.axvline(x=ts, color='g', linestyle='--', label=f'Settling Time = {ts:.2f} s')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.title('First-Order System Step Response')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return times, responses


def example_second_order_system():
    """Example: Second-order system analysis"""
    print("\n" + "=" * 60)
    print("SECOND-ORDER SYSTEM ANALYSIS")
    print("=" * 60)
    
    # System parameters
    K = 1.0  # Process gain
    omega_n = 2.0  # Natural frequency (rad/s)
    zeta_values = [0.3, 0.7, 1.0, 1.5]  # Different damping ratios
    
    plt.figure(figsize=(12, 8))
    
    for zeta in zeta_values:
        times = np.linspace(0, 10, 500)
        responses = [SecondOrderSystems.step_response_second_order(
            K, omega_n, zeta, t) for t in times]
        
        # Calculate performance metrics for underdamped case
        if zeta < 1:
            tp = SecondOrderSystems.peak_time_underdamped(omega_n, zeta)
            Mp = SecondOrderSystems.overshoot_percentage(zeta)
            ts = SecondOrderSystems.settling_time_second_order(omega_n, zeta, percentage=2)
            tr = SecondOrderSystems.rise_time_underdamped(omega_n, zeta)
            
            print(f"\nDamping Ratio (ζ) = {zeta:.1f} (Underdamped)")
            print(f"  Peak Time (tp): {tp:.3f} s")
            print(f"  Overshoot (%): {Mp:.2f}%")
            print(f"  Settling Time (±2%): {ts:.2f} s")
            print(f"  Rise Time (10-90%): {tr:.3f} s")
        
        elif zeta == 1.0:
            ts = SecondOrderSystems.settling_time_second_order(omega_n, zeta, percentage=2)
            print(f"\nDamping Ratio (ζ) = {zeta:.1f} (Critically Damped)")
            print(f"  Settling Time (±2%): {ts:.2f} s")
        
        else:
            ts = SecondOrderSystems.settling_time_second_order(omega_n, zeta, percentage=2)
            print(f"\nDamping Ratio (ζ) = {zeta:.1f} (Overdamped)")
            print(f"  Settling Time (±2%): {ts:.2f} s")
        
        plt.plot(times, responses, linewidth=2, 
                label=f'ζ = {zeta} ({["Underdamped", "Critically Damped", "Overdamped"][int(min(zeta, 1.0))]})')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.title('Second-Order System Step Response (Different Damping Ratios)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def example_pid_controller():
    """Example: PID controller implementation"""
    print("\n" + "=" * 60)
    print("PID CONTROLLER SIMULATION")
    print("=" * 60)
    
    # Controller parameters
    Kc = 2.0  # Controller gain
    tau_i = 1.0  # Integral time (s)
    tau_d = 0.5  # Derivative time (s)
    
    # Simulation parameters
    dt = 0.01  # Time step (s)
    t_final = 10.0  # Simulation time (s)
    times = np.arange(0, t_final, dt)
    
    # Setpoint (step change at t=2)
    setpoint = np.zeros_like(times)
    setpoint[times >= 2.0] = 1.0
    
    # Simple first-order process simulation
    Kp = 1.0  # Process gain
    tau_p = 2.0  # Process time constant (s)
    
    # Initialize
    y = np.zeros_like(times)
    e_sum = 0.0
    e_prev = 0.0
    
    # PID controller output
    u = np.zeros_like(times)
    
    for i in range(1, len(times)):
        # Calculate error
        e = setpoint[i] - y[i-1]
        
        # PID controller
        u[i], e_sum = PIDController.pid_output(
            Kc, e, e_prev, e_sum, dt, tau_i, tau_d)
        
        # Process dynamics (first-order with control input)
        # dy/dt = (Kp*u - y) / tau_p
        y[i] = y[i-1] + dt * (Kp * u[i] - y[i-1]) / tau_p
        
        e_prev = e
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(times, setpoint, 'r--', linewidth=2, label='Setpoint')
    plt.plot(times, y, 'b-', linewidth=2, label='Process Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.title('PID Control Response')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(times, u, 'g-', linewidth=2, label='Controller Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Signal (u)')
    plt.title('PID Controller Output')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nController Parameters:")
    print(f"  Kc = {Kc}")
    print(f"  τi = {tau_i} s")
    print(f"  τd = {tau_d} s")
    print(f"\nProcess Parameters:")
    print(f"  Kp = {Kp}")
    print(f"  τp = {tau_p} s")


def example_ziegler_nichols_tuning():
    """Example: Ziegler-Nichols tuning methods"""
    print("\n" + "=" * 60)
    print("ZIEGLER-NICHOLS TUNING METHODS")
    print("=" * 60)
    
    # Method 1: Ultimate Gain Method
    print("\n1. Ultimate Gain Method:")
    print("-" * 40)
    Ku = 5.0  # Ultimate gain
    Pu = 2.0  # Ultimate period (s)
    
    zn_params = ZieglerNicholsTuning.ultimate_gain_method(Ku, Pu)
    
    print(f"Ultimate Gain (Ku) = {Ku}")
    print(f"Ultimate Period (Pu) = {Pu} s\n")
    
    for controller_type in ['P', 'PI', 'PID']:
        Kc, tau_i, tau_d = zn_params[controller_type]
        print(f"{controller_type} Controller:")
        print(f"  Kc = {Kc:.4f}")
        print(f"  τi = {tau_i:.4f} s" if tau_i != float('inf') else "  τi = ∞")
        print(f"  τd = {tau_d:.4f} s")
        print()
    
    # Method 2: Reaction Curve Method
    print("\n2. Reaction Curve Method:")
    print("-" * 40)
    K = 2.0  # Process gain
    tau = 3.0  # Time constant (s)
    theta = 0.5  # Dead time (s)
    
    rc_params = ZieglerNicholsTuning.reaction_curve_method(K, tau, theta)
    
    print(f"Process Gain (K) = {K}")
    print(f"Time Constant (τ) = {tau} s")
    print(f"Dead Time (θ) = {theta} s\n")
    
    for controller_type in ['P', 'PI', 'PID']:
        Kc, tau_i, tau_d = rc_params[controller_type]
        print(f"{controller_type} Controller:")
        print(f"  Kc = {Kc:.4f}")
        print(f"  τi = {tau_i:.4f} s" if tau_i != float('inf') else "  τi = ∞")
        print(f"  τd = {tau_d:.4f} s")
        print()
    
    # Method 3: Tyreus-Luyben (more conservative)
    print("\n3. Tyreus-Luyben Tuning (Conservative):")
    print("-" * 40)
    Kc_tl, tau_i_tl, tau_d_tl = ZieglerNicholsTuning.tyreus_luyben_tuning(Ku, Pu)
    print(f"PID Controller (Tyreus-Luyben):")
    print(f"  Kc = {Kc_tl:.4f}")
    print(f"  τi = {tau_i_tl:.4f} s")
    print(f"  τd = {tau_d_tl:.4f} s")


def example_frequency_response():
    """Example: Frequency response analysis"""
    print("\n" + "=" * 60)
    print("FREQUENCY RESPONSE ANALYSIS")
    print("=" * 60)
    
    # First-order system
    K1 = 2.0
    tau1 = 1.0
    
    # Second-order system
    K2 = 1.0
    omega_n = 5.0
    zeta = 0.7
    
    # Frequency range
    omega = np.logspace(-2, 2, 1000)  # 0.01 to 100 rad/s
    
    # Calculate magnitude and phase for first-order
    mag1 = [FrequencyResponse.magnitude_first_order(K1, tau1, w) for w in omega]
    phase1 = [math.degrees(FrequencyResponse.phase_first_order(tau1, w)) for w in omega]
    
    # Calculate magnitude and phase for second-order
    mag2 = [FrequencyResponse.magnitude_second_order(K2, omega_n, zeta, w) 
            for w in omega]
    phase2 = [math.degrees(FrequencyResponse.phase_second_order(omega_n, zeta, w)) 
              for w in omega]
    
    # Convert magnitude to dB
    mag1_db = [20 * math.log10(m) if m > 0 else -100 for m in mag1]
    mag2_db = [20 * math.log10(m) if m > 0 else -100 for m in mag2]
    
    # Plot Bode plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # First-order system
    axes[0, 0].semilogx(omega, mag1_db, 'b-', linewidth=2, label='First-Order')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_title('First-Order System Bode Plot')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[1, 0].semilogx(omega, phase1, 'b-', linewidth=2, label='First-Order')
    axes[1, 0].set_xlabel('Frequency (rad/s)')
    axes[1, 0].set_ylabel('Phase (degrees)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Second-order system
    axes[0, 1].semilogx(omega, mag2_db, 'r-', linewidth=2, label='Second-Order')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].set_title('Second-Order System Bode Plot')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 1].semilogx(omega, phase2, 'r-', linewidth=2, label='Second-Order')
    axes[1, 1].set_xlabel('Frequency (rad/s)')
    axes[1, 1].set_ylabel('Phase (degrees)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFirst-Order System:")
    print(f"  K = {K1}, τ = {tau1} s")
    print(f"\nSecond-Order System:")
    print(f"  K = {K2}, ωn = {omega_n} rad/s, ζ = {zeta}")


def example_process_models():
    """Example: Process model calculations"""
    print("\n" + "=" * 60)
    print("PROCESS MODEL CALCULATIONS")
    print("=" * 60)
    
    # CSTR example
    print("\n1. CSTR Steady-State Concentration:")
    print("-" * 40)
    C_A0 = 1.0  # Inlet concentration (mol/L)
    V = 100.0  # Reactor volume (L)
    F = 10.0  # Flow rate (L/s)
    k = 0.05  # Reaction rate constant (1/s)
    
    C_A_ss = ProcessModels.cstr_steady_state(C_A0, V, F, k)
    tau = ProcessModels.cstr_time_constant(V, F)
    
    print(f"Inlet Concentration: {C_A0} mol/L")
    print(f"Reactor Volume: {V} L")
    print(f"Flow Rate: {F} L/s")
    print(f"Rate Constant: {k} 1/s")
    print(f"Residence Time (τ = V/F): {tau} s")
    print(f"Steady-State Concentration: {C_A_ss:.4f} mol/L")
    
    # Heat exchanger example
    print("\n2. Heat Exchanger Process Gain:")
    print("-" * 40)
    F = 1.0  # Flow rate (kg/s)
    Cp = 4180.0  # Heat capacity (J/(kg·K))
    
    K_he = ProcessModels.heat_exchanger_gain(None, F, None, Cp, None, None, None)
    print(f"Flow Rate: {F} kg/s")
    print(f"Heat Capacity: {Cp} J/(kg·K)")
    print(f"Process Gain: K ≈ 1/(F·Cp) = {K_he:.6e} K/W")


def example_closed_loop_analysis():
    """Example: Closed-loop system analysis"""
    print("\n" + "=" * 60)
    print("CLOSED-LOOP ANALYSIS")
    print("=" * 60)
    
    # Process parameters
    Kp = 2.0  # Process gain
    tau_p = 3.0  # Process time constant (s)
    
    # Controller gains to test
    Kc_values = [0.5, 1.0, 2.0, 5.0]
    
    print(f"\nProcess: Kp = {Kp}, τp = {tau_p} s")
    print("\nClosed-Loop Performance (P Control):")
    print("-" * 60)
    print(f"{'Kc':>8} {'K_cl':>12} {'τ_cl (s)':>12} {'Offset Factor':>15}")
    print("-" * 60)
    
    for Kc in Kc_values:
        K_cl, tau_cl = ClosedLoopAnalysis.closed_loop_transfer_function(
            Kc, Kp, tau_p)
        offset_factor = 1 / (1 + Kc * Kp)
        print(f"{Kc:>8.2f} {K_cl:>12.4f} {tau_cl:>12.4f} {offset_factor:>15.4f}")
    
    # Setpoint change example
    print("\nSetpoint Tracking (Step change = 1.0):")
    print("-" * 60)
    setpoint_change = 1.0
    Kc = 2.0
    
    error = ClosedLoopAnalysis.setpoint_tracking_error(
        Kc, Kp, setpoint_change)
    print(f"With Kc = {Kc}:")
    print(f"  Steady-state error = {error:.4f}")
    print(f"  Final output = {setpoint_change - error:.4f}")
    print(f"  Offset = {error/setpoint_change * 100:.2f}%")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("PROCESS DYNAMICS AND CONTROL - EXAMPLE CALCULATIONS")
    print("=" * 60)
    
    # Run examples
    example_first_order_system()
    example_second_order_system()
    example_pid_controller()
    example_ziegler_nichols_tuning()
    example_frequency_response()
    example_process_models()
    example_closed_loop_analysis()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
