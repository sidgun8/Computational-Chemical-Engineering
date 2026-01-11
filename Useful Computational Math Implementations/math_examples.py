"""
Examples: Computational Mathematics for Chemical Engineering
Demonstrates usage of various numerical methods and mathematical tools
"""

import math
from computationalMath import (
    NumericalIntegration, LinearAlgebra, RootFinding, 
    Optimization, FourierAnalysis, Interpolation, CurveFitting, Statistics
)


# ============================================================================
# Example 1: Runge-Kutta Method for ODE Solving
# ============================================================================
print("=" * 70)
print("Example 1: Solving ODE using Runge-Kutta 4th order")
print("=" * 70)
print("Problem: Solve dy/dt = -k*y (first-order decay reaction)")
print("Initial condition: y(0) = 1.0, k = 0.5\n")

def decay_ode(t, y):
    """dy/dt = -k*y"""
    k = 0.5
    return -k * y

# Solve using RK4
times, values = NumericalIntegration.runge_kutta_4(
    f=decay_ode, y0=1.0, t0=0.0, t_end=5.0, h=0.1
)

print(f"Solution at t = {times[-1]:.1f}: y = {values[-1]:.6f}")
print(f"Analytical solution: y = {math.exp(-0.5 * times[-1]):.6f}")
print(f"Error: {abs(values[-1] - math.exp(-0.5 * times[-1])):.8f}\n")


# ============================================================================
# Example 2: System of ODEs (CSTR with multiple reactions)
# ============================================================================
print("=" * 70)
print("Example 2: System of ODEs - CSTR with reactions")
print("=" * 70)
print("Problem: dA/dt = -k1*A, dB/dt = k1*A - k2*B")
print("Initial: A(0) = 1.0, B(0) = 0.0\n")

def cstr_reactions(t, Y):
    """System of ODEs for CSTR"""
    A, B = Y[0], Y[1]
    k1, k2 = 0.3, 0.1
    dA_dt = -k1 * A
    dB_dt = k1 * A - k2 * B
    return [dA_dt, dB_dt]

times_sys, values_sys = NumericalIntegration.runge_kutta_4_system(
    f=cstr_reactions, y0=[1.0, 0.0], t0=0.0, t_end=10.0, h=0.1
)

print(f"At t = {times_sys[-1]:.1f}:")
print(f"  A = {values_sys[-1][0]:.6f}")
print(f"  B = {values_sys[-1][1]:.6f}\n")


# ============================================================================
# Example 3: Gauss-Jordan Elimination for Linear Systems
# ============================================================================
print("=" * 70)
print("Example 3: Solving Linear System using Gauss-Jordan Elimination")
print("=" * 70)
print("Problem: Solve Ax = b where:")
print("  [2  1 -1]   [x1]   [8]")
print("  [-3 -1  2] [x2] = [-11]")
print("  [-2  1  2]  [x3]   [-3]\n")

A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
b = [8, -11, -3]

x = LinearAlgebra.gauss_jordan_elimination(A, b)
print(f"Solution: x1 = {x[0]:.2f}, x2 = {x[1]:.2f}, x3 = {x[2]:.2f}")
print(f"Verification:")
print(f"  Equation 1: 2*{x[0]:.2f} + {x[1]:.2f} - {x[2]:.2f} = {2*x[0] + x[1] - x[2]:.2f}")
print(f"  Expected: 8.0\n")


# ============================================================================
# Example 4: LU Decomposition
# ============================================================================
print("=" * 70)
print("Example 4: LU Decomposition")
print("=" * 70)

A_lu = [[4, 3], [6, 3]]
b_lu = [10, 12]

L, U, pivot = LinearAlgebra.lu_decomposition(A_lu)
x_lu = LinearAlgebra.solve_lu(L, U, pivot, b_lu)

print("Matrix A:")
for row in A_lu:
    print(f"  {row}")
print(f"\nSolution: x1 = {x_lu[0]:.2f}, x2 = {x_lu[1]:.2f}\n")


# ============================================================================
# Example 5: Root Finding - Newton-Raphson
# ============================================================================
print("=" * 70)
print("Example 5: Root Finding using Newton-Raphson Method")
print("=" * 70)
print("Problem: Find root of f(x) = x^3 - x - 2\n")

def func(x):
    return x**3 - x - 2

def dfunc(x):
    return 3*x**2 - 1

root, iterations = RootFinding.newton_raphson(func, dfunc, x0=1.5, tol=1e-8)
print(f"Root: x = {root:.8f}")
print(f"Iterations: {iterations}")
print(f"Verification: f({root:.8f}) = {func(root):.10e}\n")


# ============================================================================
# Example 6: Root Finding - Bisection Method
# ============================================================================
print("=" * 70)
print("Example 6: Root Finding using Bisection Method")
print("=" * 70)
print("Problem: Find root of f(x) = x^2 - 4\n")

def func2(x):
    return x**2 - 4

root_bisect, iter_bisect = RootFinding.bisection_method(func2, a=0.0, b=3.0, tol=1e-8)
print(f"Root: x = {root_bisect:.8f}")
print(f"Iterations: {iter_bisect}")
print(f"Verification: f({root_bisect:.8f}) = {func2(root_bisect):.10e}")
print(f"Expected: x = 2.0\n")


# ============================================================================
# Example 7: Optimization - Gradient Descent
# ============================================================================
print("=" * 70)
print("Example 7: Optimization using Gradient Descent")
print("=" * 70)
print("Problem: Minimize f(x,y) = (x-2)^2 + (y-3)^2\n")

def objective(p):
    x, y = p[0], p[1]
    return (x - 2)**2 + (y - 3)**2

def gradient(p):
    x, y = p[0], p[1]
    return [2*(x - 2), 2*(y - 3)]

x_opt, iters, obj_vals = Optimization.gradient_descent(
    f=objective, grad=gradient, x0=[0.0, 0.0], alpha=0.1, tol=1e-6
)

print(f"Optimal point: ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
print(f"Iterations: {iters}")
print(f"Optimal value: {objective(x_opt):.10e}")
print(f"Expected: (2.0, 3.0) with value 0.0\n")


# ============================================================================
# Example 8: Golden Section Search
# ============================================================================
print("=" * 70)
print("Example 8: Optimization using Golden Section Search")
print("=" * 70)
print("Problem: Minimize f(x) = x^2 - 4*x + 4 in [0, 5]\n")

def quadratic(x):
    return x**2 - 4*x + 4

x_golden, iter_golden = Optimization.golden_section_search(quadratic, a=0.0, b=5.0, tol=1e-8)
print(f"Optimal point: x = {x_golden:.8f}")
print(f"Iterations: {iter_golden}")
print(f"Optimal value: {quadratic(x_golden):.10e}")
print(f"Expected: x = 2.0 with value 0.0\n")


# ============================================================================
# Example 9: Fourier Series
# ============================================================================
print("=" * 70)
print("Example 9: Fourier Series for Square Wave")
print("=" * 70)

def square_wave(t):
    """Square wave: 1 for 0 < t < π, -1 for π < t < 2π"""
    period = 2 * math.pi
    t_mod = t % period
    return 1.0 if t_mod < period / 2 else -1.0

period = 2 * math.pi
an, bn = FourierAnalysis.fourier_series_coefficients(square_wave, period, n_terms=10)

print("First few coefficients:")
print(f"  a0/2 = {an[0]:.6f}")
for n in range(1, min(6, len(an))):
    print(f"  a{n} = {an[n]:.6f}, b{n} = {bn[n]:.6f}")

# Evaluate at a point
t_eval = math.pi / 4
value = FourierAnalysis.fourier_series_evaluate(an, bn, period, t_eval)
print(f"\nFourier series value at t = π/4: {value:.6f}")
print(f"Actual square wave value: {square_wave(t_eval):.6f}\n")


# ============================================================================
# Example 10: Interpolation
# ============================================================================
print("=" * 70)
print("Example 10: Lagrange Interpolation")
print("=" * 70)
print("Problem: Interpolate data points and find value at x = 1.5\n")

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [1.0, 4.0, 9.0, 16.0]  # y = x^2

x_interp = 1.5
y_interp = Interpolation.lagrange_interpolation(x_data, y_data, x_interp)

print(f"Data points: {list(zip(x_data, y_data))}")
print(f"Interpolated value at x = {x_interp}: y = {y_interp:.6f}")
print(f"Actual value (x^2): {x_interp**2:.6f}\n")


# ============================================================================
# Example 11: Cubic Spline Interpolation
# ============================================================================
print("=" * 70)
print("Example 11: Cubic Spline Interpolation")
print("=" * 70)

x_spline = [0.0, 1.0, 2.0, 3.0, 4.0]
y_spline = [0.0, 1.0, 4.0, 9.0, 16.0]  # y = x^2

x_spline_interp = 1.5
y_spline_interp = Interpolation.cubic_spline_interpolation(x_spline, y_spline, x_spline_interp)

print(f"Interpolated value at x = {x_spline_interp}: y = {y_spline_interp:.6f}")
print(f"Actual value: {x_spline_interp**2:.6f}\n")


# ============================================================================
# Example 12: Linear Regression
# ============================================================================
print("=" * 70)
print("Example 12: Linear Regression")
print("=" * 70)

x_reg = [1.0, 2.0, 3.0, 4.0, 5.0]
y_reg = [2.1, 4.2, 6.1, 8.0, 10.1]  # y ≈ 2*x (with noise)

slope, intercept, r_squared = CurveFitting.linear_regression(x_reg, y_reg)

print(f"Data: x = {x_reg}, y = {y_reg}")
print(f"Fitted line: y = {slope:.4f} * x + {intercept:.4f}")
print(f"R-squared: {r_squared:.6f}")
print(f"Expected: y ≈ 2*x + 0\n")


# ============================================================================
# Example 13: Polynomial Regression
# ============================================================================
print("=" * 70)
print("Example 13: Polynomial Regression (2nd degree)")
print("=" * 70)

x_poly = [0.0, 1.0, 2.0, 3.0, 4.0]
y_poly = [1.0, 2.0, 5.0, 10.0, 17.0]  # y ≈ x^2 + 1

coeffs, r2 = CurveFitting.polynomial_regression(x_poly, y_poly, degree=2)

print(f"Data: x = {x_poly}, y = {y_poly}")
print(f"Fitted polynomial: y = {coeffs[2]:.4f}*x^2 + {coeffs[1]:.4f}*x + {coeffs[0]:.4f}")
print(f"R-squared: {r2:.6f}")
print(f"Expected: y ≈ x^2 + 0*x + 1\n")


# ============================================================================
# Example 14: Exponential Fit
# ============================================================================
print("=" * 70)
print("Example 14: Exponential Curve Fitting")
print("=" * 70)

x_exp = [0.0, 1.0, 2.0, 3.0, 4.0]
y_exp = [1.0, 1.5, 2.25, 3.375, 5.0625]  # y ≈ 1 * exp(0.405*x) ≈ 1.5^x

a_exp, b_exp, r2_exp = CurveFitting.exponential_fit(x_exp, y_exp)

print(f"Data: x = {x_exp}, y = {y_exp}")
print(f"Fitted exponential: y = {a_exp:.4f} * exp({b_exp:.4f} * x)")
print(f"R-squared: {r2_exp:.6f}\n")


# ============================================================================
# Example 15: Statistical Calculations
# ============================================================================
print("=" * 70)
print("Example 15: Statistical Analysis")
print("=" * 70)

data = [2.3, 4.5, 3.7, 5.1, 4.9, 3.2, 4.8, 5.3, 3.9, 4.2]

mean_val = Statistics.mean(data)
std_val = Statistics.standard_deviation(data, sample=True)
var_val = Statistics.variance(data, sample=True)

print(f"Data: {data}")
print(f"Mean: {mean_val:.4f}")
print(f"Standard deviation (sample): {std_val:.4f}")
print(f"Variance (sample): {var_val:.4f}\n")


# ============================================================================
# Example 16: Numerical Integration
# ============================================================================
print("=" * 70)
print("Example 16: Numerical Integration - Trapezoidal Rule")
print("=" * 70)
print("Problem: Integrate f(x) = x^2 from 0 to 2\n")

def integrand(x):
    return x**2

integral_trap = NumericalIntegration.trapezoidal_rule(integrand, 0.0, 2.0, n=100)
integral_simpson = NumericalIntegration.simpsons_rule(integrand, 0.0, 2.0, n=100)
exact = 2**3 / 3  # 8/3

print(f"Trapezoidal rule (n=100): {integral_trap:.8f}")
print(f"Simpson's rule (n=100): {integral_simpson:.8f}")
print(f"Exact value: {exact:.8f}")
print(f"Error (Trapezoidal): {abs(integral_trap - exact):.8f}")
print(f"Error (Simpson's): {abs(integral_simpson - exact):.8f}\n")


# ============================================================================
# Example 17: Matrix Operations
# ============================================================================
print("=" * 70)
print("Example 17: Matrix Operations")
print("=" * 70)

M1 = [[1, 2], [3, 4]]
M2 = [[5, 6], [7, 8]]

product = LinearAlgebra.matrix_multiply(M1, M2)
det = LinearAlgebra.matrix_determinant(M1)

print(f"Matrix M1: {M1}")
print(f"Matrix M2: {M2}")
print(f"M1 * M2: {product}")
print(f"det(M1): {det:.2f}")

try:
    M_inv = LinearAlgebra.matrix_inverse(M1)
    print(f"M1 inverse: {M_inv}")
except ValueError as e:
    print(f"Could not compute inverse: {e}")

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
