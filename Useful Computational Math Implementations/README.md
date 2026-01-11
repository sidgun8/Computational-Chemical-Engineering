# Computational Mathematics for Chemical Engineering

A comprehensive collection of numerical methods and mathematical tools for mathematical modeling and calculations in chemical engineering.

## Overview

This module provides implementations of essential numerical methods commonly used in chemical engineering applications, including:

- **Numerical Integration**: Runge-Kutta methods (1st, 2nd, 4th order), adaptive RK45, Euler's method
- **Linear Algebra**: Gauss-Jordan elimination, LU decomposition, matrix operations, eigenvalue computation
- **Root Finding**: Bisection, Newton-Raphson, secant method, false position
- **Optimization**: Gradient descent, Newton-Raphson optimization, golden section search
- **Fourier Analysis**: Fourier series, Discrete Fourier Transform (DFT), power spectral density
- **Interpolation**: Linear, Lagrange, Newton polynomial, cubic spline
- **Curve Fitting**: Linear regression, polynomial regression, exponential and power law fitting
- **Statistics**: Mean, variance, standard deviation, correlation coefficient
- **Numerical Integration**: Trapezoidal rule, Simpson's rule

## Structure

The module is organized into the following classes:

### `NumericalIntegration`
Methods for solving ordinary differential equations (ODEs) and numerical integration:
- `runge_kutta_4()` - 4th order Runge-Kutta for single ODE
- `runge_kutta_4_system()` - 4th order Runge-Kutta for systems of ODEs
- `runge_kutta_2()` - 2nd order Runge-Kutta (Heun's method)
- `adaptive_rk45()` - Adaptive Runge-Kutta 4/5 with step size control
- `euler_method()` - Euler's method (1st order)
- `trapezoidal_rule()` - Numerical integration using trapezoidal rule
- `simpsons_rule()` - Numerical integration using Simpson's 1/3 rule

### `LinearAlgebra`
Methods for solving linear systems and matrix operations:
- `gauss_jordan_elimination()` - Solve Ax = b using Gauss-Jordan elimination
- `lu_decomposition()` - LU decomposition with partial pivoting
- `solve_lu()` - Solve system using LU decomposition
- `matrix_multiply()` - Matrix multiplication
- `matrix_determinant()` - Calculate matrix determinant
- `matrix_inverse()` - Calculate matrix inverse
- `eigenvalues_power_method()` - Find dominant eigenvalue using power method

### `RootFinding`
Methods for finding roots of equations:
- `bisection_method()` - Bisection method
- `newton_raphson()` - Newton-Raphson method
- `secant_method()` - Secant method
- `false_position()` - False position (regula falsi) method

### `Optimization`
Methods for optimization problems:
- `gradient_descent()` - Gradient descent optimization
- `newton_raphson_optimization()` - Newton-Raphson optimization (2nd order)
- `golden_section_search()` - Golden section search for unimodal functions
- `bisection_optimization()` - Find optimum by finding where derivative is zero

### `FourierAnalysis`
Methods for Fourier analysis:
- `fourier_series_coefficients()` - Calculate Fourier series coefficients
- `fourier_series_evaluate()` - Evaluate Fourier series at a point
- `discrete_fourier_transform()` - Compute DFT
- `inverse_dft()` - Compute inverse DFT
- `power_spectral_density()` - Calculate power spectral density

### `Interpolation`
Methods for interpolation:
- `linear_interpolation()` - Linear interpolation
- `lagrange_interpolation()` - Lagrange polynomial interpolation
- `newton_interpolation()` - Newton polynomial interpolation
- `cubic_spline_interpolation()` - Cubic spline interpolation (natural spline)

### `CurveFitting`
Methods for curve fitting and regression:
- `linear_regression()` - Linear regression (y = ax + b)
- `polynomial_regression()` - Polynomial regression
- `exponential_fit()` - Exponential curve fitting (y = a * exp(b*x))
- `power_law_fit()` - Power law fitting (y = a * x^b)

### `Statistics`
Basic statistical calculations:
- `mean()` - Calculate mean
- `variance()` - Calculate variance
- `standard_deviation()` - Calculate standard deviation
- `correlation_coefficient()` - Calculate Pearson correlation coefficient

## Usage Examples

### Example 1: Solving an ODE (First-order decay reaction)

```python
from computationalMath import NumericalIntegration
import math

def decay_ode(t, y):
    """dy/dt = -k*y"""
    k = 0.5
    return -k * y

times, values = NumericalIntegration.runge_kutta_4(
    f=decay_ode, y0=1.0, t0=0.0, t_end=5.0, h=0.1
)

print(f"Solution at t = {times[-1]}: y = {values[-1]}")
```

### Example 2: Solving a Linear System

```python
from computationalMath import LinearAlgebra

A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
b = [8, -11, -3]

x = LinearAlgebra.gauss_jordan_elimination(A, b)
print(f"Solution: x = {x}")
```

### Example 3: Root Finding

```python
from computationalMath import RootFinding

def func(x):
    return x**3 - x - 2

def dfunc(x):
    return 3*x**2 - 1

root, iterations = RootFinding.newton_raphson(func, dfunc, x0=1.5)
print(f"Root: {root}, Iterations: {iterations}")
```

### Example 4: Optimization

```python
from computationalMath import Optimization

def objective(p):
    x, y = p[0], p[1]
    return (x - 2)**2 + (y - 3)**2

def gradient(p):
    x, y = p[0], p[1]
    return [2*(x - 2), 2*(y - 3)]

x_opt, iters, obj_vals = Optimization.gradient_descent(
    f=objective, grad=gradient, x0=[0.0, 0.0], alpha=0.1
)
print(f"Optimal point: {x_opt}")
```

### Example 5: Curve Fitting

```python
from computationalMath import CurveFitting

x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.1, 4.2, 6.1, 8.0, 10.1]

slope, intercept, r_squared = CurveFitting.linear_regression(x_data, y_data)
print(f"y = {slope}*x + {intercept}, R² = {r_squared}")
```

## Running Examples

The `math_examples.py` file contains comprehensive examples demonstrating all the methods:

```bash
python math_examples.py
```

## Dependencies

- Python 3.6+
- Standard library only (math, cmath, typing)

No external dependencies required! All methods are implemented using only Python's standard library.

## Chemical Engineering Applications

These methods are particularly useful for:

- **Process Dynamics**: Solving ODEs for reactor dynamics, heat exchangers, etc.
- **Process Control**: System identification, frequency response analysis
- **Optimization**: Process optimization, parameter estimation
- **Data Analysis**: Interpolation, curve fitting, regression analysis
- **Numerical Simulation**: Finite difference methods, iterative solvers
- **Signal Processing**: Analysis of process signals, FFT for frequency domain analysis

## Notes

- All methods include comprehensive error handling
- Methods use appropriate tolerances and convergence criteria
- Matrix operations include checks for singularity
- Root finding methods verify function behavior at boundaries
- Numerical methods use appropriate step sizes and iteration limits

## License

This module is provided as-is for educational and research purposes.
