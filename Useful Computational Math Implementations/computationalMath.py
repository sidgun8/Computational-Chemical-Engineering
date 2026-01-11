"""
Computational Mathematics for Chemical Engineering
Comprehensive collection of numerical methods and mathematical tools
for mathematical modeling and calculations in chemical engineering
"""

import math
import cmath
from typing import Callable, List, Tuple, Optional, Union


class NumericalIntegration:
    """
    Class for numerical integration methods including Runge-Kutta methods
    """
    
    @staticmethod
    def runge_kutta_4(f: Callable, y0: float, t0: float, t_end: float, h: float) -> Tuple[List[float], List[float]]:
        """
        Solve ODE using 4th order Runge-Kutta method.
        
        Solves dy/dt = f(t, y) with initial condition y(t0) = y0
        
        Parameters:
        f: Function f(t, y) representing dy/dt
        y0: Initial value y(t0)
        t0: Initial time
        t_end: End time
        h: Step size
        
        Returns:
        Tuple of (time_points, solution_values)
        """
        t = t0
        y = y0
        times = [t0]
        values = [y0]
        
        while t < t_end:
            k1 = h * f(t, y)
            k2 = h * f(t + h/2, y + k1/2)
            k3 = h * f(t + h/2, y + k2/2)
            k4 = h * f(t + h, y + k3)
            
            y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
            t = t + h
            
            times.append(t)
            values.append(y)
        
        return times, values
    
    @staticmethod
    def runge_kutta_4_system(f: Callable, y0: List[float], t0: float, t_end: float, h: float) -> Tuple[List[float], List[List[float]]]:
        """
        Solve system of ODEs using 4th order Runge-Kutta method.
        
        Solves dY/dt = f(t, Y) where Y is a vector
        
        Parameters:
        f: Function f(t, Y) returning a list/array of derivatives
        y0: Initial values (list)
        t0: Initial time
        t_end: End time
        h: Step size
        
        Returns:
        Tuple of (time_points, solution_values list of lists)
        """
        t = t0
        y = list(y0)  # Make a copy
        times = [t0]
        values = [y0.copy()]
        
        def vec_add(a, b):
            """Element-wise addition"""
            return [ai + bi for ai, bi in zip(a, b)]
        
        def vec_mult_scalar(v, s):
            """Multiply vector by scalar"""
            return [vi * s for vi in v]
        
        while t < t_end:
            k1 = vec_mult_scalar(f(t, y), h)
            y_k2 = vec_add(y, vec_mult_scalar(k1, 0.5))
            k2 = vec_mult_scalar(f(t + h/2, y_k2), h)
            y_k3 = vec_add(y, vec_mult_scalar(k2, 0.5))
            k3 = vec_mult_scalar(f(t + h/2, y_k3), h)
            y_k4 = vec_add(y, k3)
            k4 = vec_mult_scalar(f(t + h, y_k4), h)
            
            # y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
            k_sum = vec_add(k1, vec_mult_scalar(vec_add(k2, k3), 2))
            k_sum = vec_add(k_sum, k4)
            y = vec_add(y, vec_mult_scalar(k_sum, 1/6))
            t = t + h
            
            times.append(t)
            values.append(y.copy())
        
        return times, values
    
    @staticmethod
    def runge_kutta_2(f: Callable, y0: float, t0: float, t_end: float, h: float) -> Tuple[List[float], List[float]]:
        """
        Solve ODE using 2nd order Runge-Kutta (Heun's method).
        
        Parameters:
        f: Function f(t, y) representing dy/dt
        y0: Initial value y(t0)
        t0: Initial time
        t_end: End time
        h: Step size
        
        Returns:
        Tuple of (time_points, solution_values)
        """
        t = t0
        y = y0
        times = [t0]
        values = [y0]
        
        while t < t_end:
            k1 = h * f(t, y)
            k2 = h * f(t + h, y + k1)
            
            y = y + (k1 + k2) / 2
            t = t + h
            
            times.append(t)
            values.append(y)
        
        return times, values
    
    @staticmethod
    def adaptive_rk45(f: Callable, y0: float, t0: float, t_end: float, h0: float = 0.1, 
                      tol: float = 1e-6, max_iter: int = 10000) -> Tuple[List[float], List[float]]:
        """
        Adaptive Runge-Kutta 4/5 method with step size control.
        
        Parameters:
        f: Function f(t, y) representing dy/dt
        y0: Initial value y(t0)
        t0: Initial time
        t_end: End time
        h0: Initial step size
        tol: Error tolerance
        max_iter: Maximum number of iterations
        
        Returns:
        Tuple of (time_points, solution_values)
        """
        t = t0
        y = y0
        h = h0
        times = [t0]
        values = [y0]
        iter_count = 0
        
        while t < t_end and iter_count < max_iter:
            # 4th order RK
            k1 = h * f(t, y)
            k2 = h * f(t + h/4, y + k1/4)
            k3 = h * f(t + 3*h/8, y + 3*k1/32 + 9*k2/32)
            k4 = h * f(t + 12*h/13, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
            k5 = h * f(t + h, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
            k6 = h * f(t + h/2, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
            
            y4 = y + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
            y5 = y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
            
            error = abs(y5 - y4)
            
            if error < tol or h < 1e-10:
                y = y5
                t = t + h
                times.append(t)
                values.append(y)
            
            # Adjust step size
            if error > 0:
                h = 0.9 * h * (tol / error) ** 0.2
            else:
                h = 2 * h
            
            h = min(h, t_end - t)
            iter_count += 1
        
        return times, values
    
    @staticmethod
    def euler_method(f: Callable, y0: float, t0: float, t_end: float, h: float) -> Tuple[List[float], List[float]]:
        """
        Solve ODE using Euler's method (1st order).
        
        Parameters:
        f: Function f(t, y) representing dy/dt
        y0: Initial value y(t0)
        t0: Initial time
        t_end: End time
        h: Step size
        
        Returns:
        Tuple of (time_points, solution_values)
        """
        t = t0
        y = y0
        times = [t0]
        values = [y0]
        
        while t < t_end:
            y = y + h * f(t, y)
            t = t + h
            
            times.append(t)
            values.append(y)
        
        return times, values
    
    @staticmethod
    def trapezoidal_rule(f: Callable, a: float, b: float, n: int) -> float:
        """
        Numerical integration using trapezoidal rule.
        
        Parameters:
        f: Function to integrate
        a: Lower limit
        b: Upper limit
        n: Number of intervals
        
        Returns:
        Integral value
        """
        h = (b - a) / n
        integral = (f(a) + f(b)) / 2
        
        for i in range(1, n):
            integral += f(a + i * h)
        
        return integral * h
    
    @staticmethod
    def simpsons_rule(f: Callable, a: float, b: float, n: int) -> float:
        """
        Numerical integration using Simpson's 1/3 rule.
        
        Parameters:
        f: Function to integrate
        a: Lower limit
        b: Upper limit
        n: Number of intervals (must be even)
        
        Returns:
        Integral value
        """
        if n % 2 != 0:
            n += 1  # Make n even
        
        h = (b - a) / n
        integral = f(a) + f(b)
        
        for i in range(1, n):
            if i % 2 == 0:
                integral += 2 * f(a + i * h)
            else:
                integral += 4 * f(a + i * h)
        
        return integral * h / 3


class LinearAlgebra:
    """
    Class for linear algebra operations including Gauss-Jordan elimination
    """
    
    @staticmethod
    def gauss_jordan_elimination(A: List[List[float]], b: List[float]) -> List[float]:
        """
        Solve linear system Ax = b using Gauss-Jordan elimination.
        
        Parameters:
        A: Coefficient matrix (list of lists)
        b: Right-hand side vector (list)
        
        Returns:
        Solution vector x (list)
        """
        n = len(A)
        # Create augmented matrix [A|b]
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]
        
        # Forward elimination with partial pivoting
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # Check for singular matrix
            if abs(aug[i][i]) < 1e-10:
                raise ValueError("Matrix is singular or nearly singular")
            
            # Make all rows below this one 0 in current column
            for k in range(i + 1, n):
                factor = aug[k][i] / aug[i][i]
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]
        
        # Backward elimination
        for i in range(n - 1, -1, -1):
            # Make all rows above this one 0 in current column
            for k in range(i - 1, -1, -1):
                factor = aug[k][i] / aug[i][i]
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]
            
            # Normalize pivot
            aug[i][n] /= aug[i][i]
            aug[i][i] = 1.0
        
        # Extract solution
        return [row[n] for row in aug]
    
    @staticmethod
    def lu_decomposition(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[int]]:
        """
        Perform LU decomposition with partial pivoting.
        Returns PA = LU where P is permutation matrix (stored as pivot indices).
        
        Parameters:
        A: Square matrix (list of lists)
        
        Returns:
        Tuple of (L, U, pivot_indices)
        """
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        U = [row[:] for row in A]
        pivot = list(range(n))
        
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(U[k][i]) > abs(U[max_row][i]):
                    max_row = k
            
            if max_row != i:
                U[i], U[max_row] = U[max_row], U[i]
                pivot[i], pivot[max_row] = pivot[max_row], pivot[i]
            
            # Compute L and U
            for j in range(i, n):
                if i == j:
                    L[i][i] = 1.0
                else:
                    L[j][i] = U[j][i] / U[i][i]
                    for k in range(i, n):
                        U[j][k] -= L[j][i] * U[i][k]
        
        return L, U, pivot
    
    @staticmethod
    def solve_lu(L: List[List[float]], U: List[List[float]], pivot: List[int], b: List[float]) -> List[float]:
        """
        Solve system using LU decomposition.
        
        Parameters:
        L: Lower triangular matrix
        U: Upper triangular matrix
        pivot: Pivot indices from LU decomposition
        b: Right-hand side vector
        
        Returns:
        Solution vector
        """
        n = len(L)
        # Permute b
        b_perm = [b[pivot[i]] for i in range(n)]
        
        # Forward substitution: Ly = b_perm
        y = [0.0] * n
        for i in range(n):
            y[i] = b_perm[i]
            for j in range(i):
                y[i] -= L[i][j] * y[j]
            y[i] /= L[i][i]
        
        # Backward substitution: Ux = y
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = y[i]
            for j in range(i + 1, n):
                x[i] -= U[i][j] * x[j]
            x[i] /= U[i][i]
        
        return x
    
    @staticmethod
    def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """
        Multiply two matrices.
        
        Parameters:
        A: First matrix (m x n)
        B: Second matrix (n x p)
        
        Returns:
        Product matrix (m x p)
        """
        m, n, p = len(A), len(A[0]), len(B[0])
        C = [[0.0] * p for _ in range(m)]
        
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]
        
        return C
    
    @staticmethod
    def matrix_determinant(A: List[List[float]]) -> float:
        """
        Calculate determinant using LU decomposition.
        
        Parameters:
        A: Square matrix
        
        Returns:
        Determinant value
        """
        n = len(A)
        if n != len(A[0]):
            raise ValueError("Matrix must be square")
        
        _, U, pivot = LinearAlgebra.lu_decomposition(A)
        
        det = 1.0
        for i in range(n):
            det *= U[i][i]
        
        # Account for row swaps (odd number of swaps = -1)
        swaps = sum(1 for i, p in enumerate(pivot) if i != p)
        if swaps % 2 == 1:
            det = -det
        
        return det
    
    @staticmethod
    def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
        """
        Calculate matrix inverse using Gauss-Jordan elimination.
        
        Parameters:
        A: Square matrix
        
        Returns:
        Inverse matrix
        """
        n = len(A)
        # Create augmented matrix [A|I]
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] 
               for i, row in enumerate(A)]
        
        # Gauss-Jordan elimination
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            if abs(aug[i][i]) < 1e-10:
                raise ValueError("Matrix is singular and cannot be inverted")
            
            # Normalize pivot row
            pivot_val = aug[i][i]
            for j in range(2 * n):
                aug[i][j] /= pivot_val
            
            # Eliminate column
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]
        
        # Extract inverse
        return [row[n:] for row in aug]
    
    @staticmethod
    def eigenvalues_power_method(A: List[List[float]], max_iter: int = 100, tol: float = 1e-6) -> Tuple[float, List[float]]:
        """
        Find dominant eigenvalue and eigenvector using power method.
        
        Parameters:
        A: Square matrix
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
        Returns:
        Tuple of (eigenvalue, eigenvector)
        """
        n = len(A)
        x = [1.0] * n  # Initial guess
        
        for _ in range(max_iter):
            # Multiply A by x
            Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
            
            # Normalize
            norm = math.sqrt(sum(ax**2 for ax in Ax))
            if norm < 1e-10:
                raise ValueError("Power method failed: zero vector")
            
            x_new = [ax / norm for ax in Ax]
            
            # Calculate eigenvalue (Rayleigh quotient)
            lambda_new = sum(Ax[i] * x[i] for i in range(n)) / sum(x[i]**2 for i in range(n))
            
            # Check convergence
            if all(abs(x_new[i] - x[i]) < tol for i in range(n)):
                return lambda_new, x_new
            
            x = x_new
        
        # Return last estimate
        Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        lambda_final = sum(Ax[i] * x[i] for i in range(n)) / sum(x[i]**2 for i in range(n))
        return lambda_final, x


class RootFinding:
    """
    Class for root finding methods
    """
    
    @staticmethod
    def bisection_method(f: Callable, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> Tuple[float, int]:
        """
        Find root using bisection method.
        
        Parameters:
        f: Function
        a: Lower bound
        b: Upper bound
        tol: Tolerance
        max_iter: Maximum iterations
        
        Returns:
        Tuple of (root, iterations)
        """
        if f(a) * f(b) > 0:
            raise ValueError("Function must have opposite signs at a and b")
        
        for i in range(max_iter):
            c = (a + b) / 2
            if abs(f(c)) < tol or (b - a) / 2 < tol:
                return c, i + 1
            
            if f(c) * f(a) < 0:
                b = c
            else:
                a = c
        
        return (a + b) / 2, max_iter
    
    @staticmethod
    def newton_raphson(f: Callable, df: Callable, x0: float, tol: float = 1e-6, max_iter: int = 100) -> Tuple[float, int]:
        """
        Find root using Newton-Raphson method.
        
        Parameters:
        f: Function
        df: Derivative of function
        x0: Initial guess
        tol: Tolerance
        max_iter: Maximum iterations
        
        Returns:
        Tuple of (root, iterations)
        """
        x = x0
        for i in range(max_iter):
            fx = f(x)
            if abs(fx) < tol:
                return x, i + 1
            
            dfx = df(x)
            if abs(dfx) < 1e-10:
                raise ValueError("Derivative is zero, cannot continue")
            
            x_new = x - fx / dfx
            if abs(x_new - x) < tol:
                return x_new, i + 1
            
            x = x_new
        
        return x, max_iter
    
    @staticmethod
    def secant_method(f: Callable, x0: float, x1: float, tol: float = 1e-6, max_iter: int = 100) -> Tuple[float, int]:
        """
        Find root using secant method.
        
        Parameters:
        f: Function
        x0: First initial guess
        x1: Second initial guess
        tol: Tolerance
        max_iter: Maximum iterations
        
        Returns:
        Tuple of (root, iterations)
        """
        for i in range(max_iter):
            fx0 = f(x0)
            fx1 = f(x1)
            
            if abs(fx1) < tol:
                return x1, i + 1
            
            if abs(fx1 - fx0) < 1e-10:
                raise ValueError("Function values are too close, cannot continue")
            
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            
            if abs(x_new - x1) < tol:
                return x_new, i + 1
            
            x0, x1 = x1, x_new
        
        return x1, max_iter
    
    @staticmethod
    def false_position(f: Callable, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> Tuple[float, int]:
        """
        Find root using false position (regula falsi) method.
        
        Parameters:
        f: Function
        a: Lower bound
        b: Upper bound
        tol: Tolerance
        max_iter: Maximum iterations
        
        Returns:
        Tuple of (root, iterations)
        """
        if f(a) * f(b) > 0:
            raise ValueError("Function must have opposite signs at a and b")
        
        for i in range(max_iter):
            fa = f(a)
            fb = f(b)
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            
            if abs(fc) < tol:
                return c, i + 1
            
            if fa * fc < 0:
                b = c
            else:
                a = c
        
        return (a + b) / 2, max_iter


class Optimization:
    """
    Class for optimization methods
    """
    
    @staticmethod
    def gradient_descent(f: Callable, grad: Callable, x0: List[float], 
                        alpha: float = 0.01, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[List[float], int, List[float]]:
        """
        Minimize function using gradient descent.
        
        Parameters:
        f: Objective function
        grad: Gradient function (returns list)
        x0: Initial point (list)
        alpha: Learning rate
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
        Returns:
        Tuple of (optimal_point, iterations, objective_values)
        """
        x = x0[:]
        obj_values = [f(x)]
        
        for i in range(max_iter):
            gradient = grad(x)
            
            # Update
            x_new = [x[j] - alpha * gradient[j] for j in range(len(x))]
            
            obj_values.append(f(x_new))
            
            # Check convergence
            if math.sqrt(sum((x_new[j] - x[j])**2 for j in range(len(x)))) < tol:
                return x_new, i + 1, obj_values
            
            x = x_new
        
        return x, max_iter, obj_values
    
    @staticmethod
    def newton_raphson_optimization(f: Callable, grad: Callable, hessian: Callable,
                                   x0: List[float], tol: float = 1e-6, max_iter: int = 100) -> Tuple[List[float], int]:
        """
        Minimize function using Newton-Raphson method (second order).
        
        Parameters:
        f: Objective function
        grad: Gradient function
        hessian: Hessian matrix function (returns list of lists)
        x0: Initial point
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
        Returns:
        Tuple of (optimal_point, iterations)
        """
        x = x0[:]
        
        for i in range(max_iter):
            g = grad(x)
            H = hessian(x)
            
            # Solve H * d = -g for direction d
            try:
                d = LinearAlgebra.gauss_jordan_elimination(H, [-gi for gi in g])
            except:
                # If Hessian is singular, use gradient descent step
                alpha = 0.01
                d = [-alpha * gi for gi in g]
            
            # Update
            x_new = [x[j] + d[j] for j in range(len(x))]
            
            # Check convergence
            if math.sqrt(sum((x_new[j] - x[j])**2 for j in range(len(x)))) < tol:
                return x_new, i + 1
            
            x = x_new
        
        return x, max_iter
    
    @staticmethod
    def golden_section_search(f: Callable, a: float, b: float, tol: float = 1e-6, 
                             max_iter: int = 100, maximize: bool = False) -> Tuple[float, int]:
        """
        Find optimum of unimodal function using golden section search.
        
        Parameters:
        f: Objective function
        a: Lower bound
        b: Upper bound
        tol: Tolerance
        max_iter: Maximum iterations
        maximize: If True, maximize; if False, minimize
        
        Returns:
        Tuple of (optimal_point, iterations)
        """
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        resphi = 2 - phi
        
        c = a + resphi * (b - a)
        d = b - resphi * (b - a)
        
        if maximize:
            fc, fd = -f(c), -f(d)
        else:
            fc, fd = f(c), f(d)
        
        for i in range(max_iter):
            if abs(c - d) < tol:
                return (c + d) / 2, i + 1
            
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = a + resphi * (b - a)
                fc = -f(c) if maximize else f(c)
            else:
                a = c
                c = d
                fc = fd
                d = b - resphi * (b - a)
                fd = -f(d) if maximize else f(d)
        
        return (a + b) / 2, max_iter
    
    @staticmethod
    def bisection_optimization(df: Callable, a: float, b: float, tol: float = 1e-6, 
                              max_iter: int = 100) -> Tuple[float, int]:
        """
        Find optimum by finding where derivative is zero using bisection.
        
        Parameters:
        df: Derivative of objective function
        a: Lower bound
        b: Upper bound
        tol: Tolerance
        max_iter: Maximum iterations
        
        Returns:
        Tuple of (optimal_point, iterations)
        """
        return RootFinding.bisection_method(df, a, b, tol, max_iter)


class FourierAnalysis:
    """
    Class for Fourier series and transform methods
    """
    
    @staticmethod
    def fourier_series_coefficients(f: Callable, period: float, n_terms: int) -> Tuple[List[float], List[float]]:
        """
        Calculate Fourier series coefficients for periodic function.
        
        f(t) = a0/2 + sum(an*cos(nωt) + bn*sin(nωt))
        
        Parameters:
        f: Periodic function f(t)
        period: Period of function (T)
        n_terms: Number of terms in series
        
        Returns:
        Tuple of (an_coefficients, bn_coefficients)
        """
        omega = 2 * math.pi / period
        
        # a0
        a0 = (2 / period) * NumericalIntegration.trapezoidal_rule(
            lambda t: f(t), 0, period, 1000)
        
        an = [a0 / 2]
        bn = [0.0]
        
        for n in range(1, n_terms + 1):
            # an coefficient
            an_val = (2 / period) * NumericalIntegration.trapezoidal_rule(
                lambda t: f(t) * math.cos(n * omega * t), 0, period, 1000)
            an.append(an_val)
            
            # bn coefficient
            bn_val = (2 / period) * NumericalIntegration.trapezoidal_rule(
                lambda t: f(t) * math.sin(n * omega * t), 0, period, 1000)
            bn.append(bn_val)
        
        return an, bn
    
    @staticmethod
    def fourier_series_evaluate(an: List[float], bn: List[float], period: float, t: float) -> float:
        """
        Evaluate Fourier series at given time.
        
        Parameters:
        an: Cosine coefficients (including a0/2)
        bn: Sine coefficients
        period: Period
        t: Time point
        
        Returns:
        Function value at t
        """
        omega = 2 * math.pi / period
        result = an[0]  # a0/2
        
        for n in range(1, len(an)):
            result += an[n] * math.cos(n * omega * t) + bn[n] * math.sin(n * omega * t)
        
        return result
    
    @staticmethod
    def discrete_fourier_transform(signal: List[float]) -> List[complex]:
        """
        Compute Discrete Fourier Transform (DFT).
        
        Parameters:
        signal: Input signal (list of values)
        
        Returns:
        Frequency domain representation (list of complex numbers)
        """
        N = len(signal)
        X = []
        
        for k in range(N):
            Xk = 0.0 + 0.0j
            for n in range(N):
                Xk += signal[n] * cmath.exp(-2j * math.pi * k * n / N)
            X.append(Xk)
        
        return X
    
    @staticmethod
    def inverse_dft(X: List[complex]) -> List[float]:
        """
        Compute Inverse Discrete Fourier Transform.
        
        Parameters:
        X: Frequency domain representation
        
        Returns:
        Time domain signal
        """
        N = len(X)
        x = []
        
        for n in range(N):
            xn = 0.0 + 0.0j
            for k in range(N):
                xn += X[k] * cmath.exp(2j * math.pi * k * n / N)
            x.append((xn / N).real)
        
        return x
    
    @staticmethod
    def power_spectral_density(signal: List[float], dt: float = 1.0) -> Tuple[List[float], List[float]]:
        """
        Calculate power spectral density from signal.
        
        Parameters:
        signal: Input signal
        dt: Time step
        
        Returns:
        Tuple of (frequencies, power_spectrum)
        """
        X = FourierAnalysis.discrete_fourier_transform(signal)
        N = len(signal)
        fs = 1.0 / dt  # Sampling frequency
        
        frequencies = [k * fs / N for k in range(N // 2)]
        power = [abs(X[k])**2 / N for k in range(N // 2)]
        
        return frequencies, power


class Interpolation:
    """
    Class for interpolation methods
    """
    
    @staticmethod
    def linear_interpolation(x: List[float], y: List[float], x_new: float) -> float:
        """
        Linear interpolation.
        
        Parameters:
        x: Known x values
        y: Known y values
        x_new: Point to interpolate
        
        Returns:
        Interpolated y value
        """
        if x_new < x[0] or x_new > x[-1]:
            raise ValueError("x_new is outside interpolation range")
        
        for i in range(len(x) - 1):
            if x[i] <= x_new <= x[i + 1]:
                return y[i] + (y[i + 1] - y[i]) * (x_new - x[i]) / (x[i + 1] - x[i])
        
        return y[-1]
    
    @staticmethod
    def lagrange_interpolation(x: List[float], y: List[float], x_new: float) -> float:
        """
        Lagrange polynomial interpolation.
        
        Parameters:
        x: Known x values
        y: Known y values
        x_new: Point to interpolate
        
        Returns:
        Interpolated y value
        """
        n = len(x)
        result = 0.0
        
        for i in range(n):
            Li = 1.0
            for j in range(n):
                if i != j:
                    Li *= (x_new - x[j]) / (x[i] - x[j])
            result += y[i] * Li
        
        return result
    
    @staticmethod
    def newton_divided_differences(x: List[float], y: List[float]) -> List[float]:
        """
        Calculate Newton's divided differences.
        
        Parameters:
        x: Known x values
        y: Known y values
        
        Returns:
        Coefficients for Newton polynomial
        """
        n = len(x)
        coef = y[:]
        
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
        
        return coef
    
    @staticmethod
    def newton_interpolation(x: List[float], y: List[float], x_new: float) -> float:
        """
        Newton polynomial interpolation.
        
        Parameters:
        x: Known x values
        y: Known y values
        x_new: Point to interpolate
        
        Returns:
        Interpolated y value
        """
        coef = Interpolation.newton_divided_differences(x, y)
        n = len(x)
        result = coef[0]
        
        for i in range(1, n):
            product = coef[i]
            for j in range(i):
                product *= (x_new - x[j])
            result += product
        
        return result
    
    @staticmethod
    def cubic_spline_interpolation(x: List[float], y: List[float], x_new: float) -> float:
        """
        Cubic spline interpolation (natural spline).
        
        Parameters:
        x: Known x values (must be sorted)
        y: Known y values
        x_new: Point to interpolate
        
        Returns:
        Interpolated y value
        """
        n = len(x) - 1
        h = [x[i + 1] - x[i] for i in range(n)]
        
        # Set up system for second derivatives
        A = [[0.0] * (n + 1) for _ in range(n + 1)]
        b = [0.0] * (n + 1)
        
        # Natural spline boundary conditions
        A[0][0] = 1.0
        A[n][n] = 1.0
        
        # Interior conditions
        for i in range(1, n):
            A[i][i - 1] = h[i - 1]
            A[i][i] = 2 * (h[i - 1] + h[i])
            A[i][i + 1] = h[i]
            b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        
        # Solve for second derivatives
        M = LinearAlgebra.gauss_jordan_elimination(A, b)
        
        # Find which interval x_new is in
        if x_new < x[0]:
            i = 0
        elif x_new > x[-1]:
            i = n - 1
        else:
            for i in range(n):
                if x[i] <= x_new <= x[i + 1]:
                    break
        
        # Evaluate spline
        t = (x_new - x[i]) / h[i]
        a = y[i]
        b = h[i] * (y[i + 1] - y[i]) / h[i] - h[i] * (M[i + 1] + 2 * M[i]) / 3
        c = M[i]
        d = (M[i + 1] - M[i]) / (3 * h[i])
        
        return a + b * t + c * t**2 + d * t**3


class CurveFitting:
    """
    Class for curve fitting and regression methods
    """
    
    @staticmethod
    def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
        """
        Perform linear regression y = ax + b.
        
        Parameters:
        x: Independent variable values
        y: Dependent variable values
        
        Returns:
        Tuple of (slope_a, intercept_b, r_squared)
        """
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi**2 for xi in x)
        sum_y2 = sum(yi**2 for yi in y)
        
        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        b = (sum_y - a * sum_x) / n
        
        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean)**2 for yi in y)
        ss_res = sum((y[i] - (a * x[i] + b))**2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return a, b, r_squared
    
    @staticmethod
    def polynomial_regression(x: List[float], y: List[float], degree: int) -> Tuple[List[float], float]:
        """
        Perform polynomial regression.
        
        Parameters:
        x: Independent variable values
        y: Dependent variable values
        degree: Polynomial degree
        
        Returns:
        Tuple of (coefficients [highest to lowest], r_squared)
        """
        n = len(x)
        
        # Create Vandermonde matrix
        X = [[x[i]**j for j in range(degree + 1)] for i in range(n)]
        
        # Solve normal equations: X^T * X * c = X^T * y
        XT = [[X[j][i] for j in range(n)] for i in range(degree + 1)]
        XTX = LinearAlgebra.matrix_multiply(XT, X)
        XTy = [sum(XT[i][j] * y[j] for j in range(n)) for i in range(degree + 1)]
        
        coefficients = LinearAlgebra.gauss_jordan_elimination(XTX, XTy)
        coefficients.reverse()  # Lowest to highest degree
        
        # Calculate R-squared
        y_mean = sum(y) / n
        ss_tot = sum((yi - y_mean)**2 for yi in y)
        ss_res = sum((y[i] - sum(coefficients[j] * x[i]**j for j in range(degree + 1)))**2 
                     for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return coefficients, r_squared
    
    @staticmethod
    def exponential_fit(x: List[float], y: List[float]) -> Tuple[float, float, float]:
        """
        Fit exponential curve y = a * exp(b * x).
        
        Parameters:
        x: Independent variable values
        y: Dependent variable values (must be positive)
        
        Returns:
        Tuple of (a, b, r_squared)
        """
        # Linearize: ln(y) = ln(a) + b*x
        y_log = [math.log(yi) if yi > 0 else 0 for yi in y]
        b, ln_a, r_squared = CurveFitting.linear_regression(x, y_log)
        a = math.exp(ln_a)
        
        return a, b, r_squared
    
    @staticmethod
    def power_law_fit(x: List[float], y: List[float]) -> Tuple[float, float, float]:
        """
        Fit power law curve y = a * x^b.
        
        Parameters:
        x: Independent variable values (must be positive)
        y: Dependent variable values (must be positive)
        
        Returns:
        Tuple of (a, b, r_squared)
        """
        # Linearize: ln(y) = ln(a) + b*ln(x)
        x_log = [math.log(xi) if xi > 0 else 0 for xi in x]
        y_log = [math.log(yi) if yi > 0 else 0 for yi in y]
        b, ln_a, r_squared = CurveFitting.linear_regression(x_log, y_log)
        a = math.exp(ln_a)
        
        return a, b, r_squared


class Statistics:
    """
    Class for statistical calculations
    """
    
    @staticmethod
    def mean(data: List[float]) -> float:
        """Calculate mean."""
        return sum(data) / len(data)
    
    @staticmethod
    def variance(data: List[float], sample: bool = True) -> float:
        """Calculate variance."""
        mean_val = Statistics.mean(data)
        n = len(data) - 1 if sample else len(data)
        return sum((x - mean_val)**2 for x in data) / n
    
    @staticmethod
    def standard_deviation(data: List[float], sample: bool = True) -> float:
        """Calculate standard deviation."""
        return math.sqrt(Statistics.variance(data, sample))
    
    @staticmethod
    def correlation_coefficient(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        mean_x = Statistics.mean(x)
        mean_y = Statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        sum_sq_x = sum((x[i] - mean_x)**2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y)**2 for i in range(n))
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        return numerator / denominator if denominator > 0 else 0.0
