#ifndef LBFGS_HPP
#define LBFGS_HPP

#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

/**
 * L-BFGS (Limited-memory BFGS) optimizer
 * 
 * This is a quasi-Newton method that approximates the inverse Hessian using
 * the last m gradient differences. It typically converges much faster than
 * conjugate gradient methods for smooth optimization problems.
 * 
 * Reference: Nocedal & Wright, "Numerical Optimization", Chapter 7
 */

namespace lbfgs {

// Dot product
inline double dot(int n, const double* x, const double* y) {
    double d = 0.0;
    for (int i = 0; i < n; i++)
        d += x[i] * y[i];
    return d;
}

// RMS of vector
inline double rms(int n, const double* x) {
    return n > 0 ? std::sqrt(dot(n, x, x) / n) : 0.0;
}

// x = x + alpha * y
inline void axpy(int n, double alpha, const double* y, double* x) {
    for (int i = 0; i < n; i++)
        x[i] += alpha * y[i];
}

// y = x
inline void copy(int n, const double* x, double* y) {
    std::memcpy(y, x, n * sizeof(double));
}

// y = alpha * x
inline void scale_copy(int n, double alpha, const double* x, double* y) {
    for (int i = 0; i < n; i++)
        y[i] = alpha * x[i];
}

/**
 * Strong Wolfe line search
 * 
 * Finds step length alpha satisfying strong Wolfe conditions:
 *   f(x + alpha*d) <= f(x) + c1 * alpha * grad_f(x)^T * d  (sufficient decrease)
 *   |grad_f(x + alpha*d)^T * d| <= c2 * |grad_f(x)^T * d|  (curvature condition)
 * 
 * Returns the step length, or 0 if line search fails
 */
inline double line_search(
    int n, 
    double* x,           // current point (modified on success)
    double f,            // current function value
    const double* g,     // current gradient
    const double* d,     // search direction (descent: g^T * d < 0)
    double* x_new,       // workspace for trial point
    double* g_new,       // workspace for trial gradient
    double& f_new,       // output: new function value
    double (*calc_fg)(int n, const double* x, double* g, void* user),
    void* user,
    int& neval,
    int maxeval,
    double c1 = 1e-4,    // sufficient decrease parameter
    double c2 = 0.9      // curvature parameter (0.9 is good for L-BFGS)
) {
    const double dg0 = dot(n, d, g);  // directional derivative at alpha=0
    if (dg0 >= 0) return 0.0;         // not a descent direction
    
    const double f0 = f;
    const int max_iter = 20;
    
    double alpha = 1.0;       // initial step (1.0 is natural for quasi-Newton)
    double alpha_lo = 0.0;
    double alpha_hi = 1e10;
    double f_lo = f0;
    
    for (int iter = 0; iter < max_iter && neval < maxeval; iter++) {
        // Evaluate at trial point
        for (int i = 0; i < n; i++)
            x_new[i] = x[i] + alpha * d[i];
        f_new = calc_fg(n, x_new, g_new, user);
        neval++;
        
        double dg = dot(n, d, g_new);
        
        // Check Armijo (sufficient decrease) condition
        if (f_new > f0 + c1 * alpha * dg0) {
            // Decrease step size
            alpha_hi = alpha;
            alpha = 0.5 * (alpha_lo + alpha_hi);
        }
        // Check curvature condition  
        else if (std::fabs(dg) > c2 * std::fabs(dg0)) {
            // Need to refine
            if (dg < 0) {
                // Still descending, increase step
                alpha_lo = alpha;
                f_lo = f_new;
                if (alpha_hi > 1e9) {
                    alpha = 2.0 * alpha;
                } else {
                    alpha = 0.5 * (alpha_lo + alpha_hi);
                }
            } else {
                // Overshot, decrease step
                alpha_hi = alpha;
                alpha = 0.5 * (alpha_lo + alpha_hi);
            }
        }
        else {
            // Both conditions satisfied
            copy(n, x_new, x);
            return alpha;
        }
        
        // Check for convergence of interval
        if (alpha_hi - alpha_lo < 1e-12 * std::max(1.0, alpha_lo)) {
            // Accept best point found
            if (f_new < f_lo) {
                copy(n, x_new, x);
                return alpha;
            } else {
                // Restore to lower bound
                for (int i = 0; i < n; i++)
                    x[i] = x[i] + alpha_lo * d[i];
                f_new = calc_fg(n, x, g_new, user);
                neval++;
                return alpha_lo;
            }
        }
    }
    
    // Line search didn't converge, but accept if we made progress
    if (f_new < f0) {
        copy(n, x_new, x);
        return alpha;
    }
    
    return 0.0;  // failed
}

/**
 * L-BFGS two-loop recursion to compute search direction
 * 
 * Computes H_k * g where H_k is the L-BFGS approximation to the inverse Hessian
 */
inline void compute_direction(
    int n,
    const double* g,        // current gradient
    double* d,              // output: search direction
    const std::vector<std::vector<double>>& s,  // position differences
    const std::vector<std::vector<double>>& y,  // gradient differences
    const std::vector<double>& rho,             // 1 / (y^T s)
    int m,                  // history size
    int k,                  // current iteration
    double* alpha_tmp       // workspace of size m
) {
    // q = g
    copy(n, g, d);
    
    int bound = std::min(k, m);
    
    // First loop (backward)
    for (int i = bound - 1; i >= 0; i--) {
        int idx = (k - 1 - (bound - 1 - i)) % m;
        alpha_tmp[i] = rho[idx] * dot(n, s[idx].data(), d);
        axpy(n, -alpha_tmp[i], y[idx].data(), d);
    }
    
    // Scale by initial Hessian approximation: H0 = gamma * I
    // gamma = s^T y / y^T y (from most recent pair)
    if (bound > 0) {
        int idx = (k - 1) % m;
        double yy = dot(n, y[idx].data(), y[idx].data());
        double sy = dot(n, s[idx].data(), y[idx].data());
        double gamma = (yy > 1e-15) ? sy / yy : 1.0;
        for (int i = 0; i < n; i++)
            d[i] *= gamma;
    }
    
    // Second loop (forward)
    for (int i = 0; i < bound; i++) {
        int idx = (k - bound + i) % m;
        double beta = rho[idx] * dot(n, y[idx].data(), d);
        axpy(n, alpha_tmp[i] - beta, s[idx].data(), d);
    }
    
    // Negate to get descent direction
    for (int i = 0; i < n; i++)
        d[i] = -d[i];
}

/**
 * L-BFGS minimization
 * 
 * @param n           Number of variables
 * @param x           Initial guess; on exit, the solution
 * @param tolerance   Convergence tolerance on RMS gradient
 * @param maxeval     Maximum function evaluations
 * @param verbose     Verbosity level (0=silent, 1=summary, 2=every iteration)
 * @param calc_fg     Function to compute f and gradient (gradient stored with negative sign)
 * @param user        User data passed to calc_fg
 * @param m           History size (default 10, typically 5-20)
 * 
 * @return 1 if converged, 0 otherwise
 * 
 * Note: calc_fg should compute f and store the NEGATIVE gradient in g
 *       (i.e., g[i] = -df/dx[i]), following the convention of the original cgmin.
 */
inline int minimize(
    int n,
    double* x,
    double tolerance,
    int maxeval,
    int verbose,
    double (*calc_fg)(int n, const double* x, double* g, void* user),
    void* user,
    int m = 10
) {
    if (n <= 0 || maxeval <= 0) return 0;
    
    // Allocate storage
    std::vector<double> g(n);      // gradient (negative, as computed by calc_fg)
    std::vector<double> g_new(n);  // new gradient
    std::vector<double> d(n);      // search direction
    std::vector<double> x_new(n);  // trial point
    std::vector<double> alpha_tmp(m);  // workspace for two-loop recursion
    
    // L-BFGS history
    std::vector<std::vector<double>> s(m, std::vector<double>(n));  // x_{k+1} - x_k
    std::vector<std::vector<double>> y(m, std::vector<double>(n));  // g_{k+1} - g_k
    std::vector<double> rho(m);  // 1 / (y^T s)
    
    // Initial evaluation
    // Note: calc_fg returns negative gradient, so we work with negated values
    double f = calc_fg(n, x, g.data(), user);
    int neval = 1;
    
    // Convert to actual gradient (negate the negative gradient from calc_fg)
    for (int i = 0; i < n; i++) g[i] = -g[i];
    
    double gnorm = rms(n, g.data());
    
    if (verbose) {
        std::printf("Starting L-BFGS minimization (history size = %d)...\n", m);
        std::printf("Initial function value: %g\n", f);
        std::printf("Initial RMS gradient: %g\n", gnorm);
        std::fflush(stdout);
    }
    
    int k = 0;  // iteration counter
    
    while (gnorm > tolerance && neval < maxeval) {
        // Compute search direction using L-BFGS two-loop recursion
        if (k == 0) {
            // First iteration: steepest descent
            scale_copy(n, -1.0, g.data(), d.data());
        } else {
            compute_direction(n, g.data(), d.data(), s, y, rho, m, k, alpha_tmp.data());
        }
        
        // Line search
        double f_new;
        // Need to create wrapper that handles sign convention
        auto calc_fg_positive = [&](int nn, const double* xx, double* gg, void* uu) -> double {
            double fval = calc_fg(nn, xx, gg, uu);
            for (int i = 0; i < nn; i++) gg[i] = -gg[i];  // convert to positive gradient
            return fval;
        };
        
        // Store old x and g for history update
        std::vector<double> x_old(n), g_old(n);
        copy(n, x, x_old.data());
        copy(n, g.data(), g_old.data());
        
        // Perform line search
        double alpha = 0.0;
        const double dg0 = dot(n, d.data(), g.data());
        
        if (dg0 < 0) {  // valid descent direction
            const double f0 = f;
            const int max_ls = 20;
            double alpha_lo = 0.0, alpha_hi = 1e10;
            alpha = 1.0;
            
            for (int ls_iter = 0; ls_iter < max_ls && neval < maxeval; ls_iter++) {
                for (int i = 0; i < n; i++)
                    x_new[i] = x_old[i] + alpha * d[i];
                f_new = calc_fg(n, x_new.data(), g_new.data(), user);
                neval++;
                for (int i = 0; i < n; i++) g_new[i] = -g_new[i];
                
                double dg = dot(n, d.data(), g_new.data());
                
                // Armijo condition
                if (f_new > f0 + 1e-4 * alpha * dg0) {
                    alpha_hi = alpha;
                    alpha = 0.5 * (alpha_lo + alpha_hi);
                }
                // Curvature condition
                else if (std::fabs(dg) > 0.9 * std::fabs(dg0)) {
                    if (dg < 0) {
                        alpha_lo = alpha;
                        alpha = (alpha_hi > 1e9) ? 2.0 * alpha : 0.5 * (alpha_lo + alpha_hi);
                    } else {
                        alpha_hi = alpha;
                        alpha = 0.5 * (alpha_lo + alpha_hi);
                    }
                }
                else {
                    // Success
                    copy(n, x_new.data(), x);
                    copy(n, g_new.data(), g.data());
                    f = f_new;
                    break;
                }
                
                if (alpha_hi - alpha_lo < 1e-12) {
                    if (f_new < f) {
                        copy(n, x_new.data(), x);
                        copy(n, g_new.data(), g.data());
                        f = f_new;
                    }
                    break;
                }
            }
        }
        
        // Check if we made progress
        if (f >= f - 1e-15 * std::fabs(f) && alpha <= 1e-12) {
            // No progress, try steepest descent
            if (k > 0) {
                k = 0;  // reset L-BFGS history
                continue;
            } else {
                break;  // even steepest descent failed
            }
        }
        
        // Update L-BFGS history
        int idx = k % m;
        for (int i = 0; i < n; i++) {
            s[idx][i] = x[i] - x_old[i];
            y[idx][i] = g[i] - g_old[i];
        }
        double ys = dot(n, y[idx].data(), s[idx].data());
        if (ys > 1e-15) {
            rho[idx] = 1.0 / ys;
        } else {
            rho[idx] = 1e15;  // skip this update effectively
        }
        
        gnorm = rms(n, g.data());
        k++;
        
        if (verbose > 1) {
            std::printf("Iter %4d: f = %14.8e, |g| = %14.8e, alpha = %.6e\n", 
                       k, f, gnorm, alpha);
            std::fflush(stdout);
        }
    }
    
    if (verbose) {
        std::printf("Final function value: %g\n", f);
        std::printf("Final RMS gradient: %g\n", gnorm);
        std::printf(gnorm <= tolerance ? "Converged " : "*** Did not converge ");
        std::printf("to tolerance of %g with %d iterations and %d function evaluations\n",
                   tolerance, k, neval);
        std::fflush(stdout);
    }
    
    return gnorm <= tolerance ? 1 : 0;
}

}  // namespace lbfgs

#endif  // LBFGS_HPP
