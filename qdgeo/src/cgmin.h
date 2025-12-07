#ifndef CGMIN_H
#define CGMIN_H

#ifdef __cplusplus
extern "C" {
#endif
  
  /***
   * Conjugate gradient minimization with the Polak-Riviere algorithm.
   *
   * int n: number of parameters
   * double *x: initial guess for parameters; on exit, final parameter values
   * double *r: residual (negative gradient) of function to be minimized
   * double *s: preconditioned residual of function to be minimized
   *            (for instance, s = H^-1 r, where H is an approximate Hessian)
   *            The preconditioner must be positive definite.
   *            Ignored if calc_s is null.
   * double tolerance: terminate when root-mean-square gradient falls below this value
   * double linesearch_tolerance: terminate each linesearch when gradient
   *                              has decreased by this factor (suggested value: 0.5)
   * int maxeval: terminate when the function has been calculated this many times
   * verbose: how much to print.  0 prints nothing, higher values print more.
   * double (*calc_fr)(): pointer to the function to be minimized.
   *                      The function value should be returned for the current
   *                      values of the parameters in x and the residual 
   *                      (negative gradient) should be written to r.
   * void (*calc_s)(): pointer to procedure to calculate the preconditioned residual, 
   *                   which should be written to s.  calc_s is only called after calc_fr 
   *                   has been called for the same parameter values.
   *                   If null, no preconditioning is done.
   * void *user: a pointer to user data which is forwarded to calls to calc_fr and calc_s
   *             but otherwise ignored.
   * double *work: workspace, size 4*n
   *
   * Return value is 1 if the minimization converged
   * (i.e., the RMS gradient fell below tolerance)
   * and 0 if the maximum number of function evaluations were exceeded.
   *
   ***/
  int conjugate_gradient_minimize(int n, double *x, double *r, double *s,
				  double tolerance, double linesearch_tolerance,
				  int maxeval, int verbose,
				  double (*calc_fr)(int n, const double *x,
						    double *r, void *user),
				  void (*calc_s)(int n, const double *x,
						 const double *r, double *s, void *user),
				  void *user, double *work);
  
#ifdef __cplusplus
}
#endif

#endif /* CGMIN_H */
