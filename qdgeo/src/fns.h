#ifndef FNS_H
#define FNS_H

#include <math.h>

#define UNDEF_VAL -3.1415e15

#ifdef __cplusplus
extern "C" {
#endif

  inline static double sq(double x) { return x*x; }
  inline static double cube(double x) { return x*x*x; }
  inline static double round_to_nearest_integer(double x) { return floor(x+0.5); }
  inline static double fractional_part(double x) { return x - round_to_nearest_integer(x); }
  inline static double periodic(double x, double l) { return x - l*round_to_nearest_integer(x/l); }
  inline static double ex_x(double x) { return exp(-x)/x; }
  inline static double ex2_x2(double x) { return ex_x(x*x); }

  double machine_epsilon();
  double small_val();
  int is_almost_zero(double x);
  double zero_if_almost_zero(double x);
  int are_approximately_equal(double x, double y);
  int is_integer(double x);
  int is_perfect_square(double x);
  int is_perfect_cube(double x);
  double relative_error(double x, double y);

  double factorial(int n);
  double double_factorial(int n);
  int prime_near(int n);
  double combinations(int n, int k);

  /* Given three points (x,y), returns the slope at the middle point */
  double three_point_slope(double x1, double y1, double x2, double y2, double x3, double y3);

  /* Complex error function: u+iv = erf(x+iy) */
  void erfz(double x, double y, double *u, double *v);

  /* Integrate by Gaussian quadrature */
  double integrate(double (*f)(double x, const void *user), double a, double b, const void *user);

  /* Find maximum between a and b -- assumes smooth on scale (b-a)/100 */
  void find_maximum(double (*f)(double x, const void *user), double a, double b, const void *user,
		    double *xmax, double *fmax);
  
  inline static int mymod(int n, int m)
  {
    int k;
    if (n >= 0)
      return n % m;
    k = -n % m;
    if (k > 0)
      return m - k;
    else
      return 0;
  }

  int is_not_a_number(double x);

#ifdef __cplusplus
}
#endif

#endif /* FNS_H */
