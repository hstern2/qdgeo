#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "cgmin.h"

#define MAX_LINESEARCH 20

/***
 * Find the minimum of the cubic passing through
 * (a, fa) with slope dfa and (b, fb) with slope dfb, 
 * where a < b and either fa < fb and dfa < 0 or fb < fa and dfb > 0.
 * If things go wrong, give up and return (a+b)/2.
 ***/
static double interpolate(const double a, const double fa, const double dfa,
			  const double b, const double fb, const double dfb)
{
  const double a2 = a*a, b2 = b*b;
  const double a3 = a2*a, b3 = b2*b;
  const double a_b = a - b;
  const double a_b3 = a_b*a_b*a_b;
  double va, vb, vc, x, eps;
  assert(a < b);
  assert((fa <= fb && dfa < 0) || (fb <= fa && dfb > 0));
  if (fabs(a_b3) < 1e-10)
    return 0.5*(a+b);
  va = (a_b*(dfa+dfb) - 2*fa + 2*fb)/a_b3;
  vb = (-(a2*(dfa + 2*dfb)) + a*(-(b*dfa) + b*dfb + 3*fa - 3*fb) + 
	b*(2*b*dfa + b*dfb + 3*fa - 3*fb))/a_b3;
  vc = (-(b3*dfa) + a3*dfb + a2*b*(2*dfa + dfb) - 
	a*b*(b*dfa + 2*b*dfb + 6*fa - 6*fb))/a_b3;
  if (fabs(va*vc) > 1e-8*fabs(vb*vb)) {
    const double vtmp = vb*vb - 3*va*vc;
    assert(vtmp > 0);
    x = (-vb + sqrt(vtmp))/(3*va);
  } else {
    x = -vc/(2*vb);
  }
  eps = -1e-3*a_b;
  if (a + eps < x && x < b - eps)
    return x;
  else
    return 0.5*(a+b);
}

/***
 * Find the minimum of the parabola passing through 
 * (a,fa) and (b,fb) with slope dfb, where a < b and dfb < 0
 * If no minimum exists, extrapolate by returning 3b - 2a.
 ***/
static double extrapolate(const double a, const double fa,
			  const double b, const double fb, const double dfb)
{
  const double tmp = (b-a)*dfb + fa - fb;
  double x;
  assert(a < b);
  assert(dfb < 0);
  if (tmp > 1e-10)
    x = -(a*a*dfb - b*(b*dfb + 2*fa - 2*fb))/(2*tmp);
  else
    x = 3*b - 2*a;
  assert(x > b);
  return x;
}

static double dot(int n, const double *x, const double *y)
{
  int i;
  double d = 0;
  for (i = 0; i < n; i++)
    d += x[i]*y[i];
  return d;
}

static double rms(int n, const double *x)
{
  return n > 0 ? sqrt(dot(n,x,x)/n) : 0.0;
}

int conjugate_gradient_minimize(int n, double *x, double *r, double *s,
				double tolerance, double linesearch_tolerance,
				int maxeval, int verbose,
				double (*calc_fr)(int n, const double *x, 
						  double *r, void *user),
				void (*calc_s)(int n, const double *x, 
					       const double *r, double *s, void *user),
				void *user, double *work)
{
  int first_time = 1;
  int niter, neval; /* number of line-search iterations, CG iterations, function evaluations */
  const int size = n*sizeof(double);
  double *d = work; /* search direction */
  double *d0 = work+n; /* last search direction */
  double *s0 = work+2*n; /* last preconditioned residual */
  double *x0 = work+3*n; /* point of departure for line search */
  double rs=0, r0s0=0, rd=0, beta=0;
  double a=0, b=0, c=0, t=1.0; /* points in line search, t is first initial guess */
  double fa=0, fb=0, fc=0, ft=0;
  double f0=0, dfa=0, dfb=0, dfc=0, dft=0, df0=0, lo=0, flo=0;
  int i, bdefined, cdefined, iline;
  if (verbose) {
    printf("Starting conjugate gradient minimization...\n");
    printf("Preconditioning: %s\n", s ? "yes" : "no");
    fflush(stdout);
  }
  ft = calc_fr(n,x,r,user);
  if (verbose) {
    printf("Initial function value: %g\n", ft);
    printf("Initial RMS gradient: %g\n", rms(n,r));
  }
  neval = 1;
  niter = 0;
  while (neval < maxeval && rms(n,r) > tolerance) {
    /* Find a search direction */
    if (calc_s)
      calc_s(n,x,r,s,user);
    else
      s = r;
    rs = dot(n,r,s);
    switch (first_time) {
    case 0:
      beta = (rs - dot(n,r,s0))/r0s0;
      for (i = 0; i < n; i++)
	d[i] = s[i] + beta*d0[i];
      if ((rd = dot(n,r,d)) > 0)
	break; /* a good search direction */
    case 1: /* first time, or bad search direction */
      memcpy(d,s,size);
      rd = dot(n,r,d);
      t = 1.0;
    }
    assert(rd > 0); /* This will fail if the preconditioner is not positive-definite */
    first_time = 0;
    memcpy(d0,d,size);
    memcpy(s0,s,size);
    r0s0 = rs;
    /* Do a line search */
    a = lo = 0;
    f0 = fa = flo = ft;
    df0 = dfa = -rd;
    bdefined = cdefined = 0;
    memcpy(x0,x,size);
    iline = 0;
    for (;;) {
      for (i = 0; i < n; i++)
	x[i] = x0[i] + t*d[i];
      ft = calc_fr(n,x,r,user);
      dft = -dot(n,r,d);
      if (ft < flo) {
	flo = ft;
	lo = t;
      }
      ++neval;
      if (verbose > 1) {
	printf("%d a: %14.8e %14.8e %14.8e  ", niter, a,fa,dfa);
	if (bdefined)
	  printf("b: %14.8e %14.8e %14.8e  ", b,fb,dfb);
	if (cdefined)
	  printf("c: %14.8e %14.8e %14.8e  ", c,fc,dfc);
	printf("t: %14.8e %14.8e %14.8e\n", t,ft,dft);
	fflush(stdout);
      }
      if ((ft < f0 && fabs(dft) < linesearch_tolerance * fabs(df0)))
	break; /* converged */
      if (iline++ > MAX_LINESEARCH || neval >= maxeval) {
	/* Got into trouble - restore lowest point */
	for (i = 0; i < n; i++)
	  x[i] = x0[i] + lo*d[i];
	ft = calc_fr(n,x,r,user);
	first_time = 1;
	break;
      }
      if (bdefined) {
	if (t < b) {
	  if (ft < fb) { 
	    c = b; fc = fb; dfc = dfb; b = t; fb = ft; dfb = dft; bdefined = cdefined = 1; 
	  } else { 
	    a = t; fa = ft; dfa = dft; 
	  }
	} else {
	  if (ft < fb) { 
	    a = b; fa = fb; dfa = dfb; b = t; fb = ft; dfb = dft; bdefined = 1; 
	  } else {
	    c = t; fc = ft; dfc = dft; cdefined = 1;
	  }
	}
      } else {
	if (ft < fa) { 
	  b = t; fb = ft; dfb = dft; bdefined = 1; 
	} else {
	  c = t; fc = ft; dfc = dft; cdefined = 1; 
	}
      }
      assert(!bdefined || (a <= b && fb <= fa));
      assert(!cdefined || a <= c);
      assert(!bdefined || !cdefined || (b <= c && fb <= fc));
      if (bdefined)
	if (dfb < 0)
	  if (cdefined)
	    t = interpolate(b, fb, dfb, c, fc, dfc);
	  else
	    t = extrapolate(a, fa, b, fb, dfb);
	else
	  t = interpolate(a, fa, dfa, b, fb, dfb);
      else 
	t = interpolate(a, fa, dfa, c, fc, dfc);
    }
    niter++;
  }
  if (verbose) {
    const double rmsg = rms(n,r);
    printf("Final function value: %g\n", ft);
    printf("Final RMS gradient: %g\n", rmsg);
    printf(rmsg < tolerance ? "Converged " : "*** Did not converge ");
    printf("to tolerance of %g with %d iterations and %d function evaluations\n", 
	   tolerance, niter, neval);
    if (neval > 0)
      printf("(averaged %.2f evaluations per line search, linesearch_tolerance = %g)\n", 
	     niter > 0 ? (double) neval/(double) niter : 0, linesearch_tolerance);
    fflush(stdout);
  }
  return neval <= maxeval;
}

#ifdef SIMPLE_EXAMPLE

static double calc_fr(int n, const double *x, double *r, void *user)
{
  const double *x0 = (const double *) user;
  int i;
  double u = 0;
  for (i = 0; i < n; i++) {
    u += (x[i]-x0[i])*(x[i]-x0[i]);
    r[i] = -2*(x[i]-x0[i]); 
  }
  return u;
}

int main()
{
  int i;
  const int n = 3;
  const double tolerance = 1e-6;
  const double linesearch_tolerance = 0.5;
  const int maxeval = 1000;
  const int verbose = 1;
  double x[3] = {0.0, 0.0, 0.0}; /* initial guess */
  const double x0[3] = {-3.45, 0.677, -0.25};  /* 'user data' for function to minimize */
  double r[3]; /* workspace to hold residual (negative gradient) */
  double work[12];
  
  printf("Initial parameter values: ");
  for (i = 0; i < n; i++)
    printf("%f ", x[i]);
  printf("\n");

  int converged = conjugate_gradient_minimize(n, x, r, 0, tolerance, linesearch_tolerance, 
					      maxeval, verbose, calc_fr, 0, (void *) x0,
					      work);

  printf("Converged: %s\n", converged ? "yes" : "no");

  printf("Final parameter values: ");
  for (i = 0; i < n; i++)
    printf("%f ", x[i]);
  printf("\n");

  return 0;
}

#endif /* SIMPLE_EXAMPLE */
