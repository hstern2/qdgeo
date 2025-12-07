#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "fns.h"

int is_not_a_number(double x) { return isnan(x); }

double machine_epsilon()
{
  static double mach_eps = 0;
  static int first_time = 1;
  if (first_time) {
    mach_eps = 1;
    while (mach_eps+1 > 1)
      mach_eps /= 2;
    mach_eps *= 2;
    first_time = 0;
  }
  return mach_eps;
}

double small_val()
{
  return 64*machine_epsilon();
}

double relative_error(double x, double y) { return fabs(x-y)/(fabs(y)+small_val()); }

int is_almost_zero(double x) { return fabs(x) < small_val(); }
double zero_if_almost_zero(double x) { return is_almost_zero(x) ? 0 : x; }
int are_approximately_equal(double x, double y) { return is_almost_zero(x-y); }
int is_integer(double x) { return are_approximately_equal(x, round_to_nearest_integer(x)); }
int is_perfect_square(double x) { return is_integer(sqrt(x)); }
int is_perfect_cube(double x) { return is_integer(cbrt(x)); }

double factorial(int n)
{
  double m;
  const double f[10] = {1,1,2,6,24,120,720,5040,40320,362880};
  assert(n >= 0);
  if (n < 10)
    return f[n];
  for (m = 1; n > 1; m *= n--)
    ;
  return m;
}

int prime_near(int n)
{
  const int p[11] = {29,101,601,1291,2053,2819,3643,4493,5387,6221,7103};
  int i;
  for (i = 0; i < 10; i++)
    if (p[i] >= n)
      return p[i];
  return p[10];
}

double three_point_slope(double x1, double y1, double x2, double y2, double x3, double y3)
{
  return (sq(x3)*(y1-y2) - 2*x2*(x3*(y1-y2)+x1*(y2-y3)) + sq(x2)*(y1-y3) + sq(x1)*(y2-y3)) /
    ((x1-x2)*(x1-x3)*(x2-x3));
}

double double_factorial(int n)
{
  double d = 1.0;
  while (n > 1) {
    d *= n;
    n -= 2;
  }
  return d;
}

double combinations(int n, int k)
{
  assert(n >= k && k >= 0);
  return factorial(n)/(factorial(k)*factorial(n-k));
}


static void expnz2(double x, double y, double *u, double *v)
{
  const double t = exp(y*y-x*x);
  const double t2 = -2*x*y;
  *u = t*cos(t2);
  *v = t*sin(t2);
}

static void wofz (double xi, double yi, double *u, double *v) 
{
  /*
    c  given a complex number z = (xi,yi), this subroutine computes
    c  the value of the faddeeva-function w(z) = exp(-z**2)*erfc(-i*z),
    c  where erfc is the complex complementary error-function and i
    c  means sqrt(-1).
    c  the accuracy of the algorithm for z in the 1st and 2nd quadrant
    c  is 14 significant digits; in the 3rd and 4th it is 13 significant
    c  digits outside a circular region with radius 0.126 around a zero
    c  of the function.
    c  all real variables in the program are double precision.
    c
    c  the code contains a few compiler-dependent parameters :
    c     rmaxreal = the maximum value of rmaxreal equals the root of
    c                rmax = the largest number which can still be
    c                implemented on the computer in double precision
    c                floating-point arithmetic
    c     rmaxexp  = ln(rmax) - ln(2)
    c     rmaxgoni = the largest possible argument of a double precision
    c                goniometric function (cos, sin, ...)
    c  the reason why these parameters are needed as they are defined will
    c  be explained in the code by means of comments
    c
    c
    c  parameter list
    c     xi     = real      part of z
    c     yi     = imaginary part of z
    c     u      = real      part of w(z)
    c     v      = imaginary part of w(z)
    c     flag   = an error flag indicating whether overflow will
    c              occur or not; type logical;
    c              the values of this variable have the following
    c              meaning :
    c              flag=.false. : no error condition
    c              flag=.true.  : overflow will occur, the routine
    c                             becomes inactive
    c  xi, yi      are the input-parameters
    c  u, v, flag  are the output-parameters
    c
    c  furthermore the parameter factor equals 2/sqrt(pi)
    c
    c  the routine is not underflow-protected but any variable can be
    c  put to 0 upon underflow;
    c
    c  reference - gpm poppe, cmj wijers; more efficient computation of
    c  the complex error-function, acm trans. math. software.
  */
  
  int a, b;
  const double factor = 1.12837916709551257388;
  /*
    const double rmaxreal = 0.5e154;
    const double rmaxexp  = 708.503061461606;
    const double rmaxgoni = 3.53711887601422e15;
  */
  double xabs, yabs, x, y, qrho, xabsq, xquad, yquad, xsum, ysum, xaux, daux;
  double u1, v1, u2=0, v2=0, h, h2=0, rx, ry, sx, sy, tx, ty, c, qlambda=0, w1;
  int n, j, i, kapn, nu, np1;
  xabs = fabs(xi);
  yabs = fabs(yi);
  x    = xabs/6.3;
  y    = yabs/4.4;

  /* if ((xabs > rmaxreal) || (yabs > rmaxreal)) return; */

  qrho = x*x + y*y;
  xabsq = xabs*xabs;
  xquad = xabsq - yabs*yabs;
  yquad = 2*xabs*yabs;
  a     = qrho<0.085264;
  if (a) { /* here */
    qrho  = (1-0.85*y)*sqrt(qrho);
    n     = (int) ceil(6 + 72*qrho);
    j     = 2*n+1;
    xsum  = 1.0/j;
    ysum  = 0.0;
    for (i = n; i >= 1; i--) {
      j    = j - 2;
      xaux = (xsum*xquad - ysum*yquad)/i;
      ysum = (xsum*yquad + ysum*xquad)/i;
      xsum = xaux + 1.0/j;
    }
    u1   = -factor*(xsum*yabs + ysum*xabs) + 1.0;
    v1   =  factor*(xsum*xabs - ysum*yabs);
    daux =  exp(-xquad);
    u2   =  daux*cos(yquad);
    v2   = -daux*sin(yquad);
    *u    = u1*u2 - v1*v2;
    *v    = u1*v2 + v1*u2;
  } else {
    if (qrho > 1.0) {
      h    = 0.0;
      kapn = 0;
      qrho = sqrt(qrho);
      nu   = (int) ceil(3 + (1442/(26*qrho+77)));
    } else {
      qrho = (1-y)*sqrt(1-qrho);
      h    = 1.88*qrho;
      h2   = 2*h;
      kapn = (int) ceil(7  + 34*qrho);
      nu   = (int) ceil(16 + 26*qrho);
    }
    b = (h > 0.0);
    if (b) qlambda = pow(h2,kapn);
    rx = 0.0;
    ry = 0.0;
    sx = 0.0;
    sy = 0.0;
    for (n = nu; n >= 0; n--) {
      np1 = n + 1;
      tx  = yabs + h + np1*rx;
      ty  = xabs - np1*ry;
      c   = 0.5/(tx*tx + ty*ty);
      rx  = c*tx;
      ry  = c*ty;
      if (b && (n <= kapn)) {
	tx = qlambda + sx;
	sx = rx*tx - ry*sy;
	sy = ry*tx + rx*sy;
	qlambda = qlambda/h2;
      }
    }
    if (fabs(h) < 1e-10) {
      *u = factor*rx;
      *v = factor*ry;
    } else {
      *u = factor*sx;
      *v = factor*sy;
    }
    if (fabs(yabs) < 1e-10) *u = exp(-xabs*xabs);
  }
  if (yi<0.0) {
    if (a) {
      u2    = 2*u2;
      v2    = 2*v2;
    } else {
      xquad =  -xquad;
      /* if ((yquad > rmaxgoni) || (xquad > rmaxexp)) return; */
      w1 =  2*exp(xquad);
      u2  =  w1*cos(yquad);
      v2  = -w1*sin(yquad);
    }
    *u = u2 - *u;
    *v = v2 - *v;
    if (xi > 0.0) 
      *v = -*v;
  } else {
    if (xi<0.0) 
      *v = -*v;
  }
}

void erfz(double x, double y, double *u, double *v)
{
  double wx, wy, ex, ey, ewx, ewy;
  wofz(-y,x,&wx,&wy);
  expnz2(x,y,&ex,&ey);
  ewx = ex*wx - ey*wy;
  ewy = ex*wy + ey*wx;
  *u = 1 - ewx;
  *v = -ewy;
}

static void gauleg(double x[], double w[], int m)
{
  int i;
  double z1,z,pp,p3,p2,p1;
  const double eps = 4*machine_epsilon();
  const int n = 2*m;
  for (i=0;i<m;i++) {
    z=cos(M_PI*(i+0.75)/(n+0.5));
    do {
      int j;
      p1=1.0;
      p2=0.0;
      for (j=1;j<=n;j++) {
	p3=p2;
	p2=p1;
	p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
      }
      pp=n*(z*p1-p2)/(z*z-1.0);
      z1=z;
      z=z1-p1/pp;
    } while (fabs(z-z1) > eps);
    x[i]=z;
    w[i]=2.0/((1.0-z*z)*pp*pp);
  }
}

#define NQUAD 128

double integrate(double (*f)(double x, const void *user), double a, double b, const void *user)
{
  const double xm = 0.5*(b+a);
  const double xl = 0.5*(b-a);
  static int first_time = 1;
  static double x[NQUAD], w[NQUAD];
  double s = 0;
  int i;
  if (first_time) {
    gauleg(x,w,NQUAD);
    first_time = 0;
  }
  for (i = 0; i < NQUAD; i++) {
    const double dx = xl*x[i];
    s += w[i]*(f(xm+dx,user) + f(xm-dx,user));
  }
  return s*xl;
}

void find_maximum(double (*f)(double x, const void *user), double a, double b, const void *user,
		  double *xmax, double *fmax)
{
  const double eps = 64*small_val();
  const int n = 100;
  int i,ibest = 0;
  double best,r,s,t,fr,fs,ft,dx;  
  if (b < a) {
    r = a;
    a = b;
    b = r;
  }
  dx = (b-a)/n;
  best = f(a,user);
  for (i = 1; i <= n; i++) {
    const double y = f(i*dx+a,user);
    if (y >= best) {
      best = y;
      ibest = i;
    }
  }
  if (ibest == 0 || ibest == n) {
    *xmax = ibest*dx + a;
    *fmax = best;
    return;
  }
  *xmax = s = ibest*dx + a;
  r = s - dx;
  t = s + dx;
  fr = f(r,user);
  *fmax = fs = f(s,user);
  ft = f(t,user);
  for (;;) {
    double fs_ft = fs-ft;
    double ft_fr = ft-fr;
    double fr_fs = fr-fs;
    double tmp = 2*((fs-ft)*r + (ft-fr)*s + (fr-fs)*t);
    if (fabs(tmp) < eps)
      return;
    assert(r <= s);
    assert(s <= t);
    assert(fs >= fr);
    assert(fs >= ft);
    *xmax = fabs(tmp) > 0 ? (fs_ft*sq(r) + ft_fr*sq(s) + fr_fs*sq(t))/tmp : s;
    *fmax = f(*xmax,user);
    if (relative_error(s,*xmax) < eps || 
	relative_error(fs,*fmax) < eps)
      return;
    assert(*fmax >= fs);
    if (*xmax < s) {
      t = s;
      ft = fs;
      s = *xmax;
      fs = *fmax;
    } else {
      r = s;
      fr = fs;
      s = *xmax;
      fs = *fmax;
    }
  }
}

