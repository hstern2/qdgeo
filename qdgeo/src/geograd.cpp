#include "geograd.hpp"
#include "coord.hpp"
#include "fns.h"

double Angle(const Cartesian &a, const Cartesian &b, const Cartesian &c)
{ 
  double t = (a-b).as_unit_vector() * (c-b).as_unit_vector();
  if (t > 1)
    t = 1;
  else if (t < -1)
    t = -1;
  return acos(t);
}

double Dihedral(const Cartesian &a, const Cartesian &b, 
		const Cartesian &c, const Cartesian &d)
{
  const Cartesian ab = a - b;
  const Cartesian bc = b - c;
  const Cartesian cd = c - d;
  const Cartesian abcd = ab.cross(cd);
  const Cartesian abbc = ab.cross(bc).as_unit_vector();
  const Cartesian bccd = bc.cross(cd).as_unit_vector();
  double t = abbc*bccd;
  if (t > 1)
    t = 1;
  else if (t < -1)
    t = -1;
  const double u = acos(t);
  if (abcd*bc > 0)
    return u;
  else
    return -u;
}

double NormalDistance(const Cartesian &r0, const Cartesian &r1, 
		      const Cartesian &r2, const Cartesian &r3)
{
  const Cartesian u = (r2-r1).cross(r3-r1);
  const double umag = u.magnitude();
  if (umag > 0)
    return (u*(r0-r1))/umag;
  else
    return 0;
}

Cartesian ZLocation(const Cartesian &a, const Cartesian &b, const Cartesian &c,
		    double r, double theta, double phi)
{
  Tensor j;
  j.col(1) = (c-a).cross(b-a).as_unit_vector();
  j.col(0) = (b-a).cross(j.col(1)).as_unit_vector();
  j.col(2) = j.col(0).cross(j.col(1)).as_unit_vector();
  return a + j * Cartesian(r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), -r*cos(theta));
}

static double theta(double ct, double &dt_dct)
{
  if (ct > 1)
    ct = 1;
  else if (ct < -1)
    ct = -1;
  const double s2t = 1 - ct*ct;
  if (s2t < small_val()) {
    dt_dct = 0;
    return ct > 0 ? 0 : M_PI;
  } else {
    dt_dct = -1/sqrt(s2t);
    return acos(ct);
  }
}

double AngleGradient(const Cartesian &r1, const Cartesian &r2, const Cartesian &r3,
		     Cartesian &g1, Cartesian &g2, Cartesian &g3)
{
  const Cartesian a = r1 - r2, b = r3 - r2;
  const double a2 = a.sq(), b2 = b.sq();
  if (is_almost_zero(a2) || is_almost_zero(b2)) {
    g1.zero();
    g2.zero();
    g3.zero();
    return 0;
  }
  const double ma2 = 1.0/a2, mb2 = 1.0/b2, ab = a*b;
  const double ma = sqrt(ma2), mb = sqrt(mb2);
  const double ma3 = ma*ma2, mb3 = mb*mb2;
  const double t0 = ma*mb, t3 = -ma3*mb*ab, t4 = -ab*ma*mb3;
  g1.x = b.x*t0 + a.x*t3;
  g1.y = b.y*t0 + a.y*t3;
  g1.z = b.z*t0 + a.z*t3;
  g3.x = a.x*t0 + b.x*t4;
  g3.y = a.y*t0 + b.y*t4;
  g3.z = a.z*t0 + b.z*t4;
  g2.x = -g1.x - g3.x;
  g2.y = -g1.y - g3.y;
  g2.z = -g1.z - g3.z;
  double dt_dct;
  double t = theta(ab*t0,dt_dct);
  g1 *= dt_dct;
  g2 *= dt_dct;
  g3 *= dt_dct;
  return t;
}

#ifdef NOT_USED
/***
 * Return cos(theta) 
 * gi  = d cos(theta) / dri   i = 1,2,3
 * hijab = d cos(theta) / dria drjb  i,j = 1,2,3  a,b = x,y,z
 ***/
double AngleGradient(const Cartesian &r1, const Cartesian &r2, const Cartesian &r3,
		     Cartesian &g1, Cartesian &g2, Cartesian &g3,
		     Tensor &h11, Tensor &h12, Tensor &h13, 
		     Tensor &h22, Tensor &h23, Tensor &h33)
{
  const double ax = r1.x - r2.x;
  const double ay = r1.y - r2.y;
  const double az = r1.z - r2.z;
  const double bx = r3.x - r2.x;
  const double by = r3.y - r2.y;
  const double bz = r3.z - r2.z;
  const double a2 = ax*ax + ay*ay + az*az;
  const double b2 = bx*bx + by*by + bz*bz;
  if (is_almost_zero(a2) || is_almost_zero(b2)) {
    g1.zero();
    g2.zero();
    g3.zero();
    h11.zero();
    h12.zero();
    h13.zero();
    h22.zero();
    h23.zero();
    h33.zero();
    return 0;
  }
  const double ma2 = 1.0/(ax*ax+ay*ay+az*az);
  const double mb2 = 1.0/(bx*bx+by*by+bz*bz); 
  const double ab = ax*bx+ay*by+az*bz;
  const double ma = sqrt(ma2), mb = sqrt(mb2);
  const double ma3 = ma*ma2, ma5 = ma3*ma2;
  const double mb3 = mb*mb2, mb5 = mb3*mb2;
  const double t0 = ma*mb, t1 = -ma3*mb, t2 = -ma*mb3;
  const double t3 = ab*t1, t4 = ab*t2, t5 = ab*ma3*mb3;
  const double t6 = 3*ab*ma5*mb, t7 = 3*ab*ma*mb5;
  g1.x = bx*t0 + ax*t3;
  g1.y = by*t0 + ay*t3;
  g1.z = bz*t0 + az*t3;
  g3.x = ax*t0 + bx*t4;
  g3.y = ay*t0 + by*t4;
  g3.z = az*t0 + bz*t4;
  g2.x = -g1.x - g3.x;
  g2.y = -g1.y - g3.y;
  g2.z = -g1.z - g3.z;
  const double AAxx = ax*ax, AAxy = ax*ay, AAxz = ax*az;
  const double AAyy = ay*ay, AAyz = ay*az, AAzz = az*az;
  const double BBxx = bx*bx, BBxy = bx*by, BBxz = bx*bz;
  const double BByy = by*by, BByz = by*bz, BBzz = bz*bz;
  const double ABxx = ax*bx, ABxy = ax*by, ABxz = ax*bz;
  const double AByx = ay*bx, AByy = ay*by, AByz = ay*bz;
  const double ABzx = az*bx, ABzy = az*by, ABzz = az*bz;
  h11.xx = 2*ABxx*t1 + t3 + AAxx*t6;
  h11.xy = h11.yx = ABxy*t1 + AByx*t1 + AAxy*t6;
  h11.xz = h11.zx = ABxz*t1 + ABzx*t1 + AAxz*t6;
  h11.yy = 2*AByy*t1 + t3 + AAyy*t6;
  h11.yz = h11.zy = AByz*t1 + ABzy*t1 + AAyz*t6;
  h11.zz = 2*ABzz*t1 + t3 + AAzz*t6;
  h33.xx = 2*ABxx*t2 + t4 + BBxx*t7;
  h33.xy = h33.yx = ABxy*t2 + AByx*t2 + BBxy*t7;
  h33.xz = h33.zx = ABxz*t2 + ABzx*t2 + BBxz*t7;
  h33.yy = 2*AByy*t2 + t4 + BByy*t7;
  h33.yz = h33.zy = AByz*t2 + ABzy*t2 + BByz*t7;
  h33.zz = 2*ABzz*t2 + t4 + BBzz*t7;
  h13.xx = t0 + AAxx*t1 + BBxx*t2 + ABxx*t5;
  h13.yx = AAxy*t1 + BBxy*t2 + AByx*t5;
  h13.zx = AAxz*t1 + BBxz*t2 + ABzx*t5;
  h13.xy = AAxy*t1 + BBxy*t2 + ABxy*t5;
  h13.yy = t0 + AAyy*t1 + BByy*t2 + AByy*t5;
  h13.zy = AAyz*t1 + BByz*t2 + ABzy*t5;
  h13.xz = AAxz*t1 + BBxz*t2 + ABxz*t5;
  h13.yz = AAyz*t1 + BByz*t2 + AByz*t5;
  h13.zz = t0 + AAzz*t1 + BBzz*t2 + ABzz*t5;
  h12 = -h11 - h13;
  h22 = h11 + h13 + h13.transpose() + h33;
  h23 = -h33 - h13;
  return ab*t0;
}
#endif

double DihedralGradient(const Cartesian &r1, const Cartesian &r2,
			const Cartesian &r3, const Cartesian &r4,
			Cartesian &g1, Cartesian &g2, 
			Cartesian &g3, Cartesian &g4)
{
  const Cartesian r12 = r1 - r2;
  const Cartesian r23 = r2 - r3;
  const Cartesian r34 = r3 - r4;
  const double s12 = r12.sq();
  const double s23 = r23.sq();
  const double s34 = r34.sq();
  const double r1223 = r12*r23;
  const double r2334 = r23*r34;
  const double r1234 = r12*r34;
  const double a = r1223*r2334 - r1234*s23;
  const double b = s12*s23 - r1223*r1223;
  const double c = s23*s34 - r2334*r2334;
  const double bc = b*c;
  if (bc < small_val()) {
    g1.zero();
    g2.zero();
    g3.zero();
    g4.zero();
    return 0;
  }
  const Cartesian g1a = r23*r2334 - r34*s23;
  const Cartesian g1b = 2*r12*s23 - 2*r1223*r23;
  const Cartesian g1c(0,0,0);
  const Cartesian g2a = (r12-r23)*r2334 + r1223*r34 + r34*s23 - 2*r1234*r23;
  const Cartesian g2b = -2*r12*s23 + 2*s12*r23 - 2*r1223*(r12-r23);
  const Cartesian g2c = 2*r23*s34 - 2*r2334*r34;
  const Cartesian g3a = -r12*r2334 + r1223*(r23-r34) - r12*s23 + 2*r1234*r23;
  const Cartesian g3b = -2*s12*r23 + 2*r1223*r12;
  const Cartesian g3c = -2*r23*s34 + 2*s23*r34 - 2*r2334*(r23-r34);
  const Cartesian g4a = -r1223*r23 + r12*s23;
  const Cartesian g4b(0,0,0);
  const Cartesian g4c = -2*s23*r34 + 2*r2334*r23;
  const double bc12 = 1/sqrt(bc);
  const double bc32 = bc12/bc;
  const double halfabc32 = 0.5*a*bc32;
  g1 = g1a*bc12 - halfabc32*(g1b*c + b*g1c);
  g2 = g2a*bc12 - halfabc32*(g2b*c + b*g2c);
  g3 = g3a*bc12 - halfabc32*(g3b*c + b*g3c);
  g4 = g4a*bc12 - halfabc32*(g4b*c + b*g4c);
  double dt_dct;
  double t = theta(a*bc12,dt_dct);
  if (r12.cross(r34)*r23 < 0) {
    dt_dct = -dt_dct;
    t = -t;
  }
  g1 *= dt_dct;
  g2 *= dt_dct;
  g3 *= dt_dct;
  g4 *= dt_dct;
  return t;
}

static Tensor skew(const Cartesian &u)
{
  return Tensor(0,-u.z,u.y,
		u.z,0,-u.x,
		-u.y,u.x,0);
}

double NormalDistanceGradient(const Cartesian &r0, const Cartesian &r1, 
			      const Cartesian &r2, const Cartesian &r3,
			      Cartesian &g0, Cartesian &g1, Cartesian &g2, Cartesian &g3)
{
  const Cartesian r21 = r2 - r1;
  const Cartesian r31 = r3 - r1;
  const Cartesian r01 = r0 - r1;
  const Cartesian u = r21.cross(r31);
  const double umag = u.magnitude();
  if (umag <= 0) {
    g0.zero();
    g1.zero();
    g2.zero();
    g3.zero();
    return 0;
  }
  const Cartesian uu = u/umag;
  const Cartesian duur01 = (Tensor(1) - Tensor(uu,uu))*r01/umag;
  g0 = uu;
  g2 = skew(r31)*duur01;
  g3 = skew(r21)*duur01;
  g1 = -(g0+g2+g3);
  return uu*r01;
}
