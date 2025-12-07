#ifndef COORD_H
#define COORD_H

#include <cmath>
#include <cassert>
#include "cart.h"
#include "out.hpp"

class Cartesian : public cart_t
{
public:
  Cartesian() { }
  Cartesian(double xx, double yy, double zz = 0) { x = xx; y = yy; z = zz; }
  Cartesian & operator+=(const Cartesian &c)
  { x += c.x; y += c.y; z += c.z; return *this; }
  Cartesian & operator-=(const Cartesian &c)
  { x -= c.x; y -= c.y; z -= c.z; return *this; }
  Cartesian & operator*=(double r)
  { x *= r; y *= r; z *= r; return *this; }
  Cartesian & operator/=(double r)
  { x /= r; y /= r; z /= r; return *this; }
  Cartesian & zero()
  { x = 0; y = 0; z = 0; return *this; }
  Cartesian & set(double xx, double yy, double zz)
  { x = xx; y = yy; z = zz; return *this; }
  Cartesian operator-() const
  { return Cartesian(-x,-y,-z); }
  friend Cartesian operator+(const Cartesian &c1, const Cartesian &c2)
  { return Cartesian(c1.x + c2.x, c1.y + c2.y, c1.z + c2.z); }
  friend Cartesian operator-(const Cartesian &c1, const Cartesian &c2)
  { return Cartesian(c1.x - c2.x, c1.y - c2.y, c1.z - c2.z); }
  friend double operator*(const Cartesian &c1, const Cartesian &c2)
  { return c1.x * c2.x + c1.y * c2.y + c1.z * c2.z; }
  friend Cartesian operator*(double r, const Cartesian &c)
  { return Cartesian(r*c.x, r*c.y, r*c.z); }
  friend Cartesian operator*(const Cartesian &c, double r)
  { return r*c; }
  friend Cartesian operator/(const Cartesian &c, double r)
  { return Cartesian(c.x/r, c.y/r, c.z/r); }
  double sq() const
  { return x*x + y*y + z*z; }
  double magnitude() const
  { return sqrt(sq()); }
  double distance(const Cartesian &c) const 
  { return (*this - c).magnitude(); }
  Cartesian & scale_to_unit_magnitude()
  { return *this /= magnitude(); }
  Cartesian as_unit_vector() const
  { return *this / magnitude(); }
  Cartesian cross(const Cartesian &c) const
  { return Cartesian(y*c.z - z*c.y, z*c.x - x*c.z, x*c.y - y*c.x); }
  Cartesian multiply(const Cartesian &c) const
  { return Cartesian(x*c.x, y*c.y, z*c.z); }
  Cartesian divide(const Cartesian &c) const
  { return Cartesian(x/c.x, y/c.y, z/c.z); }
  operator double *()
  { return (double *) this; }
  operator const double *() const
  { return (const double *) this; }
  Cartesian & apply(double (*f)(double))
  { x = f(x); y = f(y); z = f(z); return *this; }
  Cartesian map(double (*f)(double)) const
  { return Cartesian(f(x),f(y),f(z)); }
  double coord(int c) const
  {
    switch (c) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    assert(0);
    return 0;
  }
  double &coord(int c)
  {
    switch (c) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    assert(0);
    return x;
  }
};

class Tensor : public tensor_t
{
public:
  Tensor() { }
  Tensor(double r) // scalar -- create diagonal tensor
  { xx = yy = zz = r; xy = yx = xz = zx = yz = zy = 0;}
  Tensor(const Cartesian &a) // diagonal tensor
  { xx = a.x; yy = a.y; zz = a.z; xy = yx = xz = zx = yz = zy = 0;}
  Tensor(const double &axx, const double &axy, const double &axz,
	 const double &ayx, const double &ayy, const double &ayz,
	 const double &azx, const double &azy, const double &azz) 
  { 
    xx = axx; yx = ayx; zx = azx;
    xy = axy; yy = ayy; zy = azy;
    xz = axz; yz = ayz; zz = azz;
  }
  Tensor(const Cartesian &a, const Cartesian &b) // outer product
  {
    xx = a.x*b.x; yx = a.y*b.x; zx = a.z*b.x;
    xy = a.x*b.y; yy = a.y*b.y; zy = a.z*b.y;
    xz = a.x*b.z; yz = a.y*b.z; zz = a.z*b.z;
  }
  Cartesian &col(int c) 
  {
    switch (c) {
    case 0:
      return (Cartesian &) xx;
    case 1:
      return (Cartesian &) xy;
    case 2:
      return (Cartesian &) xz;
    }
    assert(0);
    return (Cartesian &) xx;
  }
  const Cartesian &col(int c) const
  {
    switch (c) {
    case 0:
      return (const Cartesian &) xx;
    case 1:
      return (const Cartesian &) xy;
    case 2:
      return (const Cartesian &) xz;
    }
    assert(0);
    return (const Cartesian &) xx;
  }
  Cartesian row(int r) const
  {
    switch (r) {
    case 0:
      return Cartesian(xx,xy,xz);
    case 1:
      return Cartesian(yx,yy,yz);
    case 2:
      return Cartesian(zx,zy,zz);
    }
    assert(0);
    return Cartesian(0,0,0);
  }
  void zero() { xx = xy = xz = yx = yy = yz = zx = zy = zz = 0; }
  Tensor &set(double xx_, double xy_, double xz_, 
	      double yx_, double yy_, double yz_,
	      double zx_, double zy_, double zz_)
  { 
    xx = xx_; xy = xy_; xz = xz_; 
    yx = yx_; yy = yy_; yz = yz_; 
    zx = zx_; zy = zy_; zz = zz_; 
    return *this;
  }
  Tensor &outer(const Cartesian &a, const Cartesian &b)
  {
    xx = a.x*b.x; xy = a.x*b.y; xz = a.x*b.z;
    yx = a.y*b.x; yy = a.y*b.y; yz = a.y*b.z;
    zx = a.z*b.x; zy = a.z*b.y; zz = a.z*b.z;
    return *this;
  }
  void set_row1(const Cartesian &c) { xx = c.x; xy = c.y; xz = c.z; }
  void set_row2(const Cartesian &c) { yx = c.x; yy = c.y; yz = c.z; }
  void set_row3(const Cartesian &c) { zx = c.x; zy = c.y; zz = c.z; }
  operator double *()
  { return (double *) this; }
  operator const double *() const
  { return (const double *) this; }
  Tensor & operator+=(const Tensor d)
  {
    xx += d.xx; xy += d.xy; xz += d.xz;
    yx += d.yx; yy += d.yy; yz += d.yz;
    zx += d.zx; zy += d.zy; zz += d.zz;
    return *this;
  }
  Tensor & operator-=(const Tensor d)
  {
    xx -= d.xx; xy -= d.xy; xz -= d.xz;
    yx -= d.yx; yy -= d.yy; yz -= d.yz;
    zx -= d.zx; zy -= d.zy; zz -= d.zz;
    return *this;
  }
  Tensor & operator*=(double r)
  {
    xx *= r; xy *= r; xz *= r;
    yx *= r; yy *= r; yz *= r;
    zx *= r; zy *= r; zz *= r;
    return *this;
  }
  double trace() const { return xx + yy + zz; }
  Tensor transpose() const { return Tensor(xx,yx,zx,xy,yy,zy,xz,yz,zz); }
  double determinant() const { return det(this); }
  Cartesian diagonal() const { return Cartesian(xx,yy,zz); }
  Tensor inverse() const
  {
    Tensor t;
    ::inverse(&t,this);
    return t;
  }
  Tensor operator-() const { return Tensor(-xx,-xy,-xz,-yx,-yy,-yz,-zx,-zy,-zz); }
  Tensor & apply(double (*f)(double))
  {
    xx = f(xx); xy = f(xy); xz = f(xz);
    yx = f(yx); yy = f(yy); yz = f(yz);
    zx = f(zx); zy = f(zy); zz = f(zz);
    return *this;
  }
  Tensor map(double (*f)(double)) const
  { 
    return Tensor(f(xx),f(xy),f(xz),
		  f(yx),f(yy),f(yz),
		  f(zx),f(zy),f(zz)); 
  }
  friend Cartesian operator*(const Tensor &t, const Cartesian &c)
  {
    Cartesian z(c);
    right_operate(&t,&z);
    return z;
  }
  friend Cartesian operator*(const Cartesian &c, const Tensor &t)
  {
    Cartesian z(c);
    left_operate(&z,&t);
    return z;
  }
  friend Tensor operator*(double r, const Tensor &t)
  {
    return Tensor(r*t.xx, r*t.xy, r*t.xz,
		  r*t.yx, r*t.yy, r*t.yz,
		  r*t.zx, r*t.zy, r*t.zz);
  }
  friend Tensor operator/(const Tensor &t, double r)
  {
    return Tensor(t.xx/r, t.xy/r, t.xz/r,
		  t.yx/r, t.yy/r, t.yz/r,
		  t.zx/r, t.zy/r, t.zz/r);
  }
  friend Tensor operator*(const Tensor &t, double r)
  { return r*t; }
  friend Tensor operator+(const Tensor &t1, const Tensor &t2)
  {
    return Tensor(t1.xx+t2.xx, t1.xy+t2.xy, t1.xz+t2.xz,
		  t1.yx+t2.yx, t1.yy+t2.yy, t1.yz+t2.yz,
		  t1.zx+t2.zx, t1.zy+t2.zy, t1.zz+t2.zz);
  }
  friend Tensor operator-(const Tensor &t1, const Tensor &t2)
  {
    return Tensor(t1.xx-t2.xx, t1.xy-t2.xy, t1.xz-t2.xz,
		  t1.yx-t2.yx, t1.yy-t2.yy, t1.yz-t2.yz,
		  t1.zx-t2.zx, t1.zy-t2.zy, t1.zz-t2.zz);
  }
  friend Tensor operator*(const Tensor &d1, const Tensor &d2)
  {
    return Tensor(d1.xx*d2.xx + d1.xy*d2.yx + d1.xz*d2.zx,
		  d1.xx*d2.xy + d1.xy*d2.yy + d1.xz*d2.zy, 
		  d1.xx*d2.xz + d1.xy*d2.yz + d1.xz*d2.zz,
		  d1.yx*d2.xx + d1.yy*d2.yx + d1.yz*d2.zx, 
		  d1.yx*d2.xy + d1.yy*d2.yy + d1.yz*d2.zy, 
		  d1.yx*d2.xz + d1.yy*d2.yz + d1.yz*d2.zz,
		  d1.zx*d2.xx + d1.zy*d2.yx + d1.zz*d2.zx, 
		  d1.zx*d2.xy + d1.zy*d2.yy + d1.zz*d2.zy,
		  d1.zx*d2.xz + d1.zy*d2.yz + d1.zz*d2.zz);
  }
};

inline ostream & operator<<(ostream &s, const Cartesian &c) 
{ 
  return s << c.x << " " << c.y << " " << c.z; 
}

inline istream & operator>>(istream &s, Cartesian &c)
{ 
  return s >> c.x >> c.y >> c.z; 
}

inline ostream & operator<<(ostream &s, const Tensor &c)
{ 
  s << "\n"
    << c.xx << " " << c.xy << " " << c.xz << "\n"
    << c.yx << " " << c.yy << " " << c.yz << "\n"
    << c.zx << " " << c.zy << " " << c.zz << "\n"
    << "\n";
  return s;
}

inline istream & operator>>(istream &s, Tensor &c)
{ 
  return s >> c.xx >> c.xy >> c.xz
	   >> c.yx >> c.yy >> c.yz
	   >> c.zx >> c.zy >> c.zz;
}

#endif
