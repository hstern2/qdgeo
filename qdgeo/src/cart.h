#ifndef CART_H
#define CART_H

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct { double re, im; } complex_t;
  typedef struct { double x, y, z; } cart_t;
  typedef struct { int x, y, z; } icart_t;
  typedef struct { double xx, yx, zx, xy, yy, zy, xz, yz, zz; } tensor_t;
  
  static inline void multiply(cart_t *a, const tensor_t *t, const cart_t *c)
  {
    a->x = t->xx*c->x + t->xy*c->y + t->xz*c->z;
    a->y = t->yx*c->x + t->yy*c->y + t->yz*c->z;
    a->z = t->zx*c->x + t->zy*c->y + t->zz*c->z;
  }

  static inline void imultiply(cart_t *a, const tensor_t *t, const icart_t *c)
  {
    a->x = t->xx*c->x + t->xy*c->y + t->xz*c->z;
    a->y = t->yx*c->x + t->yy*c->y + t->yz*c->z;
    a->z = t->zx*c->x + t->zy*c->y + t->zz*c->z;
  }

  /* c -> cT */
  static inline void left_operate(cart_t *c, const tensor_t *t)
  {
    const double ax = c->x*t->xx + c->y*t->yx + c->z*t->zx;
    const double ay = c->x*t->xy + c->y*t->yy + c->z*t->zy;
    const double az = c->x*t->xz + c->y*t->yz + c->z*t->zz;
    c->x = ax;
    c->y = ay;
    c->z = az;
  }

  /* c -> Tc */
  static inline void right_operate(const tensor_t *t, cart_t *c)
  {
    const double ax = t->xx*c->x + t->xy*c->y + t->xz*c->z;
    const double ay = t->yx*c->x + t->yy*c->y + t->yz*c->z;
    const double az = t->zx*c->x + t->zy*c->y + t->zz*c->z;
    c->x = ax;
    c->y = ay;
    c->z = az;
  }
  
  static inline void transpose(tensor_t *a, const tensor_t *b)
  {
    a->xx = b->xx;  a->xy = b->yx;  a->xz = b->zx;
    a->yx = b->xy;  a->yy = b->yy;  a->yz = b->zy;
    a->zx = b->xz;  a->zy = b->yz;  a->zz = b->zz;
  }
  
  static inline void scalar_multiply(tensor_t *a, double b)
  {
    a->xx *= b;  a->xy *= b;  a->xz *= b;
    a->yx *= b;  a->yy *= b;  a->yz *= b;
    a->zx *= b;  a->zy *= b;  a->zz *= b;
  }
  
  static inline double dot(const cart_t *a, const cart_t *b)
  {
    return a->x*b->x + a->y*b->y + a->z*b->z;
  }
  
  static inline double cart_sq(const cart_t *a)
  {
    return dot(a,a);
  }
  
  static inline double det(const tensor_t *a)
  {
    return -a->xz*a->yy*a->zx + a->xy*a->yz*a->zx + 
      a->xz*a->yx*a->zy - a->xx*a->yz*a->zy - 
      a->xy*a->yx*a->zz + a->xx*a->yy*a->zz;
  }
  
  static inline void inverse(tensor_t *a, const tensor_t *b)
  {
    const double invdet = 1.0/det(b);
    a->xx = invdet*(-b->yz*b->zy + b->yy*b->zz);
    a->xy = invdet*( b->xz*b->zy - b->xy*b->zz);
    a->xz = invdet*(-b->xz*b->yy + b->xy*b->yz);
    a->yx = invdet*( b->yz*b->zx - b->yx*b->zz);
    a->yy = invdet*(-b->xz*b->zx + b->xx*b->zz);
    a->yz = invdet*( b->xz*b->yx - b->xx*b->yz);
    a->zx = invdet*(-b->yy*b->zx + b->yx*b->zy);
    a->zy = invdet*( b->xy*b->zx - b->xx*b->zy);
    a->zz = invdet*(-b->xy*b->yx + b->xx*b->yy);
  }
  
#ifdef __cplusplus
}
#endif

#endif /* CART_H */
