#ifndef GEOGRAD_H
#define GEOGRAD_H

class Cartesian;
class Tensor;

double Angle(const Cartesian &a, const Cartesian &b, const Cartesian &c);

double Dihedral(const Cartesian &a, const Cartesian &b, 
		const Cartesian &c, const Cartesian &d);

/* Distance from r0 to plane given by r1, r2, r3 */
double NormalDistance(const Cartesian &r0, const Cartesian &r1, 
		      const Cartesian &r2, const Cartesian &r3);

/***
 * Return theta and derivatives
 ***/
double AngleGradient(const Cartesian &r1, const Cartesian &r2, const Cartesian &r3,
		     Cartesian &g1, Cartesian &g2, Cartesian &g3);

/***
 * Return phi and derivatives
 ***/
double DihedralGradient(const Cartesian &r1, const Cartesian &r2,
			const Cartesian &r3, const Cartesian &r4,
			Cartesian &g1, Cartesian &g2, 
			Cartesian &g3, Cartesian &g4);

/* Distance from r0 to plane given by r1, r2, r3 and derivatives */
double NormalDistanceGradient(const Cartesian &r0, const Cartesian &r1, 
			      const Cartesian &r2, const Cartesian &r3,
			      Cartesian &g0, Cartesian &g1, Cartesian &g2, Cartesian &g3);

/***
 * Return z, where z is a distance r from a, 
 * the angle z-a-b is given by theta,
 * and the dihedral z-a-b-c is given by phi
 ***/
Cartesian ZLocation(const Cartesian &a, const Cartesian &b, const Cartesian &c,
		    double r, double theta, double phi);

#endif /* GEOGRAD_H */
