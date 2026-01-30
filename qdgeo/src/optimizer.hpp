#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <random>
#include "coord.hpp"
#include "cgmin.h"
#include "geograd.hpp"

struct Bond {
    int a1, a2;
    double len;
    Bond(int a1, int a2, double len) : a1(a1), a2(a2), len(len) {}
};

struct AngleConstraint {
    int a1, a2, a3;
    double ang;
    AngleConstraint(int a1, int a2, int a3, double ang) : a1(a1), a2(a2), a3(a3), ang(ang) {}
};

struct DihedralConstraint {
    int a1, a2, a3, a4;
    double phi;
    DihedralConstraint(int a1, int a2, int a3, int a4, double phi) : a1(a1), a2(a2), a3(a3), a4(a4), phi(phi) {}
};

struct CoordinateConstraint {
    int atom;
    double x, y, z;  // target position
    CoordinateConstraint(int atom, double x, double y, double z) : atom(atom), x(x), y(y), z(z) {}
};

struct RepulsionPair {
    int i, j;
    double sigma2;  // (sum of vdW radii)^2
    RepulsionPair(int i, int j, double sigma2) : i(i), j(j), sigma2(sigma2) {}
};

// Van der Waals radii in Angstroms (Bondi radii)
inline double vdw_radius(int atomic_num) {
    // Direct array lookup for common elements (indexed by atomic number)
    // Elements: H=1, C=6, N=7, O=8, F=9, P=15, S=16, Cl=17, Br=35, I=53
    static const double radii[] = {
        1.70,  // 0: unused (default to C)
        1.20,  // 1: H
        1.70,  // 2: He (default)
        1.70,  // 3: Li (default)
        1.70,  // 4: Be (default)
        1.70,  // 5: B (default)
        1.70,  // 6: C
        1.55,  // 7: N
        1.52,  // 8: O
        1.47,  // 9: F
    };
    if (atomic_num >= 1 && atomic_num <= 9)
        return radii[atomic_num];
    switch (atomic_num) {
        case 15: return 1.80;  // P
        case 16: return 1.80;  // S
        case 17: return 1.75;  // Cl
        case 35: return 1.85;  // Br
        case 53: return 1.98;  // I
        default: return 1.70;  // default to C radius
    }
}

class Optimizer {
public:
    Optimizer(int n, const std::vector<Bond>& bonds, const std::vector<AngleConstraint>& angles,
              double k_bond = 1.0, double k_angle = 1.0,
              const std::vector<DihedralConstraint>& dihedrals = std::vector<DihedralConstraint>(),
              double k_dihedral = 1.0, double k_repulsion = 0.0,
              const std::vector<CoordinateConstraint>& coordinates = std::vector<CoordinateConstraint>(),
              double k_coordinate = 1.0,
              const std::vector<int>& atomic_numbers = std::vector<int>(),
              unsigned int seed = 0);  // 0 means use random_device
    
    void random_coords(std::vector<Cartesian>& coords, double scale = 0.0);  // 0 = auto-scale based on n_atoms
    bool optimize(std::vector<Cartesian>& coords, double tol = 1e-6,
                  double ls_tol = 0.5, int maxeval = 1000, int verbose = 0);
    double energy(const std::vector<Cartesian>& coords) const;
    int n_atoms() const { return n_; }
    
private:
    int n_;
    std::vector<Bond> bonds_;
    std::vector<AngleConstraint> angles_;
    std::vector<DihedralConstraint> dihedrals_;
    std::vector<CoordinateConstraint> coordinates_;
    double k_bond_, k_angle_, k_dihedral_, k_repulsion_, k_coordinate_;
    std::vector<RepulsionPair> repulsion_pairs_;
    std::vector<std::vector<int>> bond_graph_;
    std::mt19937 rng_;
    
    void build_repulsion_pairs(const std::vector<int>& atomic_numbers);
    void build_bond_graph();
    double dihedral_energy(double phi, double target_phi) const;
    static double calc_fr(int n, const double* x, double* r, void* user);
    static void to_cart(const double* x, int n, std::vector<Cartesian>& coords);
    static void to_array(const std::vector<Cartesian>& coords, double* x);
};

#endif
