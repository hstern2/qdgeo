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

struct PlanarityConstraint {
    int center, a1, a2, a3;  // center atom should be in plane of a1, a2, a3
    PlanarityConstraint(int center, int a1, int a2, int a3) : center(center), a1(a1), a2(a2), a3(a3) {}
};

class Optimizer {
public:
    Optimizer(int n, const std::vector<Bond>& bonds, const std::vector<AngleConstraint>& angles,
              double k_bond = 1.0, double k_angle = 1.0,
              const std::vector<DihedralConstraint>& dihedrals = std::vector<DihedralConstraint>(),
              double k_dihedral = 1.0, double k_repulsion = 0.0, double repulsion_cutoff = 3.0,
              const std::vector<PlanarityConstraint>& planarities = std::vector<PlanarityConstraint>(),
              double k_planarity = 1.0);
    
    void random_coords(std::vector<Cartesian>& coords, double scale = 2.0);
    bool optimize(std::vector<Cartesian>& coords, double tol = 1e-6,
                  double ls_tol = 0.5, int maxeval = 1000, int verbose = 0);
    double energy(const std::vector<Cartesian>& coords) const;
    int n_atoms() const { return n_; }
    
private:
    int n_;
    std::vector<Bond> bonds_;
    std::vector<AngleConstraint> angles_;
    std::vector<DihedralConstraint> dihedrals_;
    std::vector<PlanarityConstraint> planarities_;
    double k_bond_, k_angle_, k_dihedral_, k_repulsion_, repulsion_cutoff_, k_planarity_;
    std::vector<std::pair<int, int>> exclusions_;
    std::vector<std::vector<int>> bond_graph_;
    std::mt19937 rng_;
    
    void build_exclusions();
    void build_bond_graph();
    int shortest_path(int i, int j) const;
    double dihedral_energy(double phi, double target_phi) const;
    static double calc_fr(int n, const double* x, double* r, void* user);
    static void to_cart(const double* x, int n, std::vector<Cartesian>& coords);
    static void to_array(const std::vector<Cartesian>& coords, double* x);
};

#endif
