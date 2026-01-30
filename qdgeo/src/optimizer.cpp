#include "optimizer.hpp"
#include "lbfgs.hpp"
#include "fns.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Optimizer::Optimizer(int n, const std::vector<Bond>& bonds, const std::vector<AngleConstraint>& angles,
                     double k_bond, double k_angle,
                     const std::vector<DihedralConstraint>& dihedrals,
                     double k_dihedral, double k_repulsion,
                     const std::vector<CoordinateConstraint>& coordinates,
                     double k_coordinate,
                     const std::vector<int>& atomic_numbers,
                     unsigned int seed)
    : n_(n), bonds_(bonds), angles_(angles), dihedrals_(dihedrals),
      coordinates_(coordinates),
      k_bond_(k_bond), k_angle_(k_angle), k_dihedral_(k_dihedral),
      k_repulsion_(k_repulsion),
      k_coordinate_(k_coordinate),
      rng_(seed != 0 ? seed : std::random_device{}()) {
    build_bond_graph();
    // Use provided atomic numbers or default to carbon (6)
    if (atomic_numbers.empty()) {
        build_repulsion_pairs(std::vector<int>(n_, 6));
    } else {
        build_repulsion_pairs(atomic_numbers);
    }
}

void Optimizer::build_bond_graph() {
    bond_graph_.clear();
    bond_graph_.resize(n_);
    for (const auto& b : bonds_) {
        bond_graph_[b.a1].push_back(b.a2);
        bond_graph_[b.a2].push_back(b.a1);
    }
}

void Optimizer::build_repulsion_pairs(const std::vector<int>& atomic_numbers) {
    repulsion_pairs_.clear();
    if (n_ == 0) return;
    
    // Reuse vectors across iterations to avoid repeated allocations
    std::vector<int> dist(n_);
    std::vector<int> queue;
    queue.reserve(n_);
    
    for (int i = 0; i < n_; i++) {
        // Reset distances (only need to reset nodes we visited)
        std::fill(dist.begin(), dist.end(), -1);
        queue.clear();
        
        dist[i] = 0;
        queue.push_back(i);
        
        // BFS with early termination at distance 5
        for (size_t idx = 0; idx < queue.size(); idx++) {
            int u = queue[idx];
            if (dist[u] >= 5) continue;  // Don't explore beyond distance 5
            for (int v : bond_graph_[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        
        // Only check nodes j > i to avoid duplicates
        // Include all pairs that are 1-5 or greater (d >= 4) or disconnected (d == -1)
        // 1-2 (d=1), 1-3 (d=2), 1-4 (d=3) are excluded; 1-5 (d=4) and beyond are included
        for (int j = i + 1; j < n_; j++) {
            int d = dist[j];
            if (d == -1 || d >= 4) {
                // Use sum of vdW radii as sigma
                double sigma = vdw_radius(atomic_numbers[i]) + vdw_radius(atomic_numbers[j]);
                repulsion_pairs_.emplace_back(i, j, sigma * sigma);
            }
        }
    }
}

double Optimizer::dihedral_energy(double phi, double target_phi) const {
    double cos_phi = cos(phi);
    double cos_target = cos(target_phi);
    if (fabs(cos_target - 1.0) < 0.01) {
        double one_minus_cos = 1.0 - cos_phi;
        return 0.5 * k_dihedral_ * one_minus_cos * one_minus_cos;
    } else if (fabs(cos_target + 1.0) < 0.01) {
        double one_plus_cos = 1.0 + cos_phi;
        return 0.5 * k_dihedral_ * one_plus_cos * one_plus_cos;
    } else {
        double delta = phi - target_phi;
        double cos_delta = cos(delta);
        return 0.5 * k_dihedral_ * (1.0 - cos_delta * cos_delta);
    }
}

void Optimizer::random_coords(std::vector<Cartesian>& coords, double scale) {
    coords.resize(n_);
    
    // Auto-scale based on number of atoms if scale is 0
    // Use cube root of n_atoms * 2.0 to give atoms enough room
    if (scale <= 0.0) {
        scale = std::cbrt((double)n_) * 2.0;
    }
    
    // First, initialize atoms with coordinate constraints to their target positions
    std::vector<bool> initialized(n_, false);
    for (const auto& c : coordinates_) {
        if (c.atom >= 0 && c.atom < n_) {
            coords[c.atom] = Cartesian(c.x, c.y, c.z);
            initialized[c.atom] = true;
        }
    }
    
    // Then initialize remaining atoms randomly
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (int i = 0; i < n_; i++) {
        if (!initialized[i]) {
            coords[i] = Cartesian(dist(rng_), dist(rng_), dist(rng_));
        }
    }
}

double Optimizer::calc_fr(int n, const double* x, double* r, void* user) {
    Optimizer* opt = static_cast<Optimizer*>(user);
    // Reinterpret x as array of Cartesian (zero-copy - Cartesian is POD {x,y,z})
    const Cartesian* coords = reinterpret_cast<const Cartesian*>(x);
    std::memset(r, 0, n * sizeof(double));
    
    double e = 0.0;
    const double small = small_val();
    
    // Bond contributions
    const double k_bond = opt->k_bond_;
    for (const auto& b : opt->bonds_) {
        Cartesian diff = coords[b.a1] - coords[b.a2];
        double d2 = diff.sq();
        double d = sqrt(d2);
        double delta = d - b.len;
        e += 0.5 * k_bond * delta * delta;
        
        if (d > small) {
            double g_scale = k_bond * delta / d;
            int i1 = b.a1 * 3, i2 = b.a2 * 3;
            r[i1] += g_scale * diff.x; r[i1+1] += g_scale * diff.y; r[i1+2] += g_scale * diff.z;
            r[i2] -= g_scale * diff.x; r[i2+1] -= g_scale * diff.y; r[i2+2] -= g_scale * diff.z;
        }
    }
    
    // Angle contributions
    const double k_angle = opt->k_angle_;
    for (const auto& a : opt->angles_) {
        Cartesian g1, g2, g3;
        double theta = AngleGradient(coords[a.a1], coords[a.a2], coords[a.a3], g1, g2, g3);
        double delta = theta - a.ang;
        e += 0.5 * k_angle * delta * delta;
        
        double scale = k_angle * delta;
        int i1 = a.a1 * 3, i2 = a.a2 * 3, i3 = a.a3 * 3;
        r[i1] += scale * g1.x; r[i1+1] += scale * g1.y; r[i1+2] += scale * g1.z;
        r[i2] += scale * g2.x; r[i2+1] += scale * g2.y; r[i2+2] += scale * g2.z;
        r[i3] += scale * g3.x; r[i3+1] += scale * g3.y; r[i3+2] += scale * g3.z;
    }
    
    // Dihedral contributions
    const double k_dihedral = opt->k_dihedral_;
    for (const auto& dih : opt->dihedrals_) {
        Cartesian g1, g2, g3, g4;
        double phi = DihedralGradient(coords[dih.a1], coords[dih.a2], coords[dih.a3], coords[dih.a4], g1, g2, g3, g4);
        e += opt->dihedral_energy(phi, dih.phi);
        
        double cos_phi = cos(phi);
        double sin_phi = sin(phi);
        double cos_target = cos(dih.phi);
        double scale;
        if (fabs(cos_target - 1.0) < 0.01) {
            scale = k_dihedral * (1.0 - cos_phi) * sin_phi;
        } else if (fabs(cos_target + 1.0) < 0.01) {
            scale = -k_dihedral * (1.0 + cos_phi) * sin_phi;
        } else {
            scale = k_dihedral * sin(2.0 * (phi - dih.phi));
        }
        int i1 = dih.a1 * 3, i2 = dih.a2 * 3, i3 = dih.a3 * 3, i4 = dih.a4 * 3;
        r[i1] += scale * g1.x; r[i1+1] += scale * g1.y; r[i1+2] += scale * g1.z;
        r[i2] += scale * g2.x; r[i2+1] += scale * g2.y; r[i2+2] += scale * g2.z;
        r[i3] += scale * g3.x; r[i3+1] += scale * g3.y; r[i3+2] += scale * g3.z;
        r[i4] += scale * g4.x; r[i4+1] += scale * g4.y; r[i4+2] += scale * g4.z;
    }
    
    // Coordinate constraint contributions
    const double k_coord = opt->k_coordinate_;
    for (const auto& c : opt->coordinates_) {
        double dx = coords[c.atom].x - c.x;
        double dy = coords[c.atom].y - c.y;
        double dz = coords[c.atom].z - c.z;
        e += 0.5 * k_coord * (dx*dx + dy*dy + dz*dz);
        int i = c.atom * 3;
        r[i] += k_coord * dx;
        r[i+1] += k_coord * dy;
        r[i+2] += k_coord * dz;
    }
    
    // Non-bonded repulsion for 1-5+ pairs
    // Simple 1/r^12 potential: E = k * (sigma/r)^12
    // This prevents atoms from overlapping without a hard cutoff
    const double k_rep = opt->k_repulsion_;
    if (k_rep > 0.0) {
        for (const auto& pair : opt->repulsion_pairs_) {
            double dx = coords[pair.i].x - coords[pair.j].x;
            double dy = coords[pair.i].y - coords[pair.j].y;
            double dz = coords[pair.i].z - coords[pair.j].z;
            double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 > small * small) {
                // E = k_rep * (sigma/d)^12 = k_rep * sigma2^6 / d2^6
                double ratio2 = pair.sigma2 / d2;
                double ratio6 = ratio2 * ratio2 * ratio2;
                double ratio12 = ratio6 * ratio6;
                e += k_rep * ratio12;
                // gradient: -12 * k_rep * (sigma/d)^12 / d^2 * diff
                double g_scale = -12.0 * k_rep * ratio12 / d2;
                int i1 = pair.i * 3, i2 = pair.j * 3;
                r[i1] += g_scale * dx; r[i1+1] += g_scale * dy; r[i1+2] += g_scale * dz;
                r[i2] -= g_scale * dx; r[i2+1] -= g_scale * dy; r[i2+2] -= g_scale * dz;
            }
        }
    }
    
    for (int i = 0; i < n; i++) r[i] = -r[i];
    return e;
}

void Optimizer::to_cart(const double* x, int n, std::vector<Cartesian>& coords) {
    coords.resize(n);
    std::memcpy(coords.data(), x, n * sizeof(Cartesian));
}

void Optimizer::to_array(const std::vector<Cartesian>& coords, double* x) {
    std::memcpy(x, coords.data(), coords.size() * sizeof(Cartesian));
}

static void translate_to_origin(std::vector<Cartesian>& coords) {
    if (coords.empty()) return;
    Cartesian centroid(0, 0, 0);
    for (const auto& c : coords)
        centroid += c;
    centroid /= coords.size();
    for (auto& c : coords)
        c -= centroid;
}

bool Optimizer::optimize(std::vector<Cartesian>& coords, double tol, double /* ls_tol */,
                         int maxeval, int verbose) {
    int n = n_ * 3;
    std::vector<double> x(n);
    to_array(coords, x.data());
    
    // Use L-BFGS optimizer (faster convergence than conjugate gradient)
    // History size m=10 is a good default for molecular optimization
    int conv = lbfgs::minimize(n, x.data(), tol, maxeval, verbose, calc_fr, this, 10);
    to_cart(x.data(), n_, coords);
    // Don't translate to origin if coordinate constraints are present (they specify absolute positions)
    if (coordinates_.empty()) {
        translate_to_origin(coords);
    }
    return conv != 0;
}

double Optimizer::energy(const std::vector<Cartesian>& coords) const {
    double e = 0.0;
    const double small = small_val();
    
    for (const auto& b : bonds_) {
        double delta = (coords[b.a1] - coords[b.a2]).magnitude() - b.len;
        e += 0.5 * k_bond_ * delta * delta;
    }
    for (const auto& a : angles_) {
        double delta = Angle(coords[a.a1], coords[a.a2], coords[a.a3]) - a.ang;
        e += 0.5 * k_angle_ * delta * delta;
    }
    for (const auto& dih : dihedrals_) {
        double phi = Dihedral(coords[dih.a1], coords[dih.a2], coords[dih.a3], coords[dih.a4]);
        e += dihedral_energy(phi, dih.phi);
    }
    for (const auto& c : coordinates_) {
        double dx = coords[c.atom].x - c.x;
        double dy = coords[c.atom].y - c.y;
        double dz = coords[c.atom].z - c.z;
        e += 0.5 * k_coordinate_ * (dx*dx + dy*dy + dz*dz);
    }
    if (k_repulsion_ > 0.0) {
        for (const auto& pair : repulsion_pairs_) {
            double d2 = (coords[pair.i] - coords[pair.j]).sq();
            if (d2 > small * small) {
                // E = k_rep * (sigma/d)^12
                double ratio2 = pair.sigma2 / d2;
                double ratio6 = ratio2 * ratio2 * ratio2;
                e += k_repulsion_ * ratio6 * ratio6;
            }
        }
    }
    return e;
}
