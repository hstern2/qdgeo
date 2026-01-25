#include "optimizer.hpp"
#include "fns.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Optimizer::Optimizer(int n, const std::vector<Bond>& bonds, const std::vector<AngleConstraint>& angles,
                     double k_bond, double k_angle,
                     const std::vector<DihedralConstraint>& dihedrals,
                     double k_dihedral, double k_repulsion, double repulsion_cutoff,
                     const std::vector<PlanarityConstraint>& planarities,
                     double k_planarity)
    : n_(n), bonds_(bonds), angles_(angles), dihedrals_(dihedrals), planarities_(planarities),
      k_bond_(k_bond), k_angle_(k_angle), k_dihedral_(k_dihedral),
      k_repulsion_(k_repulsion), repulsion_cutoff_(repulsion_cutoff),
      k_planarity_(k_planarity),
      rng_(std::random_device{}()) {
    build_bond_graph();
    build_exclusions();
}

void Optimizer::build_bond_graph() {
    bond_graph_.clear();
    bond_graph_.resize(n_);
    for (const auto& b : bonds_) {
        bond_graph_[b.a1].push_back(b.a2);
        bond_graph_[b.a2].push_back(b.a1);
    }
}

int Optimizer::shortest_path(int i, int j) const {
    if (i == j) return 0;
    std::vector<int> dist(n_, -1);
    std::vector<int> queue;
    dist[i] = 0;
    queue.push_back(i);
    for (size_t idx = 0; idx < queue.size(); idx++) {
        int u = queue[idx];
        for (int v : bond_graph_[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                if (v == j) return dist[v];
                queue.push_back(v);
            }
        }
    }
    return -1;
}

void Optimizer::build_exclusions() {
    exclusions_.clear();
    for (int i = 0; i < n_; i++) {
        for (int j = i + 1; j < n_; j++) {
            int path_len = shortest_path(i, j);
            if (path_len >= 1 && path_len <= 4) {
                exclusions_.push_back({i, j});
            }
        }
    }
    std::sort(exclusions_.begin(), exclusions_.end());
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
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (int i = 0; i < n_; i++)
        coords[i] = Cartesian(dist(rng_), dist(rng_), dist(rng_));
}

double Optimizer::calc_fr(int n, const double* x, double* r, void* user) {
    Optimizer* opt = static_cast<Optimizer*>(user);
    std::vector<Cartesian> coords;
    to_cart(x, opt->n_, coords);
    std::fill(r, r + n, 0.0);
    
    double e = 0.0;
    
    // Bond contributions
    for (const auto& b : opt->bonds_) {
        Cartesian diff = coords[b.a1] - coords[b.a2];
        double d = diff.magnitude();
        double delta = d - b.len;
        e += 0.5 * opt->k_bond_ * delta * delta;
        
        if (d > small_val()) {
            Cartesian g = (opt->k_bond_ * delta / d) * diff;
            int i1 = b.a1 * 3, i2 = b.a2 * 3;
            r[i1] += g.x; r[i1+1] += g.y; r[i1+2] += g.z;
            r[i2] -= g.x; r[i2+1] -= g.y; r[i2+2] -= g.z;
        }
    }
    
    // Angle contributions
    for (const auto& a : opt->angles_) {
        Cartesian g1, g2, g3;
        double theta = AngleGradient(coords[a.a1], coords[a.a2], coords[a.a3], g1, g2, g3);
        double delta = theta - a.ang;
        e += 0.5 * opt->k_angle_ * delta * delta;
        
        double scale = opt->k_angle_ * delta;
        int i1 = a.a1 * 3, i2 = a.a2 * 3, i3 = a.a3 * 3;
        r[i1] += scale * g1.x; r[i1+1] += scale * g1.y; r[i1+2] += scale * g1.z;
        r[i2] += scale * g2.x; r[i2+1] += scale * g2.y; r[i2+2] += scale * g2.z;
        r[i3] += scale * g3.x; r[i3+1] += scale * g3.y; r[i3+2] += scale * g3.z;
    }
    
    // Dihedral contributions
    for (const auto& d : opt->dihedrals_) {
        Cartesian g1, g2, g3, g4;
        double phi = DihedralGradient(coords[d.a1], coords[d.a2], coords[d.a3], coords[d.a4], g1, g2, g3, g4);
        e += opt->dihedral_energy(phi, d.phi);
        
        double cos_phi = cos(phi);
        double sin_phi = sin(phi);
        double cos_target = cos(d.phi);
        double scale;
        if (fabs(cos_target - 1.0) < 0.01) {
            double one_minus_cos = 1.0 - cos_phi;
            scale = opt->k_dihedral_ * one_minus_cos * sin_phi;
        } else if (fabs(cos_target + 1.0) < 0.01) {
            double one_plus_cos = 1.0 + cos_phi;
            scale = -opt->k_dihedral_ * one_plus_cos * sin_phi;
        } else {
            double delta = phi - d.phi;
            scale = opt->k_dihedral_ * sin(2.0 * delta);
        }
        int i1 = d.a1 * 3, i2 = d.a2 * 3, i3 = d.a3 * 3, i4 = d.a4 * 3;
        r[i1] += scale * g1.x; r[i1+1] += scale * g1.y; r[i1+2] += scale * g1.z;
        r[i2] += scale * g2.x; r[i2+1] += scale * g2.y; r[i2+2] += scale * g2.z;
        r[i3] += scale * g3.x; r[i3+1] += scale * g3.y; r[i3+2] += scale * g3.z;
        r[i4] += scale * g4.x; r[i4+1] += scale * g4.y; r[i4+2] += scale * g4.z;
    }
    
    // Planarity contributions (out-of-plane distance)
    for (const auto& p : opt->planarities_) {
        Cartesian g0, g1, g2, g3;
        double dist = NormalDistanceGradient(coords[p.center], coords[p.a1], coords[p.a2], coords[p.a3],
                                             g0, g1, g2, g3);
        e += 0.5 * opt->k_planarity_ * dist * dist;
        
        double scale = opt->k_planarity_ * dist;
        int i0 = p.center * 3, i1 = p.a1 * 3, i2 = p.a2 * 3, i3 = p.a3 * 3;
        r[i0] += scale * g0.x; r[i0+1] += scale * g0.y; r[i0+2] += scale * g0.z;
        r[i1] += scale * g1.x; r[i1+1] += scale * g1.y; r[i1+2] += scale * g1.z;
        r[i2] += scale * g2.x; r[i2+1] += scale * g2.y; r[i2+2] += scale * g2.z;
        r[i3] += scale * g3.x; r[i3+1] += scale * g3.y; r[i3+2] += scale * g3.z;
    }
    
    // Non-bonded repulsion (only 1-5 and 1-6 interactions)
    if (opt->k_repulsion_ > 0.0) {
        for (int i = 0; i < opt->n_; i++) {
            for (int j = i + 1; j < opt->n_; j++) {
                std::pair<int, int> pair = {i, j};
                if (std::binary_search(opt->exclusions_.begin(), opt->exclusions_.end(), pair))
                    continue;
                
                int path_len = opt->shortest_path(i, j);
                if (path_len != 5 && path_len != 6)
                    continue;
                
                Cartesian diff = coords[i] - coords[j];
                double d = diff.magnitude();
                if (d < opt->repulsion_cutoff_ && d > small_val()) {
                    double d2 = d * d;
                    double d6 = d2 * d2 * d2;
                    double d12 = d6 * d6;
                    e += opt->k_repulsion_ / d12;
                    
                    double g_mag = -12.0 * opt->k_repulsion_ / (d12 * d);
                    Cartesian g = (g_mag / d) * diff;
                    int i1 = i * 3, i2 = j * 3;
                    r[i1] += g.x; r[i1+1] += g.y; r[i1+2] += g.z;
                    r[i2] -= g.x; r[i2+1] -= g.y; r[i2+2] -= g.z;
                }
            }
        }
    }
    
    for (int i = 0; i < n; i++) r[i] = -r[i];
    return e;
}

void Optimizer::to_cart(const double* x, int n, std::vector<Cartesian>& coords) {
    coords.resize(n);
    for (int i = 0; i < n; i++)
        coords[i] = Cartesian(x[i*3], x[i*3+1], x[i*3+2]);
}

void Optimizer::to_array(const std::vector<Cartesian>& coords, double* x) {
    for (size_t i = 0; i < coords.size(); i++) {
        x[i*3] = coords[i].x;
        x[i*3+1] = coords[i].y;
        x[i*3+2] = coords[i].z;
    }
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

bool Optimizer::optimize(std::vector<Cartesian>& coords, double tol, double ls_tol,
                         int maxeval, int verbose) {
    int n = n_ * 3;
    std::vector<double> x(n), r(n), work(4 * n);
    to_array(coords, x.data());
    
    int conv = conjugate_gradient_minimize(n, x.data(), r.data(), nullptr, tol, ls_tol,
                                           maxeval, verbose, calc_fr, nullptr, this, work.data());
    to_cart(x.data(), n_, coords);
    translate_to_origin(coords);
    return conv != 0;
}

double Optimizer::energy(const std::vector<Cartesian>& coords) const {
    double e = 0.0;
    for (const auto& b : bonds_) {
        double delta = (coords[b.a1] - coords[b.a2]).magnitude() - b.len;
        e += 0.5 * k_bond_ * delta * delta;
    }
    for (const auto& a : angles_) {
        double delta = Angle(coords[a.a1], coords[a.a2], coords[a.a3]) - a.ang;
        e += 0.5 * k_angle_ * delta * delta;
    }
    for (const auto& d : dihedrals_) {
        double phi = Dihedral(coords[d.a1], coords[d.a2], coords[d.a3], coords[d.a4]);
        e += dihedral_energy(phi, d.phi);
    }
    for (const auto& p : planarities_) {
        double dist = NormalDistance(coords[p.center], coords[p.a1], coords[p.a2], coords[p.a3]);
        e += 0.5 * k_planarity_ * dist * dist;
    }
    if (k_repulsion_ > 0.0) {
        for (int i = 0; i < n_; i++) {
            for (int j = i + 1; j < n_; j++) {
                std::pair<int, int> pair = {i, j};
                if (std::binary_search(exclusions_.begin(), exclusions_.end(), pair))
                    continue;
                
                int path_len = shortest_path(i, j);
                if (path_len == 5 || path_len == 6) {
                    double d = coords[i].distance(coords[j]);
                    if (d < repulsion_cutoff_ && d > small_val()) {
                        double d2 = d * d;
                        double d6 = d2 * d2 * d2;
                        double d12 = d6 * d6;
                        e += k_repulsion_ / d12;
                    }
                }
            }
        }
    }
    return e;
}
