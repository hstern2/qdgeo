#include "optimizer.hpp"
#include "fns.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Optimizer::Optimizer(int n, const std::vector<Bond>& bonds, const std::vector<AngleConstraint>& angles,
                     double k_bond, double k_angle)
    : n_(n), bonds_(bonds), angles_(angles), k_bond_(k_bond), k_angle_(k_angle),
      rng_(std::random_device{}()) {}

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
    return e;
}
