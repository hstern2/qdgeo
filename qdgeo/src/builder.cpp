#include "builder.hpp"
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <random>

MoleculeBuilder::MoleculeBuilder(int n_atoms) : n_atoms_(n_atoms), adj_(n_atoms) {}

void MoleculeBuilder::add_bond(int a1, int a2, double length) {
    if (a1 < 0 || a1 >= n_atoms_ || a2 < 0 || a2 >= n_atoms_)
        throw std::runtime_error("Invalid atom index");
    adj_[a1].push_back(a2);
    adj_[a2].push_back(a1);
    bonds_[{std::min(a1,a2), std::max(a1,a2)}] = length;
}

void MoleculeBuilder::set_angle(int a1, int center, int a2, double angle_rad) {
    if (a1 > a2) std::swap(a1, a2);
    angles_[{a1, center, a2}] = angle_rad;
}

void MoleculeBuilder::set_torsion(int a1, int a2, int a3, int a4, double angle_rad) {
    torsions_[{a1, a2, a3, a4}] = angle_rad;
    torsions_[{a4, a3, a2, a1}] = -angle_rad;
}

void MoleculeBuilder::add_ring(const std::vector<int>& atoms) { rings_.push_back(atoms); }

double MoleculeBuilder::get_bond_length(int a1, int a2) const {
    auto it = bonds_.find({std::min(a1,a2), std::max(a1,a2)});
    return it != bonds_.end() ? it->second : 1.5;
}

double MoleculeBuilder::get_angle(int a1, int center, int a2) const {
    if (a1 > a2) std::swap(a1, a2);
    auto it = angles_.find({a1, center, a2});
    return it != angles_.end() ? it->second : std::acos(-1.0/3.0);
}

double MoleculeBuilder::get_torsion(int a1, int a2, int a3, int a4) const {
    auto it = torsions_.find({a1, a2, a3, a4});
    if (it != torsions_.end()) return it->second;
    it = torsions_.find({a4, a3, a2, a1});
    if (it != torsions_.end()) return -it->second;
    return M_PI;
}

Vec3 MoleculeBuilder::place_atom(const Vec3& p1, const Vec3& p2, const Vec3& p3,
                                  double length, double angle, double torsion) const {
    Vec3 bc = (p3 - p2).normalized();
    Vec3 ab = (p2 - p1).normalized();
    Vec3 n = ab.cross(bc);
    if (n.norm() < 1e-10) n = (std::abs(bc.x) < 0.9) ? bc.cross(Vec3(1,0,0)) : bc.cross(Vec3(0,1,0));
    n = n.normalized();
    Vec3 m = n.cross(bc);
    double d_bc = -std::cos(angle) * length;
    double d_perp = std::sin(angle) * length;
    return p3 + bc * d_bc + m * (d_perp * std::cos(torsion)) + n * (d_perp * std::sin(torsion));
}

// Find fused ring systems (rings sharing 2+ atoms)
std::vector<std::set<int>> MoleculeBuilder::find_fused_ring_systems() const {
    std::vector<std::set<int>> systems;
    std::vector<bool> assigned(rings_.size(), false);
    
    for (size_t i = 0; i < rings_.size(); i++) {
        if (assigned[i]) continue;
        std::set<int> sys(rings_[i].begin(), rings_[i].end());
        assigned[i] = true;
        
        bool changed;
        do {
            changed = false;
            for (size_t j = 0; j < rings_.size(); j++) {
                if (assigned[j]) continue;
                int shared = 0;
                for (int a : rings_[j]) if (sys.count(a)) shared++;
                if (shared >= 2) {
                    sys.insert(rings_[j].begin(), rings_[j].end());
                    assigned[j] = true;
                    changed = true;
                }
            }
        } while (changed);
        
        // Count rings in system
        int cnt = 0;
        for (size_t j = 0; j < rings_.size(); j++) {
            bool in = true;
            for (int a : rings_[j]) if (!sys.count(a)) { in = false; break; }
            if (in) cnt++;
        }
        if (cnt > 1) systems.push_back(sys);
    }
    return systems;
}

// Energy: bond stretching + angle bending + non-bonded repulsion
double MoleculeBuilder::ring_energy(const std::vector<int>& atoms, const std::vector<Vec3>& coords) const {
    double E = 0;
    int n = atoms.size();
    std::set<int> atom_set(atoms.begin(), atoms.end());
    
    // Bond terms (strong)
    for (int a : atoms) {
        for (int b : adj_[a]) {
            if (atom_set.count(b) && a < b) {
                double d = (coords[a] - coords[b]).norm() - get_bond_length(a, b);
                E += 200.0 * d * d;
            }
        }
    }
    
    // Angle terms
    for (int c : atoms) {
        std::vector<int> nbs;
        for (int nb : adj_[c]) if (atom_set.count(nb)) nbs.push_back(nb);
        for (size_t i = 0; i < nbs.size(); i++) {
            for (size_t j = i + 1; j < nbs.size(); j++) {
                Vec3 v1 = (coords[nbs[i]] - coords[c]).normalized();
                Vec3 v2 = (coords[nbs[j]] - coords[c]).normalized();
                double cos_a = std::max(-1.0, std::min(1.0, v1.dot(v2)));
                double d = std::acos(cos_a) - get_angle(nbs[i], c, nbs[j]);
                E += 20.0 * d * d;
            }
        }
    }
    
    // Non-bonded repulsion (stronger, wider range)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            bool bonded = false;
            // Also check if 1-3 (angle) neighbors
            bool angle_neighbor = false;
            for (int nb : adj_[atoms[i]]) {
                if (nb == atoms[j]) { bonded = true; break; }
                for (int nb2 : adj_[nb]) {
                    if (nb2 == atoms[j]) { angle_neighbor = true; break; }
                }
            }
            if (!bonded && !angle_neighbor) {
                double dist = (coords[atoms[i]] - coords[atoms[j]]).norm();
                double min_dist = 2.5;  // Larger minimum distance
                if (dist < min_dist) {
                    double d = min_dist - dist;
                    E += 100.0 * d * d * d;  // Cubic for stronger short-range repulsion
                }
            }
        }
    }
    return E;
}


// Check if a ring system is planar (all ring atoms are sp2)
bool MoleculeBuilder::is_planar_system(const std::vector<int>& atoms) const {
    std::set<int> atom_set(atoms.begin(), atoms.end());
    for (int a : atoms) {
        // Count neighbors in the ring system
        int ring_neighbors = 0;
        for (int nb : adj_[a]) {
            if (atom_set.count(nb)) ring_neighbors++;
        }
        // sp2 atoms have 3 total neighbors, sp3 have 4
        // If an atom has 2 ring neighbors and total 4 neighbors, it's sp3
        int total_neighbors = adj_[a].size();
        if (ring_neighbors == 2 && total_neighbors == 4) {
            return false;  // sp3 carbon in ring
        }
    }
    return true;
}

// Generate starting coordinates for ring system
std::vector<Vec3> MoleculeBuilder::initial_ring_coords(const std::vector<int>& atoms, unsigned seed, bool planar) const {
    std::vector<Vec3> coords(n_atoms_);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> angle_dist(0, 2 * M_PI);
    std::uniform_real_distribution<double> torsion_dist(-M_PI, M_PI);
    std::uniform_real_distribution<double> perturb(-0.2, 0.2);
    
    std::set<int> atom_set(atoms.begin(), atoms.end());
    std::vector<bool> placed(n_atoms_, false);
    
    if (planar) {
        // For planar systems: place each ring as regular polygon, then adjust
        // First, place atoms ring by ring
        for (const auto& ring : rings_) {
            // Check if this ring is part of our atom set
            bool in_set = true;
            for (int a : ring) if (!atom_set.count(a)) { in_set = false; break; }
            if (!in_set) continue;
            
            // Count how many atoms already placed
            int n_placed = 0;
            for (int a : ring) if (placed[a]) n_placed++;
            
            if (n_placed == 0) {
                // First ring: place as regular polygon centered at origin
                int n = ring.size();
                double avg_bond = 0;
                for (int i = 0; i < n; i++) avg_bond += get_bond_length(ring[i], ring[(i+1)%n]);
                avg_bond /= n;
                double R = avg_bond / (2.0 * std::sin(M_PI / n));
                
                for (int i = 0; i < n; i++) {
                    double angle = i * (2.0 * M_PI / n);
                    coords[ring[i]] = Vec3(R * std::cos(angle), R * std::sin(angle), 0);
                    placed[ring[i]] = true;
                }
            } else if (n_placed >= 2) {
                // Find shared edge (two adjacent placed atoms)
                int shared1 = -1, shared2 = -1;
                int n = ring.size();
                for (int i = 0; i < n; i++) {
                    int a = ring[i], b = ring[(i+1)%n];
                    if (placed[a] && placed[b]) {
                        shared1 = a; shared2 = b;
                        break;
                    }
                }
                
                if (shared1 >= 0) {
                    // Place remaining atoms of this ring
                    Vec3 edge = coords[shared2] - coords[shared1];
                    Vec3 mid = (coords[shared1] + coords[shared2]) * 0.5;
                    Vec3 perp(-edge.y, edge.x, 0);
                    perp = perp.normalized();
                    
                    // Determine which side to place new atoms
                    // Use the side away from center of mass of placed atoms
                    Vec3 com(0,0,0);
                    int cnt = 0;
                    for (int a : atoms) {
                        if (placed[a]) { com = com + coords[a]; cnt++; }
                    }
                    com = com * (1.0/cnt);
                    if ((mid + perp - com).norm() < (mid - perp - com).norm()) {
                        perp = perp * (-1);
                    }
                    
                    // Place unplaced atoms in arc
                    std::vector<int> unplaced;
                    for (int a : ring) if (!placed[a]) unplaced.push_back(a);
                    
                    int nu = unplaced.size();
                    double bond_len = edge.norm();
                    double R = bond_len / (2.0 * std::sin(M_PI / ring.size()));
                    
                    for (int i = 0; i < nu; i++) {
                        double frac = (i + 1.0) / (nu + 1.0);
                        double angle = M_PI * frac;
                        Vec3 pos = mid + perp * (R * std::sin(angle)) + edge.normalized() * (bond_len * (0.5 - frac));
                        coords[unplaced[i]] = pos;
                        placed[unplaced[i]] = true;
                    }
                }
            }
        }
        
        // Place any remaining atoms with small random perturbation
        for (int a : atoms) {
            if (!placed[a]) {
                coords[a] = Vec3(perturb(rng), perturb(rng), 0);
                placed[a] = true;
            }
        }
    } else {
        // For 3D systems: BFS with random torsions
        coords[atoms[0]] = Vec3(0, 0, 0);
        placed[atoms[0]] = true;
        
        std::queue<int> q;
        for (int nb : adj_[atoms[0]]) {
            if (atom_set.count(nb)) {
                coords[nb] = Vec3(get_bond_length(atoms[0], nb), 0, 0);
                placed[nb] = true;
                q.push(nb);
                break;
            }
        }
        
        while (!q.empty()) {
            int parent = q.front(); q.pop();
            int gp = -1;
            for (int nb : adj_[parent]) {
                if (atom_set.count(nb) && placed[nb]) { gp = nb; break; }
            }
            
            for (int nb : adj_[parent]) {
                if (atom_set.count(nb) && !placed[nb]) {
                    double len = get_bond_length(parent, nb);
                    double ang = get_angle(gp >= 0 ? gp : 0, parent, nb);
                    double tor = torsion_dist(rng);
                    
                    if (gp >= 0) {
                        Vec3 v = (coords[parent] - coords[gp]).normalized();
                        Vec3 perp = (std::abs(v.x) < 0.9) ? v.cross(Vec3(1,0,0)) : v.cross(Vec3(0,1,0));
                        Vec3 p1 = coords[gp] + perp.normalized();
                        coords[nb] = place_atom(p1, coords[gp], coords[parent], len, ang, tor);
                    } else {
                        double theta = angle_dist(rng);
                        coords[nb] = coords[parent] + Vec3(len * std::cos(theta), len * std::sin(theta), len * 0.1);
                    }
                    placed[nb] = true;
                    q.push(nb);
                }
            }
        }
    }
    return coords;
}

// Optimize with optional planarity constraint and simulated annealing
void MoleculeBuilder::optimize_ring_system(const std::vector<int>& atoms, std::vector<Vec3>& coords, 
                                            bool planar, int max_iter) const {
    const double eps = 1e-5;
    double step = 0.1;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> normal(0, 1);
    
    double temperature = planar ? 0 : 1.0;  // Start with some thermal noise for 3D
    double best_E_ever = ring_energy(atoms, coords);
    std::vector<Vec3> best_coords = coords;
    
    for (int iter = 0; iter < max_iter; iter++) {
        double E0 = ring_energy(atoms, coords);
        if (E0 < 0.01) break;
        
        // Compute gradient
        std::vector<Vec3> grad(coords.size());
        for (int a : atoms) {
            for (int d = 0; d < (planar ? 2 : 3); d++) {
                std::vector<Vec3> cp = coords, cm = coords;
                if (d == 0) { cp[a].x += eps; cm[a].x -= eps; }
                else if (d == 1) { cp[a].y += eps; cm[a].y -= eps; }
                else { cp[a].z += eps; cm[a].z -= eps; }
                double g = (ring_energy(atoms, cp) - ring_energy(atoms, cm)) / (2 * eps);
                if (d == 0) grad[a].x = g;
                else if (d == 1) grad[a].y = g;
                else grad[a].z = g;
            }
            if (planar) grad[a].z = 0;
        }
        
        double gnorm = 0;
        for (int a : atoms) gnorm += grad[a].dot(grad[a]);
        gnorm = std::sqrt(gnorm);
        if (gnorm < 1e-8) {
            temperature *= 0.95;
            if (temperature < 0.01) break;
            continue;
        }
        
        for (int a : atoms) grad[a] = grad[a] * (1.0 / gnorm);
        
        // Add thermal noise for 3D systems
        if (temperature > 0.01 && !planar) {
            for (int a : atoms) {
                grad[a].x += normal(rng) * temperature * 0.1;
                grad[a].y += normal(rng) * temperature * 0.1;
                grad[a].z += normal(rng) * temperature * 0.1;
            }
        }
        
        // Line search
        double best_step = 0, best_E = E0;
        for (double s : {step * 0.25, step * 0.5, step, step * 2.0}) {
            std::vector<Vec3> trial = coords;
            for (int a : atoms) trial[a] = coords[a] - grad[a] * s;
            double E = ring_energy(atoms, trial);
            if (E < best_E) { best_E = E; best_step = s; }
        }
        
        if (best_step > 0) {
            for (int a : atoms) coords[a] = coords[a] - grad[a] * best_step;
            step = std::min(0.5, best_step * 1.2);
            
            if (best_E < best_E_ever) {
                best_E_ever = best_E;
                best_coords = coords;
            }
        } else {
            step *= 0.5;
            if (step < 1e-8) {
                temperature *= 0.9;
                step = 0.1;
            }
        }
        
        // Cool down
        if (iter % 100 == 0) temperature *= 0.8;
    }
    
    // Use best found coordinates
    coords = best_coords;
    
    // For planar systems, ensure z=0
    if (planar) {
        for (int a : atoms) coords[a].z = 0;
    }
}

// Place fused ring system with parallel multi-start optimization
void MoleculeBuilder::place_fused_system(const std::set<int>& atom_set, std::vector<Vec3>& coords, 
                                          std::vector<bool>& placed) {
    std::vector<int> atoms(atom_set.begin(), atom_set.end());
    bool planar = is_planar_system(atoms);
    
    // More starts for complex 3D systems
    int n_starts = planar ? 4 : std::max(16, (int)std::thread::hardware_concurrency() * 4);
    int max_iter = planar ? 500 : 2000;
    
    std::vector<std::vector<Vec3>> results(n_starts);
    std::vector<double> energies(n_starts, 1e9);
    
    // Run optimizations in parallel
    std::vector<std::thread> threads;
    for (int i = 0; i < n_starts; i++) {
        threads.emplace_back([this, &atoms, &results, &energies, i, planar, max_iter]() {
            std::vector<Vec3> trial = initial_ring_coords(atoms, 42 + i * 1000, planar);
            optimize_ring_system(atoms, trial, planar, max_iter);
            results[i] = trial;
            energies[i] = ring_energy(atoms, trial);
        });
    }
    
    for (auto& t : threads) t.join();
    
    // Find best result
    int best = 0;
    for (int i = 1; i < n_starts; i++) {
        if (energies[i] < energies[best]) best = i;
    }
    
    // Copy best coordinates
    for (int a : atoms) {
        coords[a] = results[best][a];
        placed[a] = true;
    }
}

// Place simple ring as regular polygon
void MoleculeBuilder::place_ring(const std::vector<int>& ring, std::vector<Vec3>& coords, 
                                  std::vector<bool>& placed) {
    int n = ring.size();
    if (n < 3) return;
    
    // Skip if multiple atoms already placed
    int num_placed = 0;
    for (int a : ring) if (placed[a]) num_placed++;
    if (num_placed > 1) return;
    
    // Circumradius from average bond length
    double avg_bond = 0;
    for (int i = 0; i < n; i++) avg_bond += get_bond_length(ring[i], ring[(i+1) % n]);
    avg_bond /= n;
    double R = avg_bond / (2.0 * std::sin(M_PI / n));
    
    for (int i = 0; i < n; i++) {
        if (!placed[ring[i]]) {
            double angle = i * (2.0 * M_PI / n);
            coords[ring[i]] = Vec3(R * std::cos(angle), R * std::sin(angle), 0);
            placed[ring[i]] = true;
        }
    }
}

// Quick local optimization to resolve clashes for terminal atoms
void MoleculeBuilder::resolve_clashes(std::vector<Vec3>& coords) const {
    // Find terminal atoms that have actual clashes
    std::vector<int> clashing;
    for (int i = 0; i < n_atoms_; i++) {
        if (adj_[i].size() != 1) continue;  // Only terminal atoms
        int parent = adj_[i][0];
        
        // Check if this atom has any clashes
        for (int j = 0; j < n_atoms_; j++) {
            if (j == i || j == parent) continue;
            // Also skip 1-3 neighbors (angle partners)
            bool is_angle = false;
            for (int nb : adj_[parent]) {
                if (nb == j) { is_angle = true; break; }
            }
            if (is_angle) continue;
            
            double dist = (coords[i] - coords[j]).norm();
            if (dist < 1.5) {
                clashing.push_back(i);
                break;
            }
        }
    }
    
    if (clashing.empty()) return;
    
    // Move clashing atoms away from their clash partners
    for (int iter = 0; iter < 30; iter++) {
        bool improved = false;
        for (int a : clashing) {
            int parent = adj_[a][0];
            Vec3 force(0, 0, 0);
            
            for (int b = 0; b < n_atoms_; b++) {
                if (b == a || b == parent) continue;
                bool is_angle = false;
                for (int nb : adj_[parent]) if (nb == b) { is_angle = true; break; }
                if (is_angle) continue;
                
                Vec3 diff = coords[a] - coords[b];
                double dist = diff.norm();
                if (dist < 1.8 && dist > 0.01) {
                    force = force + diff.normalized() * (1.8 - dist);
                    improved = true;
                }
            }
            
            if (force.norm() > 0.01) {
                double bond_len = get_bond_length(a, parent);
                Vec3 new_pos = coords[a] + force * 0.3;
                coords[a] = coords[parent] + (new_pos - coords[parent]).normalized() * bond_len;
            }
        }
        if (!improved) break;
    }
}

std::vector<Vec3> MoleculeBuilder::build() {
    std::vector<Vec3> coords(n_atoms_);
    std::vector<bool> placed(n_atoms_, false);
    
    // 1. Fused ring systems (parallel optimization)
    std::set<int> in_fused;
    for (const auto& sys : find_fused_ring_systems()) {
        place_fused_system(sys, coords, placed);
        in_fused.insert(sys.begin(), sys.end());
    }
    
    // 2. Simple rings
    for (const auto& ring : rings_) {
        bool fused = false;
        for (int a : ring) if (in_fused.count(a)) { fused = true; break; }
        if (!fused) place_ring(ring, coords, placed);
    }
    
    // 3. Initial atoms if nothing placed
    bool any = false;
    for (int i = 0; i < n_atoms_; i++) if (placed[i]) { any = true; break; }
    if (!any && n_atoms_ > 0) {
        coords[0] = Vec3(0, 0, 0);
        placed[0] = true;
        if (!adj_[0].empty()) {
            coords[adj_[0][0]] = Vec3(get_bond_length(0, adj_[0][0]), 0, 0);
            placed[adj_[0][0]] = true;
        }
    }
    
    // 4. BFS for remaining atoms
    std::queue<int> q;
    for (int i = 0; i < n_atoms_; i++)
        if (placed[i]) for (int nb : adj_[i]) if (!placed[nb]) q.push(nb);
    
    while (!q.empty()) {
        int atom = q.front(); q.pop();
        if (placed[atom]) continue;
        
        int parent = -1;
        for (int nb : adj_[atom]) if (placed[nb]) { parent = nb; break; }
        if (parent < 0) { q.push(atom); continue; }
        
        std::vector<int> pn;  // placed neighbors of parent
        for (int nb : adj_[parent]) if (nb != atom && placed[nb]) pn.push_back(nb);
        
        double len = get_bond_length(atom, parent);
        
        if (pn.empty()) {
            coords[atom] = coords[parent] + Vec3(len, 0, 0);
        } else if (pn.size() == 1) {
            int gp = pn[0];
            int ggp = -1;
            for (int nb : adj_[gp]) if (placed[nb] && nb != parent) { ggp = nb; break; }
            
            Vec3 p1 = (ggp >= 0) ? coords[ggp] : [&]() {
                Vec3 v = (coords[parent] - coords[gp]).normalized();
                Vec3 perp = (std::abs(v.x) < 0.9) ? v.cross(Vec3(1,0,0)) : v.cross(Vec3(0,1,0));
                return coords[gp] + perp.normalized();
            }();
            
            double ang = get_angle(gp, parent, atom);
            double tor = get_torsion(ggp >= 0 ? ggp : gp, gp, parent, atom);
            coords[atom] = place_atom(p1, coords[gp], coords[parent], len, ang, tor);
        } else {
            int total = adj_[parent].size();
            if (total == 3) {
                // sp2: opposite to centroid
                Vec3 cen(0,0,0);
                for (int nb : pn) cen = cen + (coords[nb] - coords[parent]).normalized();
                Vec3 dir = cen * (-1.0);
                if (dir.norm() < 0.1) {
                    Vec3 v1 = (coords[pn[0]] - coords[parent]).normalized();
                    dir = Vec3(-v1.y, v1.x, v1.z);
                }
                coords[atom] = coords[parent] + dir.normalized() * len;
            } else {
                // sp3: tetrahedral geometry
                Vec3 v1 = (coords[pn[0]] - coords[parent]).normalized();
                Vec3 v2 = (coords[pn[1]] - coords[parent]).normalized();
                Vec3 bis = (v1 + v2).normalized();
                Vec3 norm = v1.cross(v2);
                if (norm.norm() < 1e-10) norm = (std::abs(v1.x) < 0.9) ? v1.cross(Vec3(1,0,0)) : v1.cross(Vec3(0,1,0));
                norm = norm.normalized();
                
                double half = std::acos(-1.0/3.0) / 2.0;
                double ip = -std::cos(half), op = std::sin(half);
                
                int third = -1;
                for (int nb : adj_[parent])
                    if (nb != atom && nb != pn[0] && nb != pn[1] && placed[nb]) { third = nb; break; }
                
                Vec3 dir;
                if (third >= 0) {
                    double side = (coords[third] - coords[parent]).normalized().dot(norm);
                    dir = bis * ip + norm * (side < 0 ? op : -op);
                } else {
                    dir = bis * ip + norm * op;
                }
                coords[atom] = coords[parent] + dir.normalized() * len;
            }
        }
        
        placed[atom] = true;
        for (int nb : adj_[atom]) if (!placed[nb]) q.push(nb);
    }
    
    // 5. Resolve any remaining clashes
    resolve_clashes(coords);
    
    return coords;
}
