#include "builder.hpp"
#include <stdexcept>
#include <cmath>

MoleculeBuilder::MoleculeBuilder(int n_atoms) : n_atoms_(n_atoms), adj_(n_atoms) {}

void MoleculeBuilder::add_bond(int a1, int a2, double ideal_length) {
    if (a1 < 0 || a1 >= n_atoms_ || a2 < 0 || a2 >= n_atoms_)
        throw std::runtime_error("Invalid atom index in add_bond");
    adj_[a1].push_back(a2);
    adj_[a2].push_back(a1);
    auto key = std::make_pair(std::min(a1, a2), std::max(a1, a2));
    bond_lengths_[key] = ideal_length;
}

void MoleculeBuilder::set_angle(int a1, int center, int a2, double angle_rad) {
    // Store in canonical order (smaller index first for the outer atoms)
    if (a1 > a2) std::swap(a1, a2);
    angles_[std::make_tuple(a1, center, a2)] = angle_rad;
}

void MoleculeBuilder::set_torsion(int a1, int a2, int a3, int a4, double angle_rad) {
    // Store both forward and reverse for easy lookup
    torsions_[std::make_tuple(a1, a2, a3, a4)] = angle_rad;
    torsions_[std::make_tuple(a4, a3, a2, a1)] = -angle_rad;
}

void MoleculeBuilder::add_ring(const std::vector<int>& ring_atoms) {
    RingInfo ring;
    ring.atoms = ring_atoms;
    rings_.push_back(ring);
}

double MoleculeBuilder::get_bond_length(int a1, int a2) const {
    auto key = std::make_pair(std::min(a1, a2), std::max(a1, a2));
    auto it = bond_lengths_.find(key);
    return (it != bond_lengths_.end()) ? it->second : 1.5;  // default C-C
}

double MoleculeBuilder::get_angle(int a1, int center, int a2) const {
    int lo = std::min(a1, a2);
    int hi = std::max(a1, a2);
    auto it = angles_.find(std::make_tuple(lo, center, hi));
    if (it != angles_.end()) return it->second;
    // Default to tetrahedral angle
    return std::acos(-1.0 / 3.0);  // ~109.47°
}

double MoleculeBuilder::get_torsion(int a1, int a2, int a3, int a4) const {
    auto it = torsions_.find(std::make_tuple(a1, a2, a3, a4));
    if (it != torsions_.end()) return it->second;
    // Check reverse
    it = torsions_.find(std::make_tuple(a4, a3, a2, a1));
    if (it != torsions_.end()) return -it->second;
    return DEFAULT_TORSION;  // staggered/anti
}

// Place an atom D given:
// - p1: great-grandparent position (for torsion reference)
// - p2: grandparent position
// - p3: parent position (bonded to new atom)
// - bond_length: distance from p3 to new atom
// - angle: angle p2-p3-D in radians
// - torsion: dihedral p1-p2-p3-D in radians
Vec3 MoleculeBuilder::place_atom(const Vec3& p1, const Vec3& p2, const Vec3& p3,
                                  double bond_length, double angle, double torsion) {
    // Vector from p2 to p3 (bond axis we're rotating around for torsion)
    Vec3 bc = (p3 - p2).normalized();
    
    // Vector from p1 to p2 (reference for torsion)
    Vec3 ab = (p2 - p1).normalized();
    
    // Create local coordinate frame
    // n is perpendicular to the p1-p2-p3 plane
    Vec3 n = ab.cross(bc);
    double n_norm = n.norm();
    if (n_norm < 1e-10) {
        // ab and bc are parallel, create arbitrary perpendicular
        if (std::abs(bc.x) < 0.9) {
            n = bc.cross(Vec3(1, 0, 0));
        } else {
            n = bc.cross(Vec3(0, 1, 0));
        }
    }
    n = n.normalized();
    
    // m is perpendicular to bc and n (in the abc plane, pointing toward a)
    Vec3 m = n.cross(bc);
    
    // New atom position in local frame:
    // - Along -bc by cos(angle)
    // - Perpendicular by sin(angle), rotated by torsion
    double d_bc = -std::cos(angle) * bond_length;
    double d_perp = std::sin(angle) * bond_length;
    
    // Apply torsion rotation in the plane perpendicular to bc
    double d_m = d_perp * std::cos(torsion);
    double d_n = d_perp * std::sin(torsion);
    
    return p3 + bc * d_bc + m * d_m + n * d_n;
}

void MoleculeBuilder::place_initial_atoms(std::vector<Vec3>& coords, std::vector<bool>& placed) {
    if (n_atoms_ == 0) return;
    
    // Place first atom at origin
    coords[0] = Vec3(0, 0, 0);
    placed[0] = true;
    
    if (n_atoms_ == 1 || adj_[0].empty()) return;
    
    // Place second atom along +x
    int second = adj_[0][0];
    double len = get_bond_length(0, second);
    coords[second] = Vec3(len, 0, 0);
    placed[second] = true;
}

void MoleculeBuilder::place_ring(std::vector<Vec3>& coords, std::vector<bool>& placed,
                                  const RingInfo& ring) {
    int n = ring.size();
    if (n < 3) return;
    
    // Check if any atom is already placed
    int placed_idx = -1;
    int placed_atom = -1;
    for (int i = 0; i < n; i++) {
        if (placed[ring.atoms[i]]) {
            placed_idx = i;
            placed_atom = ring.atoms[i];
            break;
        }
    }
    
    // Calculate ring geometry
    double internal_angle = M_PI * (n - 2) / n;  // internal angle of regular polygon
    
    // For 5 and 6-membered rings, use ideal geometries
    // 6-membered: chair, but we'll start with planar and it's fine for building
    // 5-membered: envelope, but planar is close enough
    
    // Calculate circumradius for regular polygon with given bond lengths
    // For simplicity, use average bond length in the ring
    double avg_bond = 0;
    for (int i = 0; i < n; i++) {
        int a1 = ring.atoms[i];
        int a2 = ring.atoms[(i + 1) % n];
        avg_bond += get_bond_length(a1, a2);
    }
    avg_bond /= n;
    
    // Circumradius R = bond_length / (2 * sin(π/n))
    double R = avg_bond / (2 * std::sin(M_PI / n));
    
    // Place ring in xy plane centered at origin (or offset from placed atom)
    Vec3 center(0, 0, 0);
    double start_angle = 0;
    
    if (placed_idx >= 0) {
        // Adjust center and rotation to match already-placed atom
        // For simplicity, place remaining atoms relative to the placed one
        double angle_step = 2 * M_PI / n;
        start_angle = -placed_idx * angle_step;
        center = coords[placed_atom] - Vec3(R * std::cos(start_angle + placed_idx * angle_step),
                                             R * std::sin(start_angle + placed_idx * angle_step), 0);
    }
    
    // Place all ring atoms
    double angle_step = 2 * M_PI / n;
    for (int i = 0; i < n; i++) {
        int atom = ring.atoms[i];
        if (!placed[atom]) {
            double angle = start_angle + i * angle_step;
            coords[atom] = center + Vec3(R * std::cos(angle), R * std::sin(angle), 0);
            placed[atom] = true;
        }
    }
}

bool MoleculeBuilder::find_reference_atoms(int atom, const std::vector<bool>& placed,
                                            int& ref1, int& ref2, int& ref3) {
    ref1 = ref2 = ref3 = -1;
    
    // Find a placed neighbor (this will be the parent = ref3)
    for (int nb : adj_[atom]) {
        if (placed[nb]) {
            ref3 = nb;
            break;
        }
    }
    if (ref3 < 0) return false;
    
    // Find a placed neighbor of ref3 (grandparent = ref2)
    for (int nb : adj_[ref3]) {
        if (placed[nb] && nb != atom) {
            ref2 = nb;
            break;
        }
    }
    if (ref2 < 0) return false;
    
    // Find a placed neighbor of ref2 (great-grandparent = ref1)
    for (int nb : adj_[ref2]) {
        if (placed[nb] && nb != ref3) {
            ref1 = nb;
            break;
        }
    }
    
    // If no great-grandparent, try another neighbor of ref3
    if (ref1 < 0) {
        for (int nb : adj_[ref3]) {
            if (placed[nb] && nb != atom && nb != ref2) {
                ref1 = nb;
                break;
            }
        }
    }
    
    // If still no ref1, create a synthetic reference point
    if (ref1 < 0) {
        // Use a point perpendicular to ref2-ref3
        return false;  // Will handle in place_remaining_atoms
    }
    
    return true;
}

void MoleculeBuilder::place_remaining_atoms(std::vector<Vec3>& coords, std::vector<bool>& placed) {
    // BFS from placed atoms
    std::queue<int> q;
    for (int i = 0; i < n_atoms_; i++) {
        if (placed[i]) {
            for (int nb : adj_[i]) {
                if (!placed[nb]) q.push(nb);
            }
        }
    }
    
    while (!q.empty()) {
        int atom = q.front();
        q.pop();
        
        if (placed[atom]) continue;
        
        // Find reference atoms
        int ref1, ref2, ref3;
        if (!find_reference_atoms(atom, placed, ref1, ref2, ref3)) {
            // Need at least one placed neighbor
            bool has_placed_nb = false;
            for (int nb : adj_[atom]) {
                if (placed[nb]) {
                    has_placed_nb = true;
                    ref3 = nb;
                    break;
                }
            }
            if (!has_placed_nb) {
                q.push(atom);  // Try again later
                continue;
            }
            
            // Only have parent - this is the second atom from a component
            // Place along x from parent
            if (ref2 < 0) {
                double len = get_bond_length(atom, ref3);
                // Find a direction that doesn't collide
                Vec3 dir(1, 0, 0);
                for (int nb : adj_[ref3]) {
                    if (placed[nb]) {
                        Vec3 existing = coords[nb] - coords[ref3];
                        if (existing.norm() > 1e-6) {
                            // Place perpendicular to existing bond
                            dir = Vec3(-existing.y, existing.x, 0).normalized();
                            break;
                        }
                    }
                }
                coords[atom] = coords[ref3] + dir * len;
                placed[atom] = true;
                for (int nb : adj_[atom]) {
                    if (!placed[nb]) q.push(nb);
                }
                continue;
            }
            
            // Have parent and grandparent, but no great-grandparent
            // Create synthetic reference
            Vec3 v32 = coords[ref2] - coords[ref3];
            Vec3 perp;
            if (std::abs(v32.x) < 0.9 * v32.norm()) {
                perp = v32.cross(Vec3(1, 0, 0)).normalized();
            } else {
                perp = v32.cross(Vec3(0, 1, 0)).normalized();
            }
            Vec3 synthetic_ref1 = coords[ref2] + perp;
            
            double len = get_bond_length(atom, ref3);
            double ang = get_angle(ref2, ref3, atom);
            double tor = get_torsion(ref2, ref2, ref3, atom);  // Use ref2 twice as placeholder
            
            coords[atom] = place_atom(synthetic_ref1, coords[ref2], coords[ref3], len, ang, tor);
            placed[atom] = true;
            for (int nb : adj_[atom]) {
                if (!placed[nb]) q.push(nb);
            }
            continue;
        }
        
        // Have all three reference atoms
        double len = get_bond_length(atom, ref3);
        double ang = get_angle(ref2, ref3, atom);
        double tor = get_torsion(ref1, ref2, ref3, atom);
        
        coords[atom] = place_atom(coords[ref1], coords[ref2], coords[ref3], len, ang, tor);
        placed[atom] = true;
        
        // Add unplaced neighbors to queue
        for (int nb : adj_[atom]) {
            if (!placed[nb]) q.push(nb);
        }
    }
}

std::vector<Vec3> MoleculeBuilder::build() {
    std::vector<Vec3> coords(n_atoms_);
    std::vector<bool> placed(n_atoms_, false);
    
    // Place rings first (they have specific geometry requirements)
    for (const auto& ring : rings_) {
        place_ring(coords, placed, ring);
    }
    
    // Place initial atoms if nothing placed yet
    bool any_placed = false;
    for (int i = 0; i < n_atoms_; i++) {
        if (placed[i]) { any_placed = true; break; }
    }
    if (!any_placed) {
        place_initial_atoms(coords, placed);
    }
    
    // Place remaining atoms via BFS
    place_remaining_atoms(coords, placed);
    
    return coords;
}
