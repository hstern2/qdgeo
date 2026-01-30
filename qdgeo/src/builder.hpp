#ifndef BUILDER_HPP
#define BUILDER_HPP

#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <queue>
#include <set>

// Simple 3D vector struct
struct Vec3 {
    double x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    
    double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    
    Vec3 cross(const Vec3& v) const {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    
    double norm() const { return std::sqrt(x * x + y * y + z * z); }
    
    Vec3 normalized() const {
        double n = norm();
        return n > 1e-10 ? Vec3(x / n, y / n, z / n) : Vec3(1, 0, 0);
    }
};

// Bond information
struct BondInfo {
    int atom1, atom2;
    double length;
    BondInfo(int a1, int a2, double len) : atom1(a1), atom2(a2), length(len) {}
};

// Angle information (for 3-atom angles)
struct AngleInfo {
    int atom1, center, atom2;  // angle is atom1-center-atom2
    double angle_rad;
    AngleInfo(int a1, int c, int a2, double ang) : atom1(a1), center(c), atom2(a2), angle_rad(ang) {}
};

// Torsion specification
struct TorsionSpec {
    int atom1, atom2, atom3, atom4;  // torsion is 1-2-3-4
    double angle_rad;
    TorsionSpec(int a1, int a2, int a3, int a4, double ang) 
        : atom1(a1), atom2(a2), atom3(a3), atom4(a4), angle_rad(ang) {}
};

// Ring information for special handling
struct RingInfo {
    std::vector<int> atoms;
    int size() const { return (int)atoms.size(); }
};

class MoleculeBuilder {
public:
    MoleculeBuilder(int n_atoms);
    
    // Set up the molecular graph
    void add_bond(int a1, int a2, double ideal_length);
    void set_angle(int a1, int center, int a2, double angle_rad);
    void set_torsion(int a1, int a2, int a3, int a4, double angle_rad);
    void add_ring(const std::vector<int>& ring_atoms);
    
    // Build the molecule - returns coordinates
    std::vector<Vec3> build();
    
    // Get the default torsion angle for a bond (e.g., staggered = 180°)
    static constexpr double DEFAULT_TORSION = M_PI;  // 180° (anti/trans)
    
private:
    int n_atoms_;
    std::vector<std::vector<int>> adj_;  // adjacency list
    std::map<std::pair<int,int>, double> bond_lengths_;
    std::map<std::tuple<int,int,int>, double> angles_;
    std::map<std::tuple<int,int,int,int>, double> torsions_;
    std::vector<RingInfo> rings_;
    
    // Build helpers
    double get_bond_length(int a1, int a2) const;
    double get_angle(int a1, int center, int a2) const;
    double get_torsion(int a1, int a2, int a3, int a4) const;
    
    // Place an atom given reference frame
    Vec3 place_atom(const Vec3& p1, const Vec3& p2, const Vec3& p3,
                    double bond_length, double angle, double torsion);
    
    // Place first few atoms (special cases)
    void place_initial_atoms(std::vector<Vec3>& coords, std::vector<bool>& placed);
    
    // Build rings with proper geometry
    void place_ring(std::vector<Vec3>& coords, std::vector<bool>& placed, 
                    const RingInfo& ring);
    
    // BFS placement of remaining atoms
    void place_remaining_atoms(std::vector<Vec3>& coords, std::vector<bool>& placed);
    
    // Find a reference frame for placing an atom
    bool find_reference_atoms(int atom, const std::vector<bool>& placed,
                              int& ref1, int& ref2, int& ref3);
};

#endif // BUILDER_HPP
