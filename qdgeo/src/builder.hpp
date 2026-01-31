#ifndef BUILDER_HPP
#define BUILDER_HPP

#include <vector>
#include <map>
#include <set>
#include <cmath>

struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    double dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    Vec3 cross(const Vec3& v) const { return {y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x}; }
    double norm() const { return std::sqrt(dot(*this)); }
    Vec3 normalized() const { double n = norm(); return n > 1e-10 ? (*this)*(1/n) : Vec3(1,0,0); }
};

class MoleculeBuilder {
public:
    explicit MoleculeBuilder(int n_atoms);
    
    void add_bond(int a1, int a2, double length);
    void set_angle(int a1, int center, int a2, double angle_rad);
    void set_torsion(int a1, int a2, int a3, int a4, double angle_rad);
    void add_ring(const std::vector<int>& atoms);
    
    std::vector<Vec3> build();

private:
    int n_atoms_;
    std::vector<std::vector<int>> adj_;
    std::map<std::pair<int,int>, double> bonds_;
    std::map<std::tuple<int,int,int>, double> angles_;
    std::map<std::tuple<int,int,int,int>, double> torsions_;
    std::vector<std::vector<int>> rings_;
    
    double get_bond_length(int a1, int a2) const;
    double get_angle(int a1, int center, int a2) const;
    double get_torsion(int a1, int a2, int a3, int a4) const;
    
    Vec3 place_atom(const Vec3& p1, const Vec3& p2, const Vec3& p3,
                    double length, double angle, double torsion) const;
    
    // Ring handling
    void place_ring(const std::vector<int>& ring, std::vector<Vec3>& coords, std::vector<bool>& placed);
    std::vector<std::set<int>> find_fused_ring_systems() const;
    void place_fused_system(const std::set<int>& atoms, std::vector<Vec3>& coords, std::vector<bool>& placed);
    
    // Optimization
    double ring_energy(const std::vector<int>& atoms, const std::vector<Vec3>& coords) const;
    bool is_planar_system(const std::vector<int>& atoms) const;
    std::vector<Vec3> initial_ring_coords(const std::vector<int>& atoms, unsigned seed, bool planar) const;
    void optimize_ring_system(const std::vector<int>& atoms, std::vector<Vec3>& coords, bool planar, int max_iter) const;
    void resolve_clashes(std::vector<Vec3>& coords) const;
};

#endif
