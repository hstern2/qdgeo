#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "builder.hpp"

namespace py = pybind11;

// Convert Vec3 vector to numpy array
static py::array_t<double> coords_to_numpy(const std::vector<Vec3>& coords) {
    int n = (int)coords.size();
    py::array_t<double> result({n, 3});
    auto r = result.mutable_unchecked<2>();
    for (int i = 0; i < n; i++) {
        r(i, 0) = coords[i].x;
        r(i, 1) = coords[i].y;
        r(i, 2) = coords[i].z;
    }
    return result;
}

PYBIND11_MODULE(_qdgeo, m) {
    m.doc() = "QDGeo: Quick & dirty molecular geometry construction";
    
    py::class_<MoleculeBuilder>(m, "MoleculeBuilder")
        .def(py::init<int>(), py::arg("n_atoms"),
             "Create a molecule builder for n_atoms atoms")
        .def("add_bond", &MoleculeBuilder::add_bond,
             py::arg("atom1"), py::arg("atom2"), py::arg("ideal_length"),
             "Add a bond between two atoms with ideal length")
        .def("set_angle", &MoleculeBuilder::set_angle,
             py::arg("atom1"), py::arg("center"), py::arg("atom2"), py::arg("angle_rad"),
             "Set the ideal angle (in radians) for atom1-center-atom2")
        .def("set_torsion", &MoleculeBuilder::set_torsion,
             py::arg("atom1"), py::arg("atom2"), py::arg("atom3"), py::arg("atom4"), 
             py::arg("angle_rad"),
             "Set the torsion angle (in radians) for atom1-atom2-atom3-atom4")
        .def("add_ring", &MoleculeBuilder::add_ring,
             py::arg("ring_atoms"),
             "Add a ring with the specified atom indices (in order around the ring)")
        .def("build", [](MoleculeBuilder& builder) {
            return coords_to_numpy(builder.build());
        }, "Build the molecule and return coordinates as numpy array (n_atoms, 3)");
    
    // Convenience function for quick building
    m.def("build_molecule", [](
            int n_atoms,
            const std::vector<std::tuple<int, int, double>>& bonds,
            const std::vector<std::tuple<int, int, int, double>>& angles,
            const std::vector<std::tuple<int, int, int, int, double>>& torsions,
            const std::vector<std::vector<int>>& rings) {
        
        MoleculeBuilder builder(n_atoms);
        
        for (const auto& b : bonds) {
            builder.add_bond(std::get<0>(b), std::get<1>(b), std::get<2>(b));
        }
        
        for (const auto& a : angles) {
            builder.set_angle(std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a));
        }
        
        for (const auto& t : torsions) {
            builder.set_torsion(std::get<0>(t), std::get<1>(t), std::get<2>(t), 
                               std::get<3>(t), std::get<4>(t));
        }
        
        for (const auto& r : rings) {
            builder.add_ring(r);
        }
        
        return coords_to_numpy(builder.build());
    },
    py::arg("n_atoms"),
    py::arg("bonds"),
    py::arg("angles") = std::vector<std::tuple<int, int, int, double>>(),
    py::arg("torsions") = std::vector<std::tuple<int, int, int, int, double>>(),
    py::arg("rings") = std::vector<std::vector<int>>(),
    R"doc(
    Build a molecule with rigid-body construction.
    
    Args:
        n_atoms: Number of atoms
        bonds: List of (atom1, atom2, ideal_length) tuples
        angles: List of (atom1, center, atom2, angle_rad) tuples
        torsions: List of (atom1, atom2, atom3, atom4, angle_rad) tuples
        rings: List of ring atom index lists
    
    Returns:
        numpy array of shape (n_atoms, 3) with coordinates
    )doc");
}
