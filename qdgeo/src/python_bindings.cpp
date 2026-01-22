#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <limits>
#include "optimizer.hpp"

namespace py = pybind11;

static void coords_to_numpy(const std::vector<Cartesian>& coords, py::array_t<double>& arr) {
    auto r = arr.mutable_unchecked<2>();
    for (int i = 0; i < (int)coords.size(); i++) {
        r(i, 0) = coords[i].x;
        r(i, 1) = coords[i].y;
        r(i, 2) = coords[i].z;
    }
}

static void numpy_to_coords(py::array_t<double> arr, std::vector<Cartesian>& coords) {
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
        throw std::runtime_error("Coordinates must be shape (n_atoms, 3)");
    double* ptr = static_cast<double*>(buf.ptr);
    int n = buf.shape[0];
    coords.resize(n);
    for (int i = 0; i < n; i++)
        coords[i] = Cartesian(ptr[i*3], ptr[i*3+1], ptr[i*3+2]);
}

PYBIND11_MODULE(_qdgeo, m) {
    m.doc() = "Quick and dirty molecular geometry conformation optimization";
    
    py::class_<Bond>(m, "Bond")
        .def(py::init<int, int, double>())
        .def_readwrite("atom1", &Bond::a1)
        .def_readwrite("atom2", &Bond::a2)
        .def_readwrite("ideal_length", &Bond::len);
    
    py::class_<AngleConstraint>(m, "Angle")
        .def(py::init<int, int, int, double>())
        .def_readwrite("atom1", &AngleConstraint::a1)
        .def_readwrite("atom2", &AngleConstraint::a2)
        .def_readwrite("atom3", &AngleConstraint::a3)
        .def_readwrite("ideal_angle", &AngleConstraint::ang);
    
    py::class_<DihedralConstraint>(m, "Dihedral")
        .def(py::init<int, int, int, int, double>())
        .def_readwrite("atom1", &DihedralConstraint::a1)
        .def_readwrite("atom2", &DihedralConstraint::a2)
        .def_readwrite("atom3", &DihedralConstraint::a3)
        .def_readwrite("atom4", &DihedralConstraint::a4)
        .def_readwrite("ideal_dihedral", &DihedralConstraint::phi);
    
    py::class_<Optimizer>(m, "Optimizer")
        .def(py::init<int, const std::vector<Bond>&, const std::vector<AngleConstraint>&, double, double,
                      const std::vector<DihedralConstraint>&, double, double, double>(),
             py::arg("n_atoms"), py::arg("bonds"), py::arg("angles"),
             py::arg("bond_force_constant") = 1.0, py::arg("angle_force_constant") = 1.0,
             py::arg("dihedrals") = std::vector<DihedralConstraint>(),
             py::arg("dihedral_force_constant") = 1.0,
             py::arg("repulsion_force_constant") = 0.0, py::arg("repulsion_cutoff") = 3.0)
        .def("generate_random_coords", [](Optimizer& opt, double scale = 2.0) {
            std::vector<Cartesian> coords;
            opt.random_coords(coords, scale);
            py::array_t<double> result({opt.n_atoms(), 3});
            coords_to_numpy(coords, result);
            return result;
        }, py::arg("scale") = 2.0)
        .def("optimize", [](Optimizer& opt, py::array_t<double> coords, double tol,
                            double ls_tol, int maxeval, int verbose) {
            std::vector<Cartesian> cart_coords;
            numpy_to_coords(coords, cart_coords);
            if ((int)cart_coords.size() != opt.n_atoms())
                throw std::runtime_error("Wrong number of atoms");
            bool conv = opt.optimize(cart_coords, tol, ls_tol, maxeval, verbose);
            coords_to_numpy(cart_coords, coords);
            return std::make_tuple(conv, opt.energy(cart_coords));
        }, py::arg("coords"), py::arg("tolerance") = 1e-6, py::arg("linesearch_tolerance") = 0.5,
           py::arg("maxeval") = 1000, py::arg("verbose") = 0)
        .def("get_energy", [](Optimizer& opt, py::array_t<double> coords) {
            std::vector<Cartesian> cart_coords;
            numpy_to_coords(coords, cart_coords);
            return opt.energy(cart_coords);
        }, py::arg("coords"));
    
    m.def("optimize", [](int n, const std::vector<std::tuple<int, int, double>>& bonds,
                                  const std::vector<std::tuple<int, int, int, double>>& angles,
                                  double k_bond, double k_angle, double tol, double ls_tol,
                                  int maxeval, int verbose,
                                  const std::vector<std::tuple<int, int, int, int, double>>& dihedrals,
                                  double k_dihedral, double k_repulsion, double repulsion_cutoff) {
        std::vector<Bond> b_vec;
        for (const auto& b : bonds)
            b_vec.emplace_back(std::get<0>(b), std::get<1>(b), std::get<2>(b));
        std::vector<AngleConstraint> a_vec;
        for (const auto& a : angles)
            a_vec.emplace_back(std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a));
        std::vector<DihedralConstraint> d_vec;
        for (const auto& d : dihedrals)
            d_vec.emplace_back(std::get<0>(d), std::get<1>(d), std::get<2>(d), std::get<3>(d), std::get<4>(d));
        
        Optimizer opt(n, b_vec, a_vec, k_bond, k_angle, d_vec, k_dihedral, k_repulsion, repulsion_cutoff);
        
        std::vector<Cartesian> coords;
        opt.random_coords(coords);
        bool conv = opt.optimize(coords, tol, ls_tol, maxeval, verbose);
        double energy = opt.energy(coords);
        
        py::array_t<double> result({n, 3});
        coords_to_numpy(coords, result);
        return std::make_tuple(result, conv, energy);
    }, py::arg("n_atoms"), py::arg("bonds"), py::arg("angles"),
       py::arg("bond_force_constant") = 1.0, py::arg("angle_force_constant") = 1.0,
       py::arg("tolerance") = 1e-6, py::arg("linesearch_tolerance") = 0.5,
       py::arg("maxeval") = 1000, py::arg("verbose") = 0,
       py::arg("dihedrals") = std::vector<std::tuple<int, int, int, int, double>>(),
       py::arg("dihedral_force_constant") = 1.0,
       py::arg("repulsion_force_constant") = 0.0, py::arg("repulsion_cutoff") = 3.0);
}
