#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <limits>
#include <thread>
#include <random>
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
    
    py::class_<CoordinateConstraint>(m, "Coordinate")
        .def(py::init<int, double, double, double>())
        .def_readwrite("atom", &CoordinateConstraint::atom)
        .def_readwrite("x", &CoordinateConstraint::x)
        .def_readwrite("y", &CoordinateConstraint::y)
        .def_readwrite("z", &CoordinateConstraint::z);
    
    py::class_<Optimizer>(m, "Optimizer")
        .def(py::init<int, const std::vector<Bond>&, const std::vector<AngleConstraint>&, double, double,
                      const std::vector<DihedralConstraint>&, double, double, double,
                      const std::vector<CoordinateConstraint>&, double>(),
             py::arg("n_atoms"), py::arg("bonds"), py::arg("angles"),
             py::arg("bond_force_constant") = 1.0, py::arg("angle_force_constant") = 1.0,
             py::arg("dihedrals") = std::vector<DihedralConstraint>(),
             py::arg("dihedral_force_constant") = 1.0,
             py::arg("repulsion_force_constant") = 0.0, py::arg("repulsion_cutoff") = 3.0,
             py::arg("coordinates") = std::vector<CoordinateConstraint>(),
             py::arg("coordinate_force_constant") = 1.0)
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
                                  double k_dihedral, double k_repulsion, double repulsion_cutoff,
                                  const std::vector<std::tuple<int, double, double, double>>& coordinates,
                                  double k_coordinate,
                                  int n_starts) {
        std::vector<Bond> b_vec;
        for (const auto& b : bonds)
            b_vec.emplace_back(std::get<0>(b), std::get<1>(b), std::get<2>(b));
        std::vector<AngleConstraint> a_vec;
        for (const auto& a : angles)
            a_vec.emplace_back(std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a));
        std::vector<DihedralConstraint> d_vec;
        for (const auto& d : dihedrals)
            d_vec.emplace_back(std::get<0>(d), std::get<1>(d), std::get<2>(d), std::get<3>(d), std::get<4>(d));
        std::vector<CoordinateConstraint> c_vec;
        for (const auto& c : coordinates)
            c_vec.emplace_back(std::get<0>(c), std::get<1>(c), std::get<2>(c), std::get<3>(c));
        
        // Determine number of threads (use hardware concurrency, but cap at n_starts)
        unsigned int n_threads = std::min((unsigned int)n_starts, std::thread::hardware_concurrency());
        if (n_threads == 0) n_threads = 1;
        
        // Results storage for each thread
        struct Result {
            std::vector<Cartesian> coords;
            double energy = std::numeric_limits<double>::max();
            bool converged = false;
        };
        std::vector<Result> results(n_starts);
        
        // Create optimizers for each thread (each needs its own RNG state)
        // Use a single random_device call + sequential seeds to avoid race conditions
        std::random_device rd;
        unsigned int base_seed = rd();
        std::vector<Optimizer> optimizers;
        optimizers.reserve(n_starts);
        for (int i = 0; i < n_starts; i++) {
            optimizers.emplace_back(n, b_vec, a_vec, k_bond, k_angle, d_vec, k_dihedral, 
                                   k_repulsion, repulsion_cutoff, c_vec, k_coordinate,
                                   base_seed + i);  // Unique seed for each
        }
        
        // Worker function for each start
        auto worker = [&](int start_idx) {
            Optimizer& opt = optimizers[start_idx];
            std::vector<Cartesian> coords;
            opt.random_coords(coords);
            int v = (start_idx == 0) ? verbose : 0;
            bool conv = opt.optimize(coords, tol, ls_tol, maxeval, v);
            double energy = opt.energy(coords);
            results[start_idx].coords = std::move(coords);
            results[start_idx].energy = energy;
            results[start_idx].converged = conv;
        };
        
        // Launch threads in batches
        for (unsigned int batch_start = 0; batch_start < (unsigned int)n_starts; batch_start += n_threads) {
            std::vector<std::thread> threads;
            unsigned int batch_end = std::min(batch_start + n_threads, (unsigned int)n_starts);
            for (unsigned int i = batch_start; i < batch_end; i++) {
                threads.emplace_back(worker, i);
            }
            for (auto& t : threads) {
                t.join();
            }
        }
        
        // Find best result
        int best_idx = 0;
        for (int i = 1; i < n_starts; i++) {
            if (results[i].energy < results[best_idx].energy) {
                best_idx = i;
            }
        }
        
        py::array_t<double> result({n, 3});
        coords_to_numpy(results[best_idx].coords, result);
        return std::make_tuple(result, results[best_idx].converged, results[best_idx].energy);
    }, py::arg("n_atoms"), py::arg("bonds"), py::arg("angles"),
       py::arg("bond_force_constant") = 1.0, py::arg("angle_force_constant") = 1.0,
       py::arg("tolerance") = 1e-6, py::arg("linesearch_tolerance") = 0.5,
       py::arg("maxeval") = 1000, py::arg("verbose") = 0,
       py::arg("dihedrals") = std::vector<std::tuple<int, int, int, int, double>>(),
       py::arg("dihedral_force_constant") = 1.0,
       py::arg("repulsion_force_constant") = 0.0, py::arg("repulsion_cutoff") = 3.0,
       py::arg("coordinates") = std::vector<std::tuple<int, double, double, double>>(),
       py::arg("coordinate_force_constant") = 1.0,
       py::arg("n_starts") = 10);
}
