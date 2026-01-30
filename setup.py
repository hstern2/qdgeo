from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
import pybind11
import sys

src = "qdgeo/src"
cpp_files = [f"{src}/python_bindings.cpp", f"{src}/builder.cpp"]

# Optimization flags - portable across macOS and Linux
extra_compile_args = ["-O3"]
extra_link_args = []

# Platform-specific flags
if sys.platform.startswith("linux"):
    extra_compile_args.append("-march=native")
elif sys.platform == "darwin":
    import platform
    if platform.machine() == "arm64":
        extra_compile_args.append("-mcpu=native")
    else:
        extra_compile_args.append("-march=native")

setup(
    packages=["qdgeo"],
    ext_modules=[
        Pybind11Extension(
            "_qdgeo",
            cpp_files,
            include_dirs=[src, pybind11.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            cxx_std=17,
        )
    ],
    zip_safe=False,
)
