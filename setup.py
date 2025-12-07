from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
import pybind11
import os

src = os.path.join("qdgeo", "src")
c_files = [os.path.join(src, f) for f in ["cgmin.c", "fns.c"]]
cpp_files = [os.path.join(src, f) for f in ["python_bindings.cpp", "optimizer.cpp", "geograd.cpp"]]

class build_ext(_build_ext):
    def build_extensions(self):
        original = self.compiler._compile
        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.c'):
                extra_postargs = [a for a in extra_postargs if not a.startswith('-std=c++')]
            return original(obj, src, ext, cc_args, extra_postargs, pp_opts)
        self.compiler._compile = _compile
        for ext in self.extensions:
            if isinstance(ext, Pybind11Extension):
                ext.cxx_std = 11
        super().build_extensions()

setup(
    packages=["qdgeo"],
    ext_modules=[Pybind11Extension("_qdgeo", cpp_files + c_files, 
                                   include_dirs=[src, pybind11.get_include()], cxx_std=11)],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
