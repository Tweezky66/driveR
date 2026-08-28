from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ext_modules = [
    Pybind11Extension(
        "risk_engine_cpp",
        ["risk_engine_src/risk_engine.cpp"],
        cxx_std= 17
    ),
]

setup(
    name="risk_engine_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
)