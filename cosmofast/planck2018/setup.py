from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "_commander",
        ["_commander.pyx"],
        include_dirs=[np.get_include()],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
        libraries=["m"],
    ), 
    Extension(
        "_plik_lite",
        ["_plik_lite.pyx"],
        #include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        #libraries=["m"],
    ),
    Extension(
        "_plik_lite_diag",
        ["_plik_lite_diag.pyx"],
        #include_dirs=[np.get_include()],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
        #libraries=["m"],
    ),
    Extension(
        "_simall",
        ["_simall.pyx"],
        include_dirs=[np.get_include()],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
        libraries=["m"],
    )
]

setup(
    name='planck2018',
    ext_modules=cythonize(ext_modules, language_level = "3"),
)