from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "constraint",
        ["constraint.pyx"],
        include_dirs=[np.get_include()],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
        libraries=["m"],
    ), 
]

setup(
    name='utils',
    ext_modules=cythonize(ext_modules, language_level = "3"),
)
