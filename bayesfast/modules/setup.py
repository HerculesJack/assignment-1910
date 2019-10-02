from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "_poly",
        ["_poly.pyx"],
        # include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='_poly',
    ext_modules=cythonize(ext_modules, language_level = "3"),
)
