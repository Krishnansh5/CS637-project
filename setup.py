from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

fast_clarans_extension = Extension(
    name="fast_clarans",
    sources=["fast_clarans.pyx"],
    libraries=["fast_clarans"],
    library_dirs=["fast_clarans_lib"],
    include_dirs=["fast_clarans_lib"],
    extra_compile_args=["-std=c++11"],
    language="c++",
)
setup(
    name="fast_clarans",
    ext_modules=cythonize([fast_clarans_extension])
)