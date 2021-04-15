from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため
#python setup.py build_ext --inplace
ext = Extension("cymodule", sources=["cymodule.pyx"], include_dirs=['.', get_include()])
setup(name="cymodule", ext_modules=cythonize([ext]))
