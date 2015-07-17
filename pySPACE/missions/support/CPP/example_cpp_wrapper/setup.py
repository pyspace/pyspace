from distutils.core import setup, Extension
import numpy

if __name__ == "__main__":
    ext = Extension('example_wrapper', sources=['example_wrapper.cpp'], include_dirs=[numpy.get_include()])

    setup(name='Example wrapper', version='1.0', ext_modules=[ext],
          description='This is an example wrapper for cpp extensions in python')