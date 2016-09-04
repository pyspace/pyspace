from distutils.core import setup, Extension
import numpy

if __name__ == "__main__":
    ext = Extension('variance_tools', sources=['variance_tools.cpp'], include_dirs=[numpy.get_include()])

    setup(name='foo', version='1.0', description='Test description', ext_modules=[ext])