This module gives you an example on how to wrap C code whitin python using the python C-API
It contains the sources example_wrapper.cpp and it's header example_wrapper.h
The setup.py needed to build the extension and the test.py in order test the compiled module.


Compiling the module:

In order to compile the example wrapper run:
python setup.py build_ext --inplace

(For further informations on the setup.py see:
https://docs.python.org/2/extending/building.html)


You'll get a example_wrapper.so inside the folder this is due to the --inplace flag. 
If you want to install the module in your site-package folder remove the --inplace flag.


Testing the module:

To test the extension simply run:
python test.py

The expected outcome would be:
The calculations are equal: True