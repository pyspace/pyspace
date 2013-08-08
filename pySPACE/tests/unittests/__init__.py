""" Elemental function tests to catch changes and ensure correct working functions

.. todo::   write something general about unittests; how to start an program them,
                advantages, pitfalls,...

There are some unittests for nodes and data types,
but not quite enough and yet there are
bigger system tests missing.

To run all unit tests, you can run 
:mod:`tests.run_unittests`
in the tests folder and
:mod:`tests.check_unittests`
gives you an overview an existing and missing unittests.

Rules:

    * Every test class name should end with TestCase.
    * It should derive from unittest.TestCase (use import unittest).
    * The file structure in the unittests folder is the same as in the software. 
    * File names get an additional `test_`-prefix.
    * Write documentation (brief as possible).
    * In `setUp` you define your testing variables and for each test group
      you define a separate function with meaningful name.
    * Unittests should be simple and short.
    * Use assertion errors.
    * Do not use prints.
    * As a name for the test use the node class name without the `Node`.
    
For further reading refer to: http://docs.python.org/library/unittest.html.
"""

