#!/usr/bin/env python
""" Runs all unit tests that can be found in the subdirectory


**Option**
    :-ep: -- Enable parallelization

.. todo:: Write automatic script to turn of parallelization. The one that is used now manually turns of each test cases.

:Author: Hendrik Woehrle (hendrik.woehrle@dfki.de) (extended by Titiruck Nuntapramote)
:Created: 2010/07/21
:Last revision: 2010/05/10
"""
import logging
import os
import unittest
import re
import sys
import glob

enable = False
def add_tests_to_suite(suite, dir, files):
    """ Method to add tests to a given test suite.

    This node plots certain information about the time series objects
    in the electrode configuration space depending on the parameters that are
    given to its constructor:

    **Arguments**
        :suite: -- The test suite, to which the tests are added.
        :dir: -- The directory which contains the files
        :files: -- List of files

    """
    #logger = logging.getLogger('TestLogger')
    for file in files:
        # add python all files, if they start with test_
        if re.match("test_.*\.py$",file) and os.path.isfile(dir+'/'+file) :
            # figure out module name
            moduleName = dir.strip("\.\/").replace("/",".")
            if len(moduleName):
                moduleName += "."
            moduleName += os.path.splitext(file)[0]
            if enable == True:
                print moduleName
                logger.log(logging.DEBUG, 'loading tests from ' + moduleName)
                print moduleName
                tempSuite = unittest.TestLoader().loadTestsFromName(moduleName)
                suite.addTests(tempSuite)
            else:
                name = moduleName.split('.')[-1]
                #for enable/disable parallel
                if name != 'test_filtering' and name != 'test_normalization' and name != 'test_subsampling':
                    print moduleName
                    logger.log(logging.DEBUG, 'loading tests from ' + moduleName)
                    tempSuite = unittest.TestLoader().loadTestsFromName(moduleName)
                    suite.addTests(tempSuite)
                else:
                    print moduleName + "   --disable"

if __name__ == '__main__':
    if len(sys.argv) == 2:
       arg = sys.argv[1]
       if arg == '-ep':
           enable = True
       else:
           print "usage: run_tests.py [-ep](enable parallelization)"
           exit()
    elif len(sys.argv) > 2:
        print "usage: run_tests.py [-ep](enable parallelization)"
        exit()
    file_path = os.path.dirname(os.path.abspath(__file__))
    pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
    if not pyspace_path in sys.path:
        sys.path.append(pyspace_path)
    import pySPACE
    import_path = os.path.abspath(os.path.join(os.path.dirname(pySPACE.__file__), os.path.pardir))
    if not import_path == pyspace_path:
        import warnings
        warnings.warn("Check your python path! "+
                      "'%s' is the expected pySPACE path,"%pyspace_path+
                      " but '%s' is used."%import_path)

    logger = logging.getLogger('TestLogger')

    suite = unittest.TestSuite()
    # collect all tests in subdirectory tree
    tree_path = os.path.dirname(os.path.abspath(__file__))
    # NOTE: For this command, tree path must end with the directory separator
    os.chdir(tree_path)
    os.path.walk('unittests', add_tests_to_suite, suite)

    # finally, run the collected tests
    logger.log(logging.DEBUG,sys.path)
    unittest.TextTestRunner(verbosity=2).run(suite)
