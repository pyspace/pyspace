#!/usr/bin/env python

""" Runs all unit tests that can be found in the subdirectory


.. todo:: Write automatic script to turn of parallelization. The one that
    is used now manually turns of each test cases.

:Author: Hendrik Woehrle (hendrik.woehrle@dfki.de) (extended by Titiruck Nuntapramote, Andrei Ignat)
:Created: 2010/07/21
:Last revision: 2014/08/08
"""
import logging
import sys
logger = logging.getLogger('TestLogger')
logger.log(logging.DEBUG, sys.path)

import os
import unittest
import re

class SpecificUnittestSuite(unittest.TestSuite):
    def populate_suite(self, enable_parallel):
        """ fill the testing suite with the available tests

        The function is an extension of the normal unittest.TestSuite
        in the sense that it looks for available tests in the `unittest`
        directory and appends the available tests to the testing suite.
        """
        # first, we check if the path of the script is the correct one
        self.check_path()

        # collect all tests in subdirectory tree
        tree_path = os.path.dirname(os.path.abspath(__file__))
        # NOTE: For this command, tree path must end with the directory separator
        os.chdir(tree_path)
        os.path.walk('unittests', self.add_tests_to_suite, (self, enable_parallel))

    @staticmethod
    def check_path():
        """ a simple method that checks the pySPACE folder"""
        # We now check if our script is able to import all the available tests
        file_path = os.path.dirname(os.path.abspath(__file__))
        pyspace_path = file_path[:file_path.rfind('pySPACE') - 1]
        if not (pyspace_path in sys.path):
            sys.path.append(pyspace_path)
        import pySPACE
        import_path = os.path.abspath(
            os.path.join(os.path.dirname(pySPACE.__file__), os.path.pardir))
        if not import_path == pyspace_path:
            import warnings
            warnings.warn("Check your Python path! " +
                          "'%s' is the expected pySPACE path," % pyspace_path +
                          " but '%s' is used." % import_path)

    @staticmethod
    def add_tests_to_suite((suite, enable_parallel), dir, files):
        """ Method to add tests to a given test suite.

        The function is built such that it can be called by os.path.walk.
        The specific call thus requires that the `suite` and
        `enable_parallel` arguments be packed into a tuple.

        **Arguments**
            :(self, enable_parallel): -- The test suite, to which the tests
                are added and a parameter that enables the testing of nodes
                that rely on the `adappt` parallelization module.
            :dir: -- The directory which contains the files
            :files: -- List of files

        """
        for file in files:
            # add all Python files, that start with test_
            if re.match("test_.*\.py$", file) and os.path.isfile(
                    dir + '/' + file):
                # figure out module name
                moduleName = os.path.join(dir, file)
                # do the necessary conversions such that the path becomes
                # a python module reference
                moduleName = moduleName\
                    .replace("\\", ".")\
                    .replace("/", ".")\
                    .replace(".py", "")
                print "--) Tests discovered in " + moduleName

                # check if the user wants to test all the modules
                if enable_parallel:
                    # enter this branch if the nodes that require the
                    # `adappt` module should also be tested
                    logging.debug('loading tests from ' + moduleName)
                    suite.addTests(
                        unittest.TestLoader().loadTestsFromName(moduleName))
                else:
                    # check which nodes rely on the `adappt` module and
                    # ignore them
                    name = moduleName.split('.')[-1]
                    if name != 'test_filtering' and \
                        name != 'test_normalization' and \
                            name != 'test_subsampling':

                        logging.debug('loading tests from ' + moduleName)
                        suite.addTests(
                            unittest.TestLoader().loadTestsFromName(moduleName))
                    else:
                        logging.debug(
                            moduleName + " is disabled in the current run")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-v', '--verbose', default=False,
                        action='store_true',
                        help='Enable this switch if you want to print the' +
                        ' results to the terminal screen instead of to the' +
                        ' log file')

    parser.add_argument('-nr', '--noreport', default=True,
                        action='store_false',
                        help='Enable this switch if you do NOT want an HTML ' +
                             'file to be generated using the results.')

    parser.add_argument('-ep', '--enableparallelization', default=False, 
                        action='store_true',
                        help='Enable this switch if you want to enable ' +
                             'parallelization when executing the available' +
                             ' unittests.')

    args = parser.parse_args()

    suite = SpecificUnittestSuite()
    suite.populate_suite(args.enableparallelization)

    # finally, run the collected tests and either print the output to the
    # screen or generate the default HTML report
    if args.verbose:
        unittest.TextTestRunner(verbosity=2).run(suite)
    if args.noreport:
        try:
            import HTMLTestRunner
            import datetime
        except ImportError:
            print "Could not successfully import the necessary dependencies."
            print "Exiting...."
            exit()

        # per default, the report will be saved in a file named
        # ``run_unittests.html``.
        desc = ('This is the result of running all the available ' +
                       'unittests as of %s') % datetime.datetime.now()
        the_html = open("run_unittests.html", 'w')
        runner = HTMLTestRunner.HTMLTestRunner(stream=the_html,
                                               title='Specific unittests',
                                               description=desc)
        runner.run(suite)

        the_html.close()
        try:
            import webbrowser
            webbrowser.open("run_unittests.html")
        except:
            pass
