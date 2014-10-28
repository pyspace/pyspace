#!/usr/bin/env python

""" Provides a class to implement a generic unittest

The unittests will only instantiate the given class with either
a default input set (see :mod:`~pySPACE.tests.utils.data.test_default_data`)
or will interpret the data given by the user. In the case that there
already is a specialized unittest available, this class will not be
called for that specific node.

:Author: Andrei Ignat, Mario Michael Krell
:Created: 2014/05/02
"""

# adding pySPACE to system path for import (code copy from launch.py)
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE') - 1]
if pyspace_path not in sys.path:
    sys.path.append(pyspace_path)

import pySPACE
pySPACE.load_configuration()

import_path = os.path.realpath(os.path.join(os.path.dirname(pySPACE.__file__),
                                            os.path.pardir))
import warnings
if not import_path == pyspace_path:
    warnings.warn("Check your python path! " +
                  "'%s' is the expected pySPACE path," % pyspace_path +
                  " but '%s' is used." % import_path)

# general imports
import unittest
from unittest import TestCase
import re

YAML_START = re.compile(r'(.. code-block:: yaml)', re.MULTILINE)


# special imports
import yaml
import pySPACE.missions.nodes
# If the script is called from the command line, it will run the generic
# unittest on all the available nodes
list_of_nodes = pySPACE.missions.nodes.DEFAULT_NODE_MAPPING

import pySPACE.missions.nodes.base_node as bn
from pySPACE.tests.utils.data.test_default_data import all_inputs


class ParametrizedTestCase(TestCase):

    """
    This class acts as a wrapper such that different parameters can
    be passed on to the test case. It is adapted from the example
    found here:

    http://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases/

    **Parameters**

        :methodName:
            The name of the test to be run
        :current_node:
            A reference to the node. Should reference the node object and
            NOT A STRING with the same name
        :specific_input:
            If a specific input is to be used, it should be defined here

    The reason behind the existence of this class is that passing parameters
    to a class that extends :class:`GenericTestCase` is close to impossible.
    If the user does not want to implement an external unittesting package such
    as `nose-parametrized <https://github.com/wolever/nose-parameterized>`_,
    one must 'fool' python and use a wrapper class such that the unit testing
    is done in a different class. The end result is that the unittest
    implementation is done within a class that has an initialization method
    which obviously accepts external parameters (such as different nodes) and
    which takes the actual tests from a class that inherits from
    :class:`GenericTestCase`.

    While this is not the most elegant solution, it was preferred over
    importing a new module which would have just enlarged the list of
    the framework dependencies.
    """

    def __init__(self, methodName='runTest',
                 current_node=None,
                 specific_input=None,
                 desired_output=None):
        super(ParametrizedTestCase, self).__init__(methodName)
        self.the_node = current_node
        self.input = specific_input
        self.output = desired_output
        self.longMessage = True

    @staticmethod
    def parametrize(current_testcase, node=None, input=None, output=None):
        """ Instantiate a new testcase for current_node """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(current_testcase)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(current_testcase(name,
                                           current_node=node,
                                           specific_input=input,
                                           desired_output=output))
        return suite

    def _nspaces(self, line):
        """ returns the indentation

        **Parameters**

            :line:
                string representation of the current line
        """
        return len(line) - len(line.lstrip())

    def _look_for_yaml(self, the_docstring):
        """ takes a string as an argument and looks for a YAML
        code block inside the string

        **Parameters**

            :the_docstring:
                the string in which the method should look for YAML code
        """
        inside_yaml = False
        first_after = False
        yaml_indentation = 0
        the_yaml = ""

        for line in the_docstring:
            if YAML_START.search(line):
                inside_yaml = True
                first_after = True
                yaml_indentation = self._nspaces(line)
                continue
            if first_after:
                first_after = False
                continue
            if inside_yaml and self._nspaces(line) > yaml_indentation:
                the_yaml += line + '\n'
            else:
                inside_yaml = False

        return the_yaml

    def _get_the_call(self):
        """ wrapper to get the YAML call """
        yaml = self._look_for_yaml(self.the_node.__doc__.split('\n'))
        return yaml

    def _initialize_node(self):
        """ initializes the node with the parameters from YAML """
        node_content = yaml.load(self._get_the_call())
        the_node = bn.BaseNode.node_from_yaml(node_content[-1])
        return the_node

    def _which_input(self, node):
        """  queries the node to find out what type of input it takes

        **Parameters**

            :node:
                Reference to the node that is currently tested
        """
        return node.get_input_types()

    def _trainable_execute(self):
        """ Execution call for trainable nodes """
        result_list = []
        the_node = self._initialize_node()
        # select only the input that the node accepts and run with it
        input_keys = self._which_input(the_node)
        # check if a specific input was defined
        if self.input is not None:
            if not isinstance(self.input, list):
                self.input = [self.input]
            inputs = self.input
        else:
            inputs = [all_inputs[key] for key in input_keys]
        for input_set in inputs:
            for single_input in input_set:
                # distinguish between supervised and unsupervised execution
                # and train the node
                if the_node.is_supervised():
                    the_node.train(single_input[0], single_input[1])
                else:
                    the_node.train(single_input[0])
            the_node.stop_training()

            for single_input in input_set:
                # execute the node on the data
                # If the script is called from the command line, it will run
                # the generic
                result = the_node.execute(single_input[0])

                self.assertEqual(type(result),
                                 the_node.get_output_type(
                                     type(single_input[0]),
                                     as_string=False),
                                 msg="Output type does not correspond to node"
                                     "specifications!")
                result_list.append(result)
            the_node = self._initialize_node()
        return result_list

    def _non_trainable_execute(self):
        """ Execution call for non-trainable nodes """
        result_list = []
        the_node = self._initialize_node()
        input_keys = self._which_input(the_node)

        if self.input is not None:
            if not isinstance(self.input, list):
                self.input = [self.input]
            inputs = self.input
        else:
            inputs = [all_inputs[key] for key in input_keys]

        for input_set in inputs:
            for single_input in input_set:
                the_node.execute(single_input[0])
            the_node = self._initialize_node()
            input_keys = self._which_input(the_node)

            for input_set in inputs:
                for single_input in input_set:
                    result = the_node.execute(single_input[0])
                    self.assertEqual(type(result),
                                     the_node.get_output_type(
                                         type(single_input[0]),
                                         as_string=False),
                                     msg="Output type does not correspond" +
                                         " to node specifications!")
                    result_list.append(result)
                the_node = self._initialize_node()
        return result_list


class OutputTestCase(ParametrizedTestCase):

    """ Serves as a wrapper  to test the output given the default input set

    **Principle**

        The OutputTestCase serves as a framework for quick checks of a node
        whereby, for the default input set, which can be found in ,
        :mod:`~pySPACE.tests.utils.data.test_default_data` , a user defined
        output is expected
    """

    def test_output(self):
        """compare the output with the expected output"""
        # first we check whether the input/output were defined by the user
        if self.input is not None:
            warnings.warn("This test is designed to work with the default " +
                          "input set. Switching to default input set")
            self.input = None
        if self.output is None:
            raise NotImplementedError("The output was not defined.")

        the_node = self._initialize_node()
        if the_node.is_trainable():
            result = self._trainable_execute()
        else:
            result = self._non_trainable_execute()

        if len(result) is 1:
            result = result[0]
            self.assertEqual(result, self.output,
                             msg="Computed output does not match" +
                                 " desired output")
        else:
            self.assertEqual(len(result), len(self.output),
                             msg="Different dimensions between computed" +
                                 " and desired output")
            for i in range(0, len(result)):
                self.assertEqual(result[i], self.output[i],
                                 msg="Computed output does not match " +
                                     "desired output at index:" + str(i))


class InputOutputTestCase(ParametrizedTestCase):

    """ Serves as a wrapper for an easy way to test the input-output relation

    **Principle**

        The InputOutputTestCase serves as a framework for quick checks of a
        node whereby, for a user defined input, a user defined output is
        desired.


    """

    def test_input_output(self):
        """obtain output for given input and compare it with desired output"""
        # first we check whether the input/output were defined by the user
        if self.input is None:
            raise NotImplementedError("The input was not defined.")
        if self.output is None:
            raise NotImplementedError("The output was not defined.")

        # now, we just run the same methods

        # test the execution of the node
        the_node = self._initialize_node()
        if the_node.is_trainable():
            result = self._trainable_execute()
        else:
            result = self._non_trainable_execute()

        if len(result) is 1:
            result = result[0]
            self.assertEqual(result, self.output,
                             msg="Computed output does not match" +
                                 " desired output")
        else:
            self.assertEqual(len(result), len(self.output),
                             msg="Different dimensions between computed" +
                                 " and desired output")
            for i in range(0, len(result)):
                self.assertEqual(result[i], self.output[0],
                                 msg="Computed output does not match " +
                                     "desired output at index:" + str(i))


class GenericTestCase(ParametrizedTestCase):

    """ Contains the methods and submethods needed to run the tests

        - whether the node has some sort of documentation
        - whether an exemplary call is present in the documentation
        - whether the node can be initialized with the default data set
        - whether the node can execute on the default data set

    An example of how to implement the generic tests in a node specific
    manner can be found under
    :mod:`~pySPACE.tests.unittests.nodes.feature_generation.test_time_domain_features`

    This example is also explained in detail in the tutorial file
    """

    def shortDescription(self):
        """ overwritten method that will display the node name as output"""
        doc = self._testMethodDoc
        message = str(self.the_node) + "\n"
        message += doc and doc.split("\n")[0].strip() or None
        return message

    def test_has_documentation(self):
        """ check if the node has some sort of documentation """
        self.assertNotEqual(self.the_node.__doc__, None)

    def test_has_exemplary_call(self):
        """ check if there is an exemplary call in the documentation """
        # if there is no YAML code, the string will be empty and thus False
        self.assertNotEqual("".join(self._get_the_call().split()), "",
                            msg="The node does not have an exemplary call")

    def test_initialize(self):
        """ check if the node can be initialized using the default data """
        # test if the node can be initialized
        self.assertRaises(Exception, self._initialize_node())

    def test_execution(self):
        """ execute the node using the default data

        **Principle**

            The test_execution method runs the node on the default data set
            and then checks whether the type of the output corresponds to the
            according output type from the theoretical point of view
        """
        # test the execution of the node
        the_node = self._initialize_node()
        if the_node.is_trainable():
            result = self._trainable_execute()
        else:
            result = self._non_trainable_execute()
        return result


def single_node_testing(node_name):
    """
    This function facilitates the testing of a single node referred to by using
    its representation in the DEFAULT_NODE_MAPPING variable.
    """
    # The output is sent to the console of the terminal
    stdout = sys.stdout

    stdout.write("\n>>>>> Single node testing: %s <<<<<\n" % (node_name))
    stdout.write('\n' + '*' * 70 + '\n' + str(node_name) + '\n' + '*' * 70 + '\n')

    # Initialize the test case
    suite = unittest.TestSuite()
    suite.addTest(ParametrizedTestCase.parametrize(
        current_testcase=GenericTestCase, node=list_of_nodes[node_name]))

    # Run the tests using the default TextRunner
    unittest.TextTestRunner(stream=stdout, verbosity=2).run(suite)


def multiple_node_testing(verbose=False, report=False):
    """
    This function ensures the testing of all available nodes.
    The results of the test are packed into an HTML file which is
    saved in the current working directory.
    """
    # we define a list of nodes that we do not want to test
    skipped_dirs = ['pySPACE.missions.nodes.sink',
                    'pySPACE.missions.nodes.source',
                    'pySPACE.missions.nodes.scikits_nodes',
                    'pySPACE.missions.nodes.splitter',
                    'pySPACE.missions.nodes.meta',
                    'pySPACE.missions.nodes.debug.subflow_timing_node',
                    'pySPACE.missions.nodes.classification.ensemble',
                    'pySPACE.missions.nodes.classification.svm_variants.sparse',
                    'pySPACE.missions.nodes.spatial_filtering.sensor_selection',
                    'pySPACE.missions.nodes.visualization.ensemble_vis',
                    'pySPACE.missions.nodes.classification.svm_variants.RMM'
                    ]

    skipped_nodes = ['FeatureNormalizationNode',
                     # this exemplary call has a hardcoded path in it
                     'ElectrodeCoordinationPlotNode',
                     'AverageFeatureVisNode',
                     'AlamgirMultiTaskClassifierNode',
                     'ICAWrapperNode',
                     # The iteration does not converge on the test data
                     # TODO:Build converging iteration
                     'JunctionNode',  # requires private modules
                     'LaplacianReferenceNode',
                     # node has hardcoded values
                     'PissfNode',  # TODO:needs specialized training data
                     # does not apply to default data
                     'MonoTimeSeries2FeatureNode'
                     ]
    # initialize some counters and log the results to a txt file
    total_tests, docu, exemplary, initialize, execution = 0, 0, 0, 0, 0

    stdout = sys.stdout

    if not verbose:
        # suppress all the different outputs from popping on screen
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')
        pySPACE.configuration.min_log_level = 1000

        # this list will be populated will unit testing suites for all the
        # available nodes
        the_report_suite = []

    count = 0

    for key, item in list_of_nodes.items():
        # we want to skip the nodes defined above
        skiptest = [item.__module__.startswith(x) for x in skipped_dirs]
        count += 1
        stdout.write('\r')


        if True in skiptest or item.__name__ in skipped_nodes:
            continue

        stdout.write('\r')
        # print an OS independent status message
        stdout.write(">>>>>>>>>>>>>>> Nodes tested already: %d. Currently testing %s "
                     "<<<<<<<<<<<<<<<" % (count, item.__name__))
        stdout.flush()

        # update the test count
        total_tests += 4

        if verbose:
            stdout.write('\n' + '*' * 70 + '\n' + str(key) +
                         '\n' + '*' * 70 + '\n')

        suite = unittest.TestSuite()
        suite.addTest(ParametrizedTestCase.parametrize(
            current_testcase=GenericTestCase, node=item))

        if verbose:
            result = unittest.TextTestRunner(stream=stdout, verbosity=2).run(suite)
        else:
            result = unittest.TextTestRunner(stream=open(os.devnull, 'w'), verbosity=2).run(suite)

        if report:
            the_report_suite.append((suite, key))

        # check which tests failed
        for item in result.failures:
            failed_test = str(item[0])
            if failed_test.startswith('test_has_documentation'):
                docu += 1
            elif failed_test.startswith('test_has_exemplary_call'):
                exemplary += 1
            elif failed_test.startswith('test_initialize'):
                initialize += 1
            elif failed_test.startswith('test_execution'):
                execution += 1

        for item in result.errors:
            failed_test = str(item[0])
            if failed_test.startswith('test_has_documentation'):
                docu += 1
            elif failed_test.startswith('test_has_exemplary_call'):
                exemplary += 1
            elif failed_test.startswith('test_initialize'):
                initialize += 1
            elif failed_test.startswith('test_execution'):
                execution += 1

    # either generate an HTML report or generate a matplotlib plot of
    # the results
    if report:
        try:
            import HTMLTestRunner
            import datetime
        except ImportError:
            print "Please download the HTMLTestRunner python script"

        the_html = open("generic_unittests.html", 'w')
        desc = ('This is the result of running the generic unit test on' +
                       ' all available nodes as of %s') % datetime.datetime.now()
        runner = HTMLTestRunner.HTMLTestRunner(stream=the_html,
                                               title='Generic unittest',
                                               description=desc)
        runner.run(the_report_suite)
        the_html.close()

        # if a webbrowser is available, open the report
        try:
            import webbrowser
            webbrowser.open("generic_unittests.html")
        except:
            pass
    else:
        # plot the results
        success = total_tests - docu - exemplary - initialize - execution
        import matplotlib.pyplot as plt
        import numpy as np
        plt.clf()
        plt.figure(125, figsize=(15, 15))
        colors = plt.cm.prism(np.linspace(0., 1., 5))

        patches = plt.pie([success, docu, exemplary, initialize, execution],
                          autopct='%2.2f%%', colors=colors,
                          pctdistance=1.1, labeldistance=0.5,
                          explode=[0.03, 0.10, 0.15, 0.03, 0.03])

        plt.legend(patches[0],
                   ["Successful (" + str(success) + ")",
                    "No documentation (" + str(docu) + ")",
                    "No exemplary call (" + str(exemplary) + ")",
                    "Initialization failed (" + str(initialize) + ")",
                    "Execution failed (" + str(execution) + ")"],
                   loc="best")

        plt.title("Total number of tests:" + str(total_tests))
        plt.savefig("generic_unittest_plot.pdf")
        plt.close()

        # some print statements with the results of the tests
        sys.stdout = stdout
        print "\n" + '*' * 70 + '\n' + "Test results" + '\n' + '*' * 70
        print "Successful tests: " + str(success) + "\n" + \
              '-' * 70 + '\n' + \
              "No documentation: " + str(docu) + "\n" + \
              "No exemplary call: " + str(exemplary) + "\n" + \
              "Initialization failed: " + str(initialize) + "\n" + \
              "Execution failed: " + str(execution) + "\n" + \
              '-' * 70 + '\n'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--verbose', default=False,
                        action='store_true',
                        help='Enable this switch if you want to print the' +
                        ' results to the terminal screen instead of to the ' +
                        ' log file')

    parser.add_argument('-sn', '--singlenode', default="",
                        help='If you want to test a single node, specify it ' +
                             'under the SINGLENODE variable')

    parser.add_argument('-r', '--report', default=False,
                        action='store_true',
                        help='Decides whether an HTML report should be ' +
                        'generated from the results of the unittest')

    args = parser.parse_args()

    # The single node execution is done by default under verbose mode
    if args.singlenode != "":
        single_node_testing(args.singlenode)
    else:
        multiple_node_testing(args.verbose, args.report)
