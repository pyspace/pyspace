#!/usr/bin/env python
"""
This module checks whether the unittests for every nodes exist.
If [-l] option is specified, the result will be printed into log file 
'check_unittests_logger_YYYY-MM-DD_HH-MM-SS.log' in the current directory.

**Options**
    :-l: -- prints the result into log file
    :-log: -- prints the result into log file

:Author: Titiruck Nuntapramote (titiruck.nuntapramote@dfki.de)
:Created: 2012/01/03
"""
import os
import inspect
import sys
import logging
import datetime

if __name__ == '__main__':
    file_path = os.path.dirname(os.path.abspath(__file__))
    pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
    if not pyspace_path in sys.path:
        sys.path.append(pyspace_path)


import pySPACE.missions.nodes as nodes
import pySPACE.resources.data_types as data_types

def existing_test (lst, dir, files):
    """ Function to get a list of existing test
    
    **Arguments**
        :lst: -- The test suite, to which the tests are added.
        :dir: -- The directory which contains the files
        :files: -- List of files 

    """
    for file in files:
        # add python all files, if they start with test_ 
        #if re.match(r"test_.*\.py$",file) and os.path.isfile(dir+'/'+file):
            
            # figure out module name
            moduleName = dir.strip("\.\/").replace("/",".")
            if len(moduleName):
                moduleName += "."
            if os.path.splitext(file)[1] == '.py':
                moduleName += os.path.splitext(file)[0]
                print moduleName
                __import__(moduleName)
                module = sys.modules[moduleName]
                for name, obj in inspect.getmembers(module, inspect.isclass):
                         if name.endswith('TestCase'):
                            name = name[:-8]
                            #print name + ": " + moduleName
                            lst[name] = moduleName

def get_class (module, lst):
    """ 
    Function to get a list of class.
    For now, this is only used to get classes in datatype.
    
    **Arguments**
        :module: -- Module to get classes from
        :lst: -- list of classes
    
    """
    for name, obj in inspect.getmembers(module, inspect.ismodule):
        __import__(obj.__name__)
        new_module = sys.modules[obj.__name__]
        for name2, obj2 in inspect.getmembers(new_module, inspect.isclass):
            #print name2 + ":" + str(obj2)
            lst[name2] = obj2

if __name__ == '__main__': 
    
    log = False
    
    if len(sys.argv) == 2:
       arg = sys.argv[1]
       if arg == '-l' or arg == '-log':
           log = True
       else:
           print "usage: run_tests.py [-l|-log](enable logging)"
           exit()
    elif len(sys.argv) > 2:
        print "usage: run_tests.py [-l|-log](enable logging)"
        exit()
    
    if log is True:
        dateTag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        LOG_FILENAME = "check_unittests_logger_%s.log" % dateTag
        FORMATTER = '%(message)s'
        logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO, format=FORMATTER)
 
    
    # list for nodes
    nodelist = nodes.DEFAULT_NODE_MAPPING
    
    #collect classes from pySPACE.resources.data_types
    typelist = {}
    get_class(data_types,typelist)
    
    testnode = {}
    testtype = {}
    
    # list of node class with no unittests
    NO_testnode = {}
    NO_testtype = {}
    
    # the list HAVE_test* is not the same as test*
    # because the matching of node names and node class is not 1 to 1
    # some node class e.g. FIRFilterNode exists as BandPassFilter, LowPassFilter, and more
    HAVE_testnode = {}
    HAVE_testtype = {}
    
    #collect test cases from nodes
    os.path.walk(os.curdir+'/unittests/nodes', existing_test, testnode)
    
    #collect test cases from data_types
    os.path.walk(os.curdir+'/unittests/data_types', existing_test, testtype)

    
    
    #compare tests with nodes and then print
    print "Nodes without test ------------------------------------------"
    
    for key, value in sorted(nodelist.iteritems(),key=lambda t: str(t[1])):
        # -6 comes from deleting "Node'>"
        if (str(value).split('.')[-1])[:-6] not in testnode:
            NO_testnode[key] = value
            #print "\t" + key + ": " 
            print "\t\t" + str(value).split('\'')[1]
        else:
            HAVE_testnode[key] = value

    print "Existing nodes: ", len(nodelist)
    print "Nodes with test: ", len(HAVE_testnode), "(actual existing unittests:", len(testnode), ")"
    print "Nodes without test: ", len(NO_testnode)
    
    #compare tests with data_types and then print
    print "Types without test ------------------------------------------"
    for key, value in typelist.iteritems():
        if key not in testtype:
            NO_testtype[key] = value
            print "\t" + key + ": " 
            print "\t   " + str(value).split('\'')[1]
        else:
            HAVE_testtype[key] = value
    print "Existing types: ", len(typelist) 
    print "Types with test: ", len(HAVE_testtype)
    print "Types without test: ",len(NO_testtype)
    print "-------------------------------------------------------------"
    print "Total existing tests: ", len(testnode)+len(testtype)
    
    #if logging is enabled, print to log
    if log is True:
        logging.info("Nodes without test -----------------------------------")
        for key, value in NO_testnode.iteritems():
            logging.info("\t" + key + ": " + str(value)[22:-2])
        logging.info("\tExisting nodes: " + str(len(nodelist)))
        logging.info("\tNodes with test: " + str(len(HAVE_testnode)) + "(actual existing tests: " + str(len(testnode)) + ")")
        logging.info("\tNodes without test: " + str(len(NO_testnode)))
        logging.info("Types without test -----------------------------------")
        for key, value in NO_testtype.iteritems():
            logging.info("\t" + key + ": " + str(value)[22:-2])
        logging.info("\tExisting types: " + str(len(typelist)))
        logging.info("\tTypes with test: " + str(len(HAVE_testtype)))
        logging.info("\tTypes without test: " + str(len(NO_testtype)))
        
    
