""" Import wrapper to get access to externally defined nodes

This module scans the list under the keyword ``external_nodes``
defined in the pySPACE configuration file
and inserts them into the normal structure and makes them available as it is
done for the internal nodes.

.. seealso::
    :ref:`default configuration file<conf>`

.. todo::
    Apigen cannot parse this module to find the containing classes,
    because it requires lines beginning with ``class ...`` to define a new
    class, which is then integrated into documentation.
    So currently, the layout is not the same as for the other files,
    but the one created by autosummary.
"""



import warnings
import pySPACE
import os
import sys
import re
import inspect
from os.path import expanduser
from pySPACE.missions.nodes.base_node import NodeMetaclass

home = expanduser("~")

# The pattern which python modules have to match
module_pattern = re.compile("[a-zA-Z0-9_][a-zA-Z0-9_]*.py$")

external_nodes = pySPACE.configuration.external_nodes
_NODE_MAPPING = {} # internal relevant variable for import

# mainly code from nodes init --> same variables used but not all needed
NODE_MAPPING = {}  # pySPACE.configuration.NODE_MAPPING
DEFAULT_NODE_MAPPING = {}  # pySPACE.configuration.DEFAULT_NODE_MAPPING

nodes = []

# list of files or nodes that should not be imported
# e.g. setup files may cause problems
file_name_black_list = ["setup.py"]

for root in pySPACE.configuration.external_nodes:
    root = os.path.expanduser(root)
    if not root in sys.path:
        sys.path.append(root)
    for dir_path, dir_names, file_names in os.walk(root,topdown=True):
        # Compute the package path for the current directory
        dir_path = dir_path[len(root)+1:]
        package_path = dir_path.replace(os.sep, ".")
        # Check all files if they are Python modules
        for file_name in file_names:
            if file_name in file_name_black_list:
                continue
            if module_pattern.match(file_name):
                # Import the module
                module_name = file_name.split(".")[0]
                if package_path == "":
                    module_path = module_name
                else:
                    module_path = package_path + '.' + module_name
                module = __import__(module_path, {}, {}, ["dummy"])
                module_nodes = inspect.getmembers(
                    module, lambda x: inspect.isclass(x) and
                    x.__name__.endswith("Node") and
                    x.__module__ == module.__name__)
                # If this module exports nodes
                if hasattr(module, "_NODE_MAPPING"):
                    # Add them to the global dict of nodes
                    for key, value in module._NODE_MAPPING.iteritems():
                        assert(key not in NODE_MAPPING.keys()), \
                            "External node with name %s at %s has already been defined at %s!" % (key, value, NODE_MAPPING[key])
                        NODE_MAPPING[key] = value
                        _NODE_MAPPING[key] = value #value.__name__
                for key,value in module_nodes:
                    # Nodes added the step before are allowed, but no other double entries
                    if key in NODE_MAPPING.keys():
                        assert(str(value)==str(NODE_MAPPING[key])), \
                            "External node (%s) with name %s has already been defined as %s!" % (str(value),key,str(NODE_MAPPING[key]))
                    if key[:-4] in NODE_MAPPING.keys():
                        assert(str(value)==str(NODE_MAPPING[str(key[:-4])])), \
                            "External node (%s) with name %s has already been defined as %s!" % (str(value),key[:-4],str(NODE_MAPPING[key[:-4]]))
                    DEFAULT_NODE_MAPPING[key] = value
                    NODE_MAPPING[key] = value
                    NODE_MAPPING[key[:-4]] = value
                    # import each node to shorten import path
                    exec "from %s import %s as %s" % (module_path, key, key)
                    note = "\n\n Method taken from: %s\n" % root
                    try:
                        exec "class %s(%s):\n    __doc__=%s.__doc__+note\n    pass" % (key,key,key)
                    except TypeError, e:
                        warnings.warn("Node %s has no documentation. You should fix this to get access." % key)
                    nodes.append(key)

# print _NODE_MAPPING
# for key in _NODE_MAPPING.keys():
#     print key, _NODE_MAPPING[key]
    #_NODE_MAPPING[key] = __import__(_NODE_MAPPING[key], {}, {}, ["dummy"])

#__all__ = nodes
# Clean up...
del(sys, os, re, module_pattern, NODE_MAPPING, DEFAULT_NODE_MAPPING,
    home, expanduser)
try:
    del(external_nodes, nodes, root, dir_path, dir_names, file_names,
        package_path, module_name, module_path, module, key, value, file_name)
except:
    pass
