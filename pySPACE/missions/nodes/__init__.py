""" Nodes are elemental signal processing steps

They are arranged in mostly :mod:`serial node chains<pySPACE.environments.chains.node_chain>`.
Between the nodes in one chain several :mod:`signal types <pySPACE.resources.data_types>` can be sent:

    * :class:`~pySPACE.resources.data_types.time_series.TimeSeries` or
    * :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector` or
    * :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVector`
    * which all inherit from a common :class:`~pySPACE.resources.data_types.base.BaseData`.

There is one :class:`~pySPACE.missions.nodes.base_node.BaseNode`
for all nodes in the package module
and so every node only specifies the
relevant transformation methods.
A basic overview on the overloading of methods can be found in the
:mod:`~pySPACE.missions.nodes.templates` module.
All nodes are grouped in extra packages, depending on their main processing
category. Bigger differences in the algorithms' concepts are considered
with the module structure or even further subpackages.

The (Node Class) - (Configuration File) - (Node Name) Mapping
-------------------------------------------------------------

A complete list of all nodes and their mapping can be found at:
:ref:`node_list`.

The `__init__` of this package imports all existing nodes and maps their names,
given by the dict _NODE_MAPPING of each imported file,
to their class name.
These are the optional names.
The standard name of a node is the class name, which has to end with **Node**.
Furthermore you can use the class name without this ending as alternative name.
Every node has an exemplary call,
where you get to know the basic configuration structure.
And there is also a list of all possible usable names at the end of
the basic description of the node.
The mapping is used in the definition of :mod:`node chains <pySPACE.environments.chains.node_chain>`
when writing the :ref:`yaml<yaml>` configuration files.
Especially when comparing different algorithms it is useful to use
their short names as parameters to avoid long names in the folder names
and the final comparison graphics.

The advantage is, that most import statements are done at the beginning
and that the user just needs to know the methods name
and not the corresponding class and module name.
The disadvantage is, that maybe not used packages are imported or
that import errors occur, because certain packages are not installed.
The import errors are mainly prevented by throwing exceptions.

.. image:: ../../graphics/node.png
   :width: 500

.. seealso::

    - :ref:`node_list`
    - :mod:`pySPACE.environments.chains.node_chain`
    - :mod:`pySPACE.missions.operations.node_chain`
    - :class:`~pySPACE.missions.nodes.base_node.BaseNode`
    - :mod:`~pySPACE.missions.nodes.templates`

.. todo:: Find out, why the import of the base node is called four times.
"""

import sys
import os
import re
import inspect
import pySPACE

# The pattern which python modules have to match
module_pattern = re.compile("[a-zA-Z0-9_][a-zA-Z0-9_]*.py$")

# The root of the search (should be the nodes directory)
root = os.sep.join(__file__.split(os.sep)[:-1])


# load module import list
try:
    # get list of nodes that should be imported
    # if the list is empty, by default all nodes are imported
    _module_import_white_list_temp = \
        pySPACE.configuration.module_import_white_list
    # None is allowed -> map to empty list
    if _module_import_white_list_temp is None:
        _module_import_white_list_temp = []
    # if file names rather than module names are specified,
    # strip the file extension
    _module_import_white_list = []
    for module_or_filename in _module_import_white_list_temp:
        if module_or_filename.endswith(".py"):
            _module_import_white_list.append(module_or_filename.split(".")[0])
        else:
            _module_import_white_list.append(module_or_filename)

    # TODO: by default, all nodes specified under "external" are also imported
    # -> this is ugly, because it changes the semantics, find a better solution
    # without code duplication
    _module_import_white_list.append("external")
except:
    _module_import_white_list = []

# The global dict of nodes
try:
    NODE_MAPPING = pySPACE.configuration.NODE_MAPPING
    DEFAULT_NODE_MAPPING = pySPACE.configuration.DEFAULT_NODE_MAPPING
except:
    NODE_MAPPING = {}
    DEFAULT_NODE_MAPPING = {}
    pySPACE.configuration.NODE_MAPPING = NODE_MAPPING
    pySPACE.configuration.DEFAULT_NODE_MAPPING = DEFAULT_NODE_MAPPING

    # search all modules in the directory subtree rooted here
    for dir_path, dir_names, file_names in os.walk(root, topdown=True):
        # Compute the package path for the current directory
        dir_path = dir_path[dir_path.rfind("pySPACE"):]
        package_path = dir_path.replace(os.sep, ".")

        if package_path == "pySPACE.missions.nodes":
            if "Noop" in NODE_MAPPING.keys():
                continue # do not visit the base module twice
            else:
                # templates are no real nodes,
                # other components get extra treatment
                for file_name in ["templates.py",
                                  "templates.pyc"]:
                    try:
                        file_names.remove(file_name)
                    except:
                        pass
        # Check all files if they are Python modules
        filtered_file_names = []
        for file_name in file_names:
            if module_pattern.match(file_name):
                filtered_file_names.append(file_name)
        for file_name in filtered_file_names:
            # Import the module
            module_name = file_name.split(".")[0]
            if len(_module_import_white_list) > 0 and \
                    module_name not in _module_import_white_list:
                continue
            module_path = package_path + '.' + module_name
            module = __import__(module_path, {}, {}, ["dummy"])
            module_nodes = inspect.getmembers(module, \
                lambda x: inspect.isclass(x) and x.__name__.endswith("Node") \
                    and x.__module__==module.__name__)
            # If this module exports nodes
            if hasattr(module, "_NODE_MAPPING"):
                if module_path == "pySPACE.missions.nodes.external":
                    # Replace wrong value with new fitting one
                    for key, value in module._NODE_MAPPING.iteritems():
                        assert(key not in NODE_MAPPING.keys()), \
                            "Node with name %s at %s has already been defined at %s!" % (key, value, NODE_MAPPING[key])
                        for new_key, new_value in module_nodes:
                            if new_value.__name__ == value.__name__:
                                NODE_MAPPING[key] = new_value
                                break
                else:
                    # Add them to the global dict of nodes
                    for key, value in module._NODE_MAPPING.iteritems():
                        assert(key not in NODE_MAPPING.keys()), \
                            "Node with name %s at %s has already been defined at %s!" % (key, value, NODE_MAPPING[key])
                        NODE_MAPPING[key] = value
            for key, value in module_nodes:
                if file_name is not "base_node.py":
                    try:
                        value.get_input_types()
                    except NotImplementedError:
                        pass
                # Nodes added the step before are allowed,
                # but no other double entries
                if key in NODE_MAPPING.keys():
                    assert(str(value)==str(NODE_MAPPING[key])), \
                        "Node (%s) with name %s has already been defined as %s!" % (str(value),key,str(NODE_MAPPING[key]))
                if key[:-4] in NODE_MAPPING.keys():
                    assert(str(value)==str(NODE_MAPPING[str(key[:-4])])), \
                        "Node (%s) with name %s has already been defined as %s!" % (str(value),key[:-4],str(NODE_MAPPING[key[:-4]]))
                DEFAULT_NODE_MAPPING[key] = value
                NODE_MAPPING[key] = value
                NODE_MAPPING[key[:-4]] = value
                # import each node to shorten import path
                exec "from %s import %s" % (module_path,key)

    # If sklearn is available, add wrapper-nodes for sklearn estimators.
    try:
        import scikit_nodes
    except ImportError:
        pass

    DEFAULT_NODE_MAPPING["BaseNode"].input_types=["TimeSeries", 
        "FeatureVector", "PredictionVector"]

    # Clean up...
    del(module_pattern, root, dir_path, dir_names, file_names,
        package_path, module_name, module_path, module, key, value, file_name)

# Clean up...
del(sys, os, re, inspect,pySPACE)


