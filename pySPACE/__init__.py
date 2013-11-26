"""
.. _api:

pySPACE API
===========

Welcome to the main module of pySPACE, the Signal Processing And Classification Environment
in Python. For a complete and up-to date documentation of the release,
we refer to
`our git project web page <http://pyspace.github.io/pyspace/index.html>`_.

This module contains the basic imports for using pySPACE and sets the default
configuration.

**Documentation**

Documentation is done with `Sphinx <http://sphinx.pocoo.org/>`_
and some helper functions coming with the software for more customization.
The folder that contains all the documentation is called ``docs``.
To compile you first have to install Sphinx 1.1 or a better version.
The documentation can be created by running ``make html`` in the ``docs`` directory
(therefore we have a ``Makefile`` in the ``docs`` folder).
To view the documentation open the ``index.html`` in the ``.build/html`` folder.

**Structure of the Software and First Steps**

This software is structured into six main parts on the top level which give you a hint of 
where you might like to go. pySPACE is a highly modular framework, so all possible
tasks and algorithms are called :mod:`~pySPACE.missions`, input and output are defined
as :mod:`~pySPACE.resources`.

To start the software, go to the :mod:`~pySPACE.run` package, to get an idea of
what you can do, see the documentation of :mod:`~pySPACE.missions` (nodes and operations) 
and the documentation in :ref:`getting_started`, the :ref:`overview` and maybe some :ref:`tutorials`.

**Where to Go to Integrate Your Own Extensions**

If you want to integrate your own application, you will probably have to create new
:mod:`~pySPACE.missions`. If your algorithm can handle single :mod:`~pySPACE.resources.data_types`
or only works with one :mod:`dataset <pySPACE.resources.dataset_defs>`
or a combination of training and test set, you can integrate it easily by defining it in the
:mod:`~pySPACE.missions.nodes` package.
Otherwise you will have to integrate it into the
:mod:`~pySPACE.missions.operations` package, which is also a subpackage of the
:mod:`~pySPACE.missions` package.
Last but not least, you may want to write only a wrapper for your algorithm and
even implement your algorithm in a different language like C++ or even Matlab.
So the wrapper is integrated as mentioned before and the real algorithm
implementation can be done in the :mod:`~pySPACE.missions.support` package.

If you should realize, that this software is unable to load, store or
process your special type of data, you should have a look into the
:mod:`~pySPACE.resources` package.
Here different stages of accumulation of data are defined,
but most probably, you will be interested in
:mod:`~pySPACE.resources.dataset_defs`.
Very likely you will not have to define a complete new dataset type but
only add the functionality you need to the existing ones.

When integrating new components you should also write small
:mod:`~pySPACE.tests.unittests` in :mod:`~pySPACE.tests`.

In case you need some of the :mod:`~pySPACE.tools` please have a look at the 
respective documentation, too. This is also true for 
:mod:`~pySPACE.environments` similar holds, e.g., if you want to implement
new :mod:`backend <pySPACE.environments.backends>`
or have an own :mod:`~pySPACE.environments.live` application.

The following methods constitute the interface of the package:

    .. currentmodule:: pySPACE.environments.big_bang.Configuration
    .. autosummary::
        :nosignatures:

        load_configuration

    .. currentmodule:: pySPACE.missions.operations.base
    .. autosummary::
        :nosignatures:

        create_operation_from_file
        create_operation

    .. currentmodule:: pySPACE.run.launch
    .. autosummary::
        :nosignatures:

        run_operation
        run_operation_chain

    .. currentmodule:: pySPACE.environments.backends.base
    .. autosummary::
        :nosignatures:

        create_backend
"""

import os
from pySPACE.environments.backends.base import create_backend
from pySPACE.environments.big_bang import Configuration
from pySPACE.environments.chains.operation_chain import create_operation_chain
from pySPACE.missions.operations.base import create_operation_from_file, create_operation
from pySPACE.run.launch import run_operation, run_operation_chain

file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep + os.path.pardir
# Setup configuration
if os.environ.get('PYSPACE_CONF_DIR', '') == '':
    # if PYSPACE_CONF_DIR is unset or empty assume default configuration  directory
    configuration = Configuration(os.path.abspath(file_path))
else:
    # set conf_dir explicitly
    configuration = Configuration(os.path.abspath(file_path),
                                     conf_dir=os.environ['PYSPACE_CONF_DIR'])

load_configuration = configuration.load_configuration

del(file_path)

__version__ = '0.5'
__short_description__ = \
    "pySPACE is a **Signal Processing And Classification Environment** "+\
    "(which is the reason for the name, which is an abbreviation). "+\
    "It allows rapid specification, execution, "+\
    "and analysis of empirical investigations (short: benchmarking) "+\
    "in signal processing and machine learning. "+\
    "Therefore it combines parallelization, YAML and Python."
#__all__ = ["load_configuration", "create_backend", "create_operation_from_file",
#           "create_operation", "create_operation_chain", "run_operation",
#           "run_operation_chain"]

