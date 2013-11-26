
.. _t_install:

Installation
------------

.. warning::
    Currently the software is maintained only under Ubuntu and MacOSX.

pySPACE is currently developed at the `RIC Bremen <http://robotik.dfki-bremen.de/>`_
(part of the DFKI GmbH) and at the `AG Robotik <http://robotik.dfki-bremen.de/>`_
of the University Bremen.

The installation process consists of several steps:

1. :ref:`download` and install pySPACE itself
2. Install the needed extra packages/modules
3. Prepare the pySPACE user environment

1. Downloading
^^^^^^^^^^^^^^^

Step 1 is quite easy: :ref:`download` the software.
Currently, there is no real installation script implemented, though this
will hopefully change in future.
So simply save the software, where you want and where you can find it easily.
For interactively using the software it might be also useful
to add the folder path to the PYTHONPATH variable.
For the normal usage this is not required.

2. Extra packages
^^^^^^^^^^^^^^^^^^
Besides the standard python library, some extra packages/modules are needed
that you have to download/install yourself. It is recommended to use a package 
manager to install these packages (e.g. apt or `macports <http://www.macports.org/>`_).

Mandatory packages are:

    :Python2.6: http://www.python.org/ (the main programming language)

                The software also works with Python2.7.
                It is important to install the relevant Python packages
                for the same Python and to finally start it with this
                version.
    :YAML:      http://www.yaml.org/

                for reading and writing configuration files
    :NumPy:     http://www.numpy.org/

                basic array handling library to handle data
    :SciPy:     http://www.scipy.org/

                more complicated signal processing or linear algebra operations
                
    .. code-block:: bash
    
        # -- for Macport users --
        $ sudo port install python26
        # select correct python version
        $ sudo port select --set python python26
        $ sudo port install py26-yaml
        # use macports own atlas package
        $ sudo port install py26-numpy +atlas
        $ sudo port install py26-scipy +atlas

Optional packages are:


    :matplotlib:  http://matplotlib.org/

                  making fancy plots
    :scikit-learn:
                  http://scikit-learn.org/

                  Many scikit algorithms are available wrapped via the
                  :mod:`~pySPACE.missions.nodes.scikits_nodes`
                  module and can be used like normal nodes.
    :PyQt4:       basis of the :mod:`guis <pySPACE.run.gui>`
    :LIBSVM:      http://www.csie.ntu.edu.tw/~cjlin/libsvm/

                  Famous library for fast SVM classifiers.
                  Be careful that the python bindings are installed correctly.
                  If you have access use modified version in external repository.
    :CVXOPT:      http://abel.ee.ucla.edu/cvxopt/

                  optimization toolbox, used for the construction of some classifiers,
                  which are described by mathematical programs
    :mpi4py:      http://mpi4py.scipy.org/

                  needed, if you want to use the
                  :class:`~pySPACE.environments.backends.mpi_backend.MpiBackend`
    :MDP:         http://mdp-toolkit.sourceforge.net/ (tested up to version 3.1)

                  currently needed only for PCA, ICA, FDA but more could be integrated
    :external:    collection of slight modifications by the pySPACE developers
                  of existing libraries, e.g. LIBSVM
    :mlpy:        For one feature selecting node.
    :tables:      Handle hierarchic data
    :. . .:       and whatever you want to integrate

                  .. warning:: When programming, always make imports
                               to the aforementioned packages optional
                               and used only in your module and add them here.





The User Environment
^^^^^^^^^^^^^^^^^^^^^

To provide the software with necessary information,
this software needs one :ref:`main configuration file<conf>` and several
further :ref:`specification files <spec>`.
The :ref:`main configuration file<conf>` specifies, were to find the other
:ref:`specification files <spec>`, were to load and store data and a lot more.

Easy Setup of Environment
+++++++++++++++++++++++++

The default structure of your environment is called the pySPACEcenter.
It is installed into your home-directory by calling

.. code-block:: bash

        python setup.py

in the software folder. This will create the folder `pySPACEcenter` and the
relevant subfolders including the relevant default main configuration file and
several examples.

Furthermore, links to the three main routines for using the software will be
created in this folder:

.. currentmodule:: pySPACE.run
.. autosummary::

    launch
    launch_live
    gui.performance_results_analysis

.. warning:: The name ``setup.py`` was chosen for future development,
             where this script will be also responsible for a real installation
             including the installation of dependencies and moving the
             needed code to the site-packages folder.
             So after running this script, pySPACE is probably not available
             in IPython or for the import in other software or scripts without
             additional effort.

Customized Setup of Environment
+++++++++++++++++++++++++++++++

Of course you can use your own locations for every part.
This is done by adapting your
`PYSPACE_CONF_DIR` and your configuration files.
Further details on setting up your main configuration file,
can be found :ref:`here<conf>`

The Default Configuration File
++++++++++++++++++++++++++++++

Here is a documented example of the default configuration file:

.. literalinclude:: ../examples/conf/example.yaml
    :language: yaml

Next Steps
^^^^^^^^^^

After the installation you might want to

    * read :ref:`some basic introduction<getting_started>` or
    * play around with a
      :ref:`first basic benchmarking example <first_operation>`.
