
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
    :numpy:     http://www.numpy.org/

                basic array handling library to handle data
    :scipy:     http://www.scipy.org/

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
    :scikit:      http://scikit-learn.org/

                  Many scikit algorithms are available wrapped via the
                  :mod:`~pySPACE.missions.nodes.scikits_nodes`
                  module and can be used like normal nodes.
    :pyqt4:       basis of the :mod:`guis <pySPACE.run.gui>`
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



   
