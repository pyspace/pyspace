
.. _t_install:

Installation
------------

.. warning::
    Currently the software is developed under Ubuntu and MacOSX. It is highly 
    recommended to use a UNIX OS for pySPACE. However it is possible to run
    `pySPACE on Windows`_.

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
Besides the standard Python library, some extra packages/modules are needed
that you have to download/install yourself. It is recommended to use a package 
manager to install these packages (e.g. apt or `macports <http://www.macports.org/>`_).

Mandatory packages are:

    :Python2.7: http://www.python.org/ (the main programming language)

                The software also works with Python2.7.
                It is important to install the relevant Python packages
                for the same Python and to finally start it with this
                version.
    :YAML:      http://www.yaml.org/

                for reading and writing configuration files
    :enum:      http://cheeseshop.python.org/pypi/enum/

                for robust enumerations in Python
    :NumPy:     http://www.numpy.org/

                basic array handling library to handle data
    :SciPy:     http://www.scipy.org/

                more complicated signal processing or linear algebra operations
                
    .. code-block:: bash
    
        # -- for Macport users --
        $ sudo port install python27
        # select correct Python version
        $ sudo port select --set python python27
	# install other mandatory packages
        $ sudo port install py27-yaml py27-enum py27-numpy py27-scipy

Optional packages are:

    :matplotlib:  http://matplotlib.org/

                  making fancy plots
    :scikit-learn:
                  http://scikit-learn.org/

                  Many scikit algorithms are available wrapped via the
                  :mod:`~pySPACE.missions.nodes.scikit_nodes`
                  module and can be used like normal nodes.

    :Sphinx1.4: Generate documentation

    :PyQt4:       basis of the :mod:`guis <pySPACE.run.gui>`
    :LIBSVM:      http://www.csie.ntu.edu.tw/~cjlin/libsvm/

                  Famous library for fast SVM classifiers.
                  Be careful that the Python bindings are installed correctly.
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

pySPACE on Windows
^^^^^^^^^^^^^^^^^^^

pySPACE can also be used under a Windows operating system. The first and most
important dependency that must be installed is a ``python`` bundle that
(preferably) comes with a large part of the heavy-weigth dependencies needed by
pySPACE, e.g. ``numpy``, ``matplotlib`` etc. The other two must-have
dependencies are a ``git`` management software and either a text editor that
can edit ``.py`` and ``.yaml`` files or an IDE that can do the same thing.
Below is a list of the software we recommend using when building and running
pySPACE:

	- Anaconda Python http://continuum.io/downloads
	- some sort of github version (we recommend this one http://msysgit.github.io/)
	- a python editor (we recommend pycharm http://www.jetbrains.com/pycharm/)

Besides these, depending on the specific nodes that you plan on using, further
dependencies might be neccessary. It is worth noting that, when the setup
script is launched, it might blacklist some of the available nodes due to
missing dependencies. Should you install these missing dependencies, you can
refresh the list of missing dependencies by running the setup script with the
``-b`` option enabled::

	python setup.py -b

This will refresh the blacklisted nodes and overwrite the previous list.
For more details related to the setup script, please run the help option of the
setup script as::

    python setup.py -h

It should be noted that pySPACE is developed using under and for UNIX systems.
Therefore, the Windows support for the software suite is limited. Nonetheless,
the basic functionalities of the software are available under a DOS operating
system and, depending on the availabilty of the necessary dependencies, can be
used to its fullest on DOS-operating systems.

.. note::
    When writing new nodes, special attention should be given to OS-independent
    implementations. As an example of this approach, the `numpy.float128`
    precision is not available under DOS-systems. There is however an
    alternative precision floating point, namely the `numpy.longdouble` that
    serves the same purpose yet is OS-independent. While this is merely an
    example that is meant to show the motivation behind an OS-independent
    approach, the general idea is the same for different issues. If there is
    an OS-independent approach, it should be favored in the implementation
    process.

.. note::
    Another example of a very important OS-independent approach is that of
    using `os.sep` from the python :mod:`os` module whenever file paths are
    in usage. Since DOS and UNIX systems use different path separators, it
    is of the utmost importance that whenever new nodes are written and some
    sort of path manipulation is necessary, the separator be obtained from
    `os.sep` and not be hardcoded in the python script.

Terminal usage
++++++++++++++

Under Windows, pySPACE can be used from the terminal in the same manner
as one would under a UNIX system. Thanks to OS-independent python shortcuts,
once you have prepaired your data and operation chain, you can save and
execute them from pySPACEcenter. While pySPACE is primarily a UNIX oriented
software package, the contributors strive to build OS-independent python
scripts.

Writing new nodes
+++++++++++++++++

Should you want to develop new pySPACE nodes under Windows, please be aware
that there are certain software packages that only run on DOS (Windows) systems
while others run only on UNIX (Mac, Linux) systems. These OS-specific packages
should be avoided as much as possible. If such a package is absolutely
necessary, please consider implementing it in a ``if`` clause that first
establishes the OS under use and then chooses the appropriate method of
implementation. In most cases though there is an OS-independent implementation
which is definitely the preferred version.

Next Steps
^^^^^^^^^^

After the installation you might want to

    * read :ref:`some basic introduction<getting_started>` or
    * play around with a
      :ref:`first basic benchmarking example <first_operation>`.
