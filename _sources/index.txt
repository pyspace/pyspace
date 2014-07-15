.. documentation master file, created by sphinx-quickstart on Fri Jan 30 13:24:06 2009.

.. _pySPACE:

pySPACE
=======

For getting a basic introduction into pySPACE and its main principles,
we refer to `the corresponding paper in the Frontiers in Neuroinformatics Journal <http://www.frontiersin.org/neuroinformatics/10.3389/fninf.2013.00040/abstract>`_.
This paper should be also used to :ref:`cite pySPACE <cite>`.

.. table:: **Quick Access**

    +---------------------------+---+---------------------------+---+---------------------------+
    |                           | # |                           | # |                           |
    +---------------------------+---+---------------------------+---+---------------------------+
    |:ref:`welcome`             |   |:ref:`download`            |   |:ref:`t_install`           |
    |                           |   |                           |   |                           |
    |Introduction               |   |How to Get it?             |   |Dependencies & Setup       |
    +---------------------------+---+---------------------------+---+---------------------------+
    |                           | # |                           | # |                           |
    +---------------------------+---+---------------------------+---+---------------------------+
    |:ref:`overview`            |   |:ref:`getting_started`     |   |:ref:`tutorials`           |
    |                           |   |                           |   |                           |
    |Data, Processing &         |   |Launch pySPACE             |   |Learn About Usage          |
    |Modalities                 |   |                           |   |                           |
    +---------------------------+---+---------------------------+---+---------------------------+
    |                           | # |                           | # |                           |
    +---------------------------+---+---------------------------+---+---------------------------+
    |:ref:`content`             |   |:ref:`missions`            |   |:ref:`FAQ <faq>`           |
    |                           |   |                           |   |                           |
    |Short & Long               |   |Algorithm Overview         |   |Frequently Asked Questions |
    +---------------------------+---+---------------------------+---+---------------------------+
    |                           | # |                           | # |                           |
    +---------------------------+---+---------------------------+---+---------------------------+

|

.. _welcome:

Welcome
-------

pySPACE is a **Signal Processing And Classification Environment** (SPACE)
written in `Python <http://www.python.org/>`_ interfacing to the user
with :ref:`YAML configuration files<yaml>` and enabling parallel process
execution (for all of these reasons we put a small *py* in front).
pySPACE allows rapid specification, execution,
and analysis of empirical investigations (short: benchmarking)
in signal processing and machine learning.
Besides the *benchmarking* way of executing pySPACE where you can evaluate your
data with your own configuration of algorithms,
the software also provides a :mod:`~pySPACE.run.launch_live` mode
where you can directly execute signal processing as soon as you have the data
in an online fashion.

If you take a look at the :ref:`mainstructure` of pySPACE you should be able to
directly see how you can interact with the software.
Besides this, user data (input, output and processing definitions)
are stored in your pySPACEcenter (until configured otherwise). You can
see more when looking at the :ref:`getting_started` page.

In pySPACE the parallelization part is not restricted to signal processing and
machine learning, though these are the only use cases currently supported.
By defining your own :mod:`~pySPACE.missions.operations`,
you can interface any library with pySPACE, using different
parallelization modes. pySPACE already provides interfaces to popular toolkit
libraries like the `Weka framework <http://www.cs.waikato.ac.nz/ml/weka/>`_,
and the `MMLF <http://mmlf.sourceforge.net/>`_ but
also comes with a lot of own algorithms,
though some algorithms are just wrapper to methods from other libraries like
the Modular toolkit for Data Processing
(`MDP <http://mdp-toolkit.sourceforge.net/>`_),
`scikit-learn <http://scikit-learn.org>`_,
`LIBSVM <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_ or
`LIBLINEAR <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_.

This software can be used to analyze and compare the performance
(e.g., classification accuracy) of different methods and parameter settings, respectively.
It allows to estimate these quantities based on different validation schemes
(including crossvalidation) and several
independent runs. Furthermore, it allows to make use of the fact that most of
these problems can be handled independently and allows to process subtasks in parallel on different
:mod:`~pySPACE.environments.backends`. For instance the
:class:`~pySPACE.environments.backends.multicore.MulticoreBackend`
allows to run as
many tasks in parallel as cores are present in the respective machine.

pySPACE supports the massive parallel execution of benchmarking experiment on grids
like systems using
`LoadLeveler <http://www-03.ibm.com/systems/software/loadleveler/>`_
or `MPI <http://www.mcs.anl.gov/research/projects/mpi/>`_
sharing a common data access.

This software has an object-oriented design,
providing classes for important entities like:

    * :ref:`datasets`,
    * :ref:`processing`, and
    * :ref:`modality`,

which define the structure of the data, how it is changed and which
parallelization mode is used.

.. _mainstructure:

Main Software Structure
-----------------------

.. currentmodule:: pySPACE
.. autosummary::

    environments
    missions
    resources
    run
    tests
    tools

Indices and Meta Information
----------------------------
.. table::

   +----------------------------+---+---------------------------+---+---------------------------+
   |                            | # |                           | # |                           |
   +----------------------------+---+---------------------------+---+---------------------------+
   |:ref:`genindex`             |   |:ref:`api`                 |   |:ref:`node_list`           |
   |                            |   |                           |   |                           |
   |The Complete Index          |   |Learn About the Structure  |   |All Elemental Algorithms   |
   +----------------------------+---+---------------------------+---+---------------------------+
   |                            | # |                           | # |                           |
   +----------------------------+---+---------------------------+---+---------------------------+
   |:ref:`glossary`             |   |:ref:`history`             |   |:ref:`doc`                 |
   |                            |   |                           |   |                           |
   |pySPACE Vocabulary          |   |Learn About the Background |   |The Doc of the Doc         |
   |                            |   |                           |   |                           |
   +----------------------------+---+---------------------------+---+---------------------------+
   |                            | # |                           | # |                           |
   +----------------------------+---+---------------------------+---+---------------------------+
   |:ref:`license`              |   |:ref:`credits`             |   |:ref:`Contact`             |
   |                            |   |                           |   |                           |
   |GPLv3                       |   |Some Contributors          |   |Praise? Curse? Report?     |
   +----------------------------+---+---------------------------+---+---------------------------+
   |                            | # |                           | # |                           |
   +----------------------------+---+---------------------------+---+---------------------------+

.. _contact:

Contact
-------

If you want to be kept informed about current changes in the framework,
add yourself to the user mailing list.

For development discussions and requests or if you want to share with us
successful software usage, contact us via the developer mailing list.

We are thankful for everybody who wants to :ref:`contribute<CLA>` and you can feel free
to join the developer mailing list to support us.

Both lists are moderated.

User mailing list : Ric-pyspace-user@dfki.de

Developer mailing list : Ric-pyspace-dev@dfki.de

In future communication might switch to official tools on GitHub or SourceForge.

Documentation TOC
-----------------

.. toctree::
    :maxdepth: 2

    content

Search
------

   - :ref:`search`
   - :ref:`genindex`
   - :ref:`modindex`
   - :ref:`node_list`

Disclaimer
----------

.. warning::
    !! IMPORTANT INFORMATION !!

    This software including all extensions and accessories is neither designed nor
    suitable for medical or diagnostic purposes.
    This software must not be used as a medical product according
    to appendixes 1 and 2 of the medical product operator regulation
    (Medizinprodukte-Betreiberverordnung).

    !! WICHTIGER HINWEIS !!

    Die vorliegende Software einschließlich aller Erweiterungen und Zubehör ist 
    für medizinische oder diagnostische Zwecke nicht bestimmt oder geeignet.
    Sie darf nich mit Zweckbestimmung eines Medizinproduktes
    im Sinne der Anlagen 1 und 2 der Medizinprodukte-Betreiberverordnung
    eingesetzt werden.