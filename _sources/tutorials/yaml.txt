.. _yaml:

Using YAML Syntax in pySPACE
------------------------------------


This chapter will give an overview on YAML ("YAML Ain't Markup Language")
and it's usage in the framework.
The focus will be on a general description
and the use in node chain specification files,
but there will also be hints to
configuration and  other specification files.
A standard example of a YAML specification file for a node chain
can be found at the end of this tutorial.

General Information on YAML
###########################

A complete description on YAML 
in general can be found at: http://www.yaml.org,
where you should have a look at `the specification website <http://www.yaml.org/spec/1.2/spec.html>`_.
There you can find the general description:
`YAML™ (rhymes with “camel”) is a human-friendly, cross language, 
Unicode based data serialization language designed 
around the common native data types of agile programming languages. 
It is broadly useful for programming needs 
ranging from configuration files to Internet messaging 
to object persistence to data auditing.`

The usage of comments in YAML is the same as in Python with a hash ("#")
at the beginning of each comment line.
For the entries in your YAML file you have always to keep in mind,
that if you do not define the type (called *untagged* in YAML framework),
the YAML loading program tries to detect it automatically
and if it can not detect it, the entry will be tagged as a string.
Strings can also be defined by using double or normal quotation marks.
YAML can detect for example:

    * integers (e.g. ``42``),
    * floats (e.g. ``42.42``),
    * null (``None`` in Python, Syntax: use ``Null``, ``NULL`` or nothing),
    * string (e.g. ``"fourtytwo"`` or ``'fourtytwo'``),
    * lists (e.g. node chain specification file),
    * dictionaries (e.g. operation specification file),
    * booleans (``True`` and ``False``),
    * and much more, which is yet not relevant for this software framework.

For the framework it is mainly important to know, how to define something as a string.
If you are further interested in explicit tagging and new data types,
just check the `the specification website <http://www.yaml.org/spec/1.2/spec.html>`_.

.. warning:: YAML does not support all float formats. So check beforehand, if
             you try special formats. Python also has problems with special
             formats

Application in the Framework
############################

YAML is mainly used in this framework,
because it

    * delivers a unified way of defining configuration and specification files,
    * is easy to read,
    * can be easily loaded into Python, and
    * the loading already detects standard data types even
      if they are used in nested structures, like dictionaries of lists.

In pySPACE configuration files, in operation specification files
and for dataset descriptions
the main structure is a *dictionary* (mapping),
which is simply defined by ``key: value`` pairs at each line.
The empty space after the colon is important.
Additional empty spaces between key and colon can be used for cosmetic reasons.
If the value is another substructure 
you may use a new line instead of the empty space,
but then you have to indent everything belonging to the definition of the value,
to show, that it belongs to this key.
For the definition of dictionaries you can also use braces (``{key1: value1, key2: value2}``).
Some of the values may be *lists*. They can be defined in two ways:

    * use a dash followed by a space ("- ") or new line with indentation or
    * use squared brackets (``[item1, item2]``) as in Python.
    
Single-line versions are mainly used if you have
lists of lists or dictionaries or dictionaries, but they do not have to be used.
These data types are called *collections* in YAML.

.. todo:: Documentation on datasets needed!

.. _eval:

Eval(uate) syntax for Python code to ease specification file writing
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Finally, it is important to mention that we made two small syntax modifications.

When loading YAML specification files with specification of
:class:`~pySPACE.missions.nodes`
you may want to use Python commands to define your parameter values.
Therefore, a string of the form ``eval(command)`` results in a new value,
which is the result of the command, and which is then send to the
initialization of the node.
Normally, this is only applicable to the values of the keys in the parameters
dictionary of the node specification.
This syntax can be even used to specify the node name.

The same syntax possibility is included in the framework,
when using the key *parameter_ranges* in :mod:`operations <pySPACE.missions.operations>`.
Here you can use for example ``eval(range(10))`` instead of
``[0,1,3,4,5,6,7,8,9,10]``.

.. note:: In contrast to the the normal Python syntax, the command must not be
          marked as string.

.. _yaml_node_chain:

Usage of YAML in node chain specification files
+++++++++++++++++++++++++++++++++++++++++++++++

Since a signal processing node chain is mainly
a linear concatenation of elemental processing steps (called *nodes*),
the corresponding specification file is a list,
specifying the elemental processing steps in the form of nodes.

A node itself is always a dictionary of the keys *node*, containing the node name,
and *parameters* (optional), which value is another dictionary,
defining the variables for the node.
Normally these variables are directly given to the initialization function of the node
via a method called ``node_from_yaml`` of the general :class:`~pySPACE.missions.nodes.base_node.BaseNode`.
Since variables may be lists or dictionaries,
you may get a lot of indentations, but this is not the normal use-case.

For constructing your node chain you first have to choose your :mod:`nodes <pySPACE.missions.nodes>`.
To find out the possibilities of a node just check its documentation.
There you find an example and (normally) an exhaustive description of all
mandatory, recommended and optional parameters.
Finally the specifications of your chosen nodes are concatenated as a list
in your file and you are finished.


Example Node Chain File
..................................

.. literalinclude:: /examples/specs/node_chains/example_flow.yaml
    :language: yaml

