.. _doc:

Documentation
+++++++++++++

Generating the Documentation
============================

Documentation is done with `Sphinx <http://sphinx.pocoo.org/>`_
and some helper functions coming with the software for more customization.
The folder that contains all the documentation is called ``docs``.
To compile, you first have to install Sphinx 1.1 or a better version and
pySPACE (see: :ref:`t_install`).
For creating the inheritance diagrams the
`Graphviz package <http://www.graphviz.org/>`_ is needed.
The documentation can be created by running ``make html`` in the ``docs`` directory
(therefore we have a ``Makefile`` in the ``docs`` folder).
To view the documentation open the ``index.html`` in the ``.build/html`` folder.

A compiled version of the documentation can be found
`on the git web page <http://pyspace.github.io/pyspace/index.html>`_.

Programmer Guideline to Docstrings
==================================

The configuration of Sphinx is done within ``conf.py``.
This file has an exhaustive documentation to inform about possible
configurations, though the current one should be sufficient.

Working on the documentation consists of three parts:
The first one is the writing of ``*.rst`` files in the ``doc`` folder.
The second is writing documentation in the python docstrings.
The third is connecting everything with links.
*rst* stands for `ReStructured text format <http://docutils.sourceforge.net/rst.html>`_.
The connection between these files is done by
`Sphinx <http://sphinx.pocoo.org/>`_, and there are
numerous possibilities to link the files and content.

Docstrings should be both

 * compatible with `Sphinx <http://sphinx.pocoo.org/>`_
 * easy to parse for GUIs
 * should mainly follow the Python PEP 257 guidelines

Adding or Changing Software Components - What has to be done in the Documentation?
----------------------------------------------------------------------------------

When adding new software modules, the module has to be at least integrated
into the API documentation. This means that the module will show up in the
documentation and that the doc-strings from the source code will be parsed.

Luckily, this integration is done automatically via special python scripts
especially for :mod:`pySPACE.missions`.
Nevertheless the doc-string has to be written, such that it is clear, what the
component does, how it is accessed and what is its relation to the whole program.

For bigger changes it is necessary to add further documentation.
Therefore, text may be added to the *rst* files created above or new files may
be added and linked.
Sometimes it is also a good idea to write a tutorial for a new function.

At last it is essential to writes tests.

Sphinx
------

Sphinx can be found at: http://sphinx.pocoo.org/.
It uses

    * the `reStructuredText <http://docutils.sourceforge.net/rst.html>`_
      mark-up language (So this language should be used in the docstrings!),
    * the `Docutils <http://docutils.sourceforge.net/>`_
      as parsing and translating suite to generate for example latex or html files
    * and several extensions from external developers.

Thoug Sphinx is mainly using reStructuredText syntax,
it also provides some extra commands (beginning always with ``..`` or
``:some_reference_command_like_mod_or_ref:``) for better highlighting of text 
and for linking between different text parts, which is very powerful, 
when using the html output.

`reStructuredText <http://docutils.sourceforge.net/rst.html>`_ essentials
-------------------------------------------------------------------------

As taken from the 
`Quick reStructuredText Reference <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_:

     :Plain text:                   typical result
     :``*emphasis*``:               *emphasis (italic)*
     :``**strong emphasis**``:      **strong emphasis (bold)**
     :```interpreted text```:       `The rendering and meaning of interpreted text 
                                    is domain- or application-dependent.
                                    It can be used for things like 
                                    index entries or explicit descriptive mark-up 
                                    (like program identifiers).`

     :````inline literal````:       ``Normally rendered as monospaced text. 
                                    Spaces should be preserved, 
                                    but line breaks will not be.``

     :``reference_``:               A simple, one-word hyperlink reference
     :A stand-alone hyperlink:      http://docutils.sf.net/


You can use `the reST test page <http://www.tele3.cz/jbar/rest/rest.html>`_
to test your reStructuredText.

Definition of a Docstring format using reStructuredText
-------------------------------------------------------

`PEP 287 -- reStructuredText Docstring Format <http://www.python.org/dev/peps/pep-0287/>`_ 
makes an approach towards the definition of a standard format - though no actual formatting is proposed.

Our format for docstrings of '''classes''' is as in the following.
This is meant to be a self explanatory template.
`The reST test page <http://www.tele3.cz/jbar/rest/rest.html>`_ 
may help to see, how the documentation compiles,
but is not able to handle Sphinx commands like ``.. codeblock:: yaml``.
To check this, you have to compile the documentation, 
check for compiling errors, and have a look at the html result,
if it looks as expected.::

    """ Always have one short explanatory sentence here

    Than comes the sophisticated explanation. Take the trouble, write much!
    Explain it all! Explain when and how and why this class/node is used and
    whatever else might be of interest.

    Use blank lines for paragraphs. Maybe you even want to use enumerations.
    You can do that as follows:

    1. Just use \"1." and so on.
    2. You can use "\\" as escape character.
    3. Use a blank line before the enumeration.
    #. For bulletpoints use \"*" or \"-" instead of \"1.".

    Before and after each enumeration, use a blank line.    
    All the other fancy stuff is described in
    http://docutils.sourceforge.net/docs/user/rst/quickref.html

    .. note:: Make extensive use of Sphinx commands like referencing, 
              code_block syntax highlighting, warnings, todos,...
              
              .. warning:: Check if you used them the right way!

    Then describe the parameters as a field list and use
    the bold term "\**Parameters**" followed by a blank line before the list.
    This should look as follows:

    **Parameters**
    You may want to have some general description of the parameters.
    If you want to have subcategories, feel free to introduce them.
    This is done in the same way as for the variable names with an additional indentation.
    If you have a lot indentations you may use 2 spaces.
    Otherwise 4 spaces are the standard.

      :parameter_name:
        Explanation of the parameter. Again, write enough so that everybody
        understands what's happening. For the name use "\:name:". The actual
        text has to be indented one tab further than the "\:name:", starting
        in the next line. You can use multiple paragraphs if you hold the
        indentation level.

      :name2:
        Explanation of the second parameter. Use "(\*optional, default:
        default_value*)" as last paragraph if the parameter is optional.

        (*optional, default: True*)

    After the parameter list there shouldn't be any more explanations.
    However, things like "Known Issues" could be discussed here. Finally write
    down an exemplary call, in the case of nodes or operation classes. 
    Sometimes, parameters have to be stated in very
    specific formats, e.g. \"'some string'". Make sure to show how this is
    done! Use \"::" to indicate that the following paragraph is a literal
    block. In the very end, use \:Author: and \:Created: to tell people when
    and by whom the code was written. Don't hesitate and introduce
    \:Reviewed:, \:Version: and whatever you want. Use no blank line before the
    final \""".
    If you totally rewrite something, feel free to replace certain keywords.
      
    **Exemplary Call**

    .. code-block:: yaml

        - 
            node : NodeName
            parameters : 
                par1 : "exemplary_string_value"
                other_parameter : True

    .. todo:: add new training parameter here

    :input: TimeSeries
    :output: FeatureVector
    :training_type: supervised
    :Author: David Feess (David.Feess@dfki.de)
    :Created: 2010/07/23
    """

For classes ending with `Node` we added a special feature.
Additionally to the class name with and without node,
you could specify extra name mapping in the 
dictionary module parameter `_NODE_MAPPING`.
For every node class we determine all possible names
and at the end of the class main documentation,
this information is added in the online documentation.

Here's a skeleton which could be used as template for any class::

    """ Always have one short explanatory sentence here as described in PEP 257

    Then comes the sophisticated explanation. Take the trouble, write much!
    Very much!

    **Parameters**

      :parameter_name1:
        Explanation of the parameter

      :parameter_name2:
        Explanation of the second parameter

        (*optional, default: True*)
      
    **Exemplary Call**

    .. code-block:: yaml

        - 
            node : NodeName
            parameters : 
                parameter_name1 : "exemplary_value"
                parameter_name2 : True

    :input:    InputVectorType
    :output:   OutputVectorType
    :training_type: None|supervised|unsupervised|optional|incremental
    :Author: Some Guy (Some.Guy@dfki.de)
    :Created: YYYY/MM/DD
    """

We use a similar approach for module and package docstrings:

**Packages** do not need anymore a field list containing each module of the package,
since this list is auto generated from the module docstrings, 
which should be good one-liners.

Much more important is a general description 
on the general purpose/functionality of this package.
So use a one-liner and a more sophisticated description.::

    """ Nodes, wrapping other groups of nodes
    
    This package contains meta nodes, 
    which call other nodes for parameter optimization,
    to skip some training phase or initialization,
    or to combine the results.
    """

**Modules** only need a good explanation::

    """ Process several other nodes together in parallel and combine the results

    This is useful to be combined with the
    :class:`~pySPACE.missions.nodes.meta.flow_node.FlowNode`.
    """

When generating the documentation of a module, several steps are done
automatically:

    1. An inheritance diagram is plotted.
    2. A class summary is added if necessary.
    3. A function summary is added if necessary.
    4. All classes and functions get headlines.


Useful commands
---------------

    * To make notes, warnings, todos use ``.. note::``, ``.. warning::``, ``.. todo::``.
      If you need more than one line, use indentations.
      Surround the command with an upper and a lower blank line.
    
    * ``.. code-block::`` cares for syntax highlighting in code.
    
    * If you want to have a text as written, the beginning command is ``::`` 
      and the text should be indented and surrounded with blank lines.
    * In the in-line mode, you may use double ````apostrophes````.
    
    * Variables are normally highlighted with ``*stars*`` for *italics* or 
      with the ````apostrophes````.
    
    * For using math formulas use the command ``.. math::`` 
      followed by the indented formula block.
      The syntax is the same as in latex but instead of one backslash 
      you have to use two as in the following example::

            The main formula comes from *M&M*:
            
            .. math::
            
                d        = \\alpha_i - \\frac{\\omega}{M[i][i]}(M[i]\\alpha-1)
                
                \\text{with } M[i][j]  = y_i y_j(<x_i,x_j>+1)
                
                \\text{and final projection: }\\alpha_i = \\max(0,\\min(d,c_i)).
                
            Here we use c for the weights for each sample in the loss term,

      which renders like:

            The main formula comes from *M&M*:
            
            .. math::
            
                d        = \alpha_i - \frac{\omega}{M[i][i]}(M[i]\alpha-1)
                
                \text{with } M[i][j]  = y_i y_j(<x_i,x_j>+1)
                
                \text{and final projection: }\alpha_i = \max(0,\min(d,c_i)).
                
            Here we use c for the weights for each sample in the loss term,

       When using formulas in the normal documentation, you have to use
       the standard backslash.

    * Linking is very easy and a good way of documentation Python and to show connections.
    
      * A normal link is introduced by the command 
        ``.. _linkname:`` followed by a section header
        and used with::
        
        ``:ref:`linkname``` or ``:ref:`linknamewrappername<linkname>```.

        The latter replaces the standard link name from the header.
    
      * For linking modules, classes or functions just use ``:mod:``, 
        ``:class:``, or ``:func:`` respectively instead of 
        ``:ref:`` as described before.
        The linkname is the path to the referee, e.g.::
        
            :mod:`pySPACE.missions.operations.node_chain`
        
        which is also the standard link name then.
        You could also use the tilde. Then only the last component is used as 
        link name and not the whole link name. 

Commons in Python: `Python PEP 257 <http://www.python.org/dev/peps/pep-0257/>`_
-------------------------------------------------------------------------------

This is a little summary of the key points from http://www.python.org/dev/peps/pep-0257/.

One-Line Docstrings
###################

 * '''Triple quotes''' are used even though the string fits on one line.
 * The closing quotes are on the *same line* as the opening quotes. This looks better for one-liners.
 * There's *no blank line* neither before nor after the docstring.
 * The docstring is a *phrase*. 
   Prescribes the function's effect as a *command*
   ("Do this", "Return that"), not as a description; 
   e.g., don't write "Returns the pathname ...".
   * In contrast to PEP we do NOT use any sign at the end of the first phrase!
 * Should NOT be a "signature" reiterating the parameters. 
   Mention return value, as it cannot be determined by introspection.

Example::

    def kos_root():
        """ Return the pathname of the KOS root directory """
        global _kos_root
    if _kos_root: return _kos_root
    ...


Multi-line Docstrings
#####################

 * Multi-line docstrings consist of a **summary line**
   just like a one-line docstring, followed by a *blank line*, 
   followed by a more *elaborate description*.
 * Insert a *blank line* before 
    and after all docstrings (one-line or multi-line), that document a *class*. 
    Not for methods.
 * The docstring of a **script**: Should be usable as its *"usage" message*.
 * The docstring for a **module**: List the classes, exceptions, 
   functions with a one-line summary of each. (Done automatically in our case.)
 * The docstring for a **function** or **method**: 
   Summarize its behavior and document its arguments, 
   return value(s), side effects, exceptions raised, restrictions. 
   Indicate optional arguments, document keyword arguments.
 * The docstring for a **class**: 
   Summarize its behavior and list the public methods and instance variables.
 * Mention, if a class *subclasses* another class. 
   Summarize the differences. Use the verbs "override" and "extend".
   Also wrapping should be mentioned.

Example::

    def complex(real=0.0, imag=0.0):
        """ Form a complex number
    
        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
    
        """
    if imag == 0.0 and real == 0.0: return complex_zero
    ...

In our case the content is similar, but the format is influenced by special rest-rules.

