""" Type superclass providing common variables that survive the data processing

The BaseData is almost invisible for the user. No matter which data type
is used, all of them inherit from this class. This inheritance is basically
performed by the function *inherit_meta_from* where a new object can inherit
the meta data from another object. This is necessary between the nodes
and is automatically performed by the :mod:`~pySPACE.missions.nodes`.

But what is the meta data? Four elements are inherent
in the BaseData type: *key*, *tag*, *specs* and *history*. These elements
are Python properties, so it is recommended to know about the set and get methods 
of these properties.

.. note:: Just by loading an element within the framework, none of these
            elements is set. Instead, they are first set within
            the processing node chain
            using generate_meta.

Central elements are the unique identifier *key* and an identifying *tag*. Both
complement each other and are automatically set by the method *generate_meta* which
is in called within the node.

The key must be a UUID and the tag must be a STRING giving
unique information about the underlying data. Every data type can have an
automatic tag generation method (called _generate_tag).
In particular, the *key* is automatically created using a uuid
when the element is processed for the first time
and then belongs to this data element. The *tag* is
very similar, but its generation can be implemented by the subclass, because
it contains semantic information (like e.g. time stamps) which is not known by
the BaseData class.

The other two properties are more flexible: the *specs* and the *history* are
 able to backup information the user is needing. *specs* is a dictionary and
 the *history* is a list. Both are empty by default.

The history is a placeholder (empty list) for storing results of previous
nodes within a node chain. It can be filled by setting the *keep_in_history*
parameter to *True* within a node chain. Then the data object is copied into
the *history* as it is after this processing step. Additionally,
the *specs* receive an entry labeled 'node_specs' containing a dictionary of
additional information from the saving node. For n times storing the results
within one node chain, the *history* property has the length n.

.. note:: (I) When a data element is stored in the *history*, it loses its own
    history. (II) Using history=myhistory appends something to the history;
    the only way of deleting information here is to empty it (history=[]).

The *specs* property is an empty dictionary which can be filled with whatever
 the user or developer needs in more than one node. The syntax is exactly the
  same as with a normal python dictionary.

.. note:: All BaseData type properties survive data type conversion within
          a :class:`~pySPACE.environments.chains.node_chain.NodeChain`
          (e.g. from TimeSeries to FeatureVector)! The central function
          used for this is inherit_meta_from().

.. warning:: When slicing the data, all meta data is copied.
             So please try to avoid slicing and better use
             'x=data.view(numpy.ndarray)' to replace *data* by *x* for further
             array processing. This does not copy memory,
             but creates a new clean reference to the array without meta data.

:Author: Hendrik Woehrle, Sirko Straube, Mario Krell, David Feess
:Created: 2010/08/09
:Major Revision: 2012/01/20
"""

import numpy
import uuid
import copy
import warnings


class BaseData(numpy.ndarray):
    """ Basic Data object
    
    Superclass for every data type - provides common variables
    that survive the data processing.

    For further details refer to the module documentation.
    """

    def __new__(subtype, input_array):
        """Constructor for BaseData object

        If the input_array has meta data, it will survive the construction process and be present
        in the new object.
        
        Refer to
        http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        for infos about subclassing ndarray
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        # hereby array finalize is called
        # and such the variables set to standard values 
        obj = numpy.asarray(input_array).view(type=subtype)
        obj._inherited = False

        return obj

    def __array_finalize__(self, obj):
        #note: getattr is necessary here, because __array_finalize__ is called by numpy.asarray()
        #just before key, tag, specs and history are set
        if not (obj is None) and not (type(obj) == numpy.ndarray):
            self.__key = getattr(obj, 'key', None)
            self.__tag = getattr(obj, 'tag', None)
            self.__specs = getattr(obj, 'specs', {})
            self.__history = getattr(obj, 'history', [])
        else:
            self.__key = None
            self.__tag = None
            self.__specs = {}
            self.__history = []
        raise_error = False
        try:
            a = obj.shape
            b = self.shape
            raise_error = False
            if not (a == b):
                # array changed because of slicing 
                # If you encounter this 'problem', try to use a raise here
                # and you should get the traceback telling you,
                # which node caused the problem.
                # If the node uses a bad try-except you have to use
                # a debugging tool!
                raise_error = True
                #print a,b
                warnings.warn(
                    "Slicing was used on your data with type %s" % type(self) +
                    "Better cast object to ndarray before slicing! " +
                    "('new_array = data.view(numpy.ndarray)' " +
                    "Metadata is probably not consistent anymore. " +
                    "For further debugging check the implementation in " +
                    "'pySPACE.resources.data_types.base.py.'"
                    )
        except:
            pass
#        if raise_error:
#            raise TypeError("Use the traceback to find out, where you used slicing on the data. Replace the data by x=data.view(numpy.ndarray)!")

    def __reduce__(self):
        # Refer to 
        # http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html
        # for infos about pickling ndarray subclasses
        object_state = list(numpy.ndarray.__reduce__(self))
        subclass_state = (self.key, self.tag, self.specs, self.history)
        object_state[2] = [object_state[2], subclass_state]
        return object_state

    def __setstate__(self, state):
        nd_state, own_state = state
        numpy.ndarray.__setstate__(self, nd_state)
        if len(
                own_state) > 4: #backward compatibility with old implementation of BaseData type
            own_state = own_state[0:4]
        (self.key, self.tag, self.specs, self.history) = own_state

    def __generate_key__(self):
        """uuid for key"""
        return uuid.uuid4()

    ###following: property definitions, setter and getter
    def __set_key__(self, new_key):
        """set method for key
        
        Generally the key should be generated one time and then stay the same.
        When creating it, a uuid has to be used! Then, the user can still change it:
        Either he can set it to None or use another uuid.
        """
        if new_key == None:
            self.__key = None
            return

        else:
            if self.key != None:
                warnings.warn(
                    "BaseData type:: data has key already. be careful when changing!")
                # in the other case we are having old data with key but not self.__key

            if type(new_key) == uuid.UUID:
                self.__key = new_key
            else:
                warnings.warn(
                    "BaseData type:: use a uuid when changing the key! attempt ignored!")

    def __get_key__(self):
        """ return key """
        return self.__key

    def __del_key__(self):
        """ delete key """
        self.__key = None

    key = property(__get_key__, __set_key__, __del_key__,
                   "Property key of BaseData type.")

    def __set_tag__(self, new_tag):
        """set method for tag
        
        Similar behavior to __set_key__. The tag is ALWAYS a string. So whatever
        is given as new_tag is casted into string (exception: None).
        """

        if new_tag == None:
            self.__tag = None
            return
        else:
            if self.tag != None:
                warnings.warn(
                    "BaseData type:: data already tagged. be careful when changing!")

            self.__tag = str(new_tag)


    def __get_tag__(self):
        """return tag
        """
        return self.__tag

    def __del_tag__(self):
        """delete tag
        """
        self.__tag = None

    tag = property(__get_tag__, __set_tag__, __del_tag__,
                   "Property tag of BaseData type.")

    def __set_specs__(self, key, value=None):
        """set method for specs
        
        The specs property is a dictionary, so every attempt to add something here
        is interpreted as an attempt to fill this dictionary with 
        dict.__setitem__(self.specs, key, value). The key is always casted into a string.
        If value is 'None' the operation is not performed.
        
        """
        if key == {} and value == None:
            self.__specs = {}
            return

        # Here the question remains, if we do not want to overwrite the specs
        # or if we choose a dictionary for setting, that we want to create a new 
        # object. Currently the dictionary is extended by the new dictionary.
        if type(key) == dict and value == None:
            for skey, svalue in key.iteritems():
                self.specs[skey] = svalue
            return

        if value == None:
            return

        if self.__specs.has_key(key):
            warnings.warn(
                "BaseData type:: specs have already an entry labeled " + str(
                    key) + ". This entry is overwritten!")

        dict.__setitem__(self.__specs, key, value)


    def __get_specs__(self):
        """return specs
        """
        return self.__specs

    def __del_specs__(self):
        """delete specs
        """
        self.__specs = {}

    specs = property(__get_specs__, __set_specs__, __del_specs__,
                     "Property specs of BaseData type. This property is a dictionary.")

    def __set_history__(self, elem):
        """set method for history
        
        The history is a list of elements.
        Therefore, the history can only be set by the user in two ways:
        - it can be deleted by setting history=[]
        or
        - it can be extended with another element
        """

        if elem == []:
            self.__history = []
        else:
            self.__history.append(elem)


    def __get_history__(self):
        """return history
        """
        return self.__history

    def __del_history__(self):
        """delete history
        """
        self.__history = []

    history = property(__get_history__, __set_history__, __del_history__,
                       "Property history of BaseData type. This property is a list.")

    ###following: user methods for interacting with the BaseData type

    def has_meta(self):
        """Return whether basic meta data is present (key and tag)"""
        return not (self.key == None and self.tag == None)

    def has_history(self):
        """Return whether history is present."""
        return self.history != []

    def get_data(self):
        """A simple method that returns the data as a numpy array"""
        return self.view(numpy.ndarray)

    def generate_meta(self):
        """generate basic meta data (key and tag)"""
        self.key = self.__generate_key__()
        if hasattr(self, "_generate_tag"): #function is implemented by subclass
            self.tag = self._generate_tag(self)
        else:
            self.tag = None

    def inherit_meta_from(self, obj):
        """ Inherit history, key, tag and specs from the passed object """
        if not getattr(self, "_inherited",
                       False) and not obj is None and not type(
                obj) == numpy.ndarray:
            # Use a copy of the history from old object as own history:
            if self.history == []:
                self.__history = copy.deepcopy(obj.history)
            else: #merge
                old_history = copy.deepcopy(self.__history)
                self.__history = copy.deepcopy(
                    obj.history) #overwrite existing history
                for elem in old_history:
                    self.history = elem

            self.specs = copy.deepcopy(
                obj.specs) #specs cannot be overwritten (see __set_specs__)

            if self.key is None:
                self.key = getattr(obj, 'key', None)
            if self.tag is None:
                self.tag = getattr(obj, 'tag', None)
            self._inherited = True

    def add_to_history(self, obj, node_specs=None):
        """Add the passed object to history.
        
        further specs can be given here which are passed into specs property
        of the object (default: None)
        """
        # Use a copy of the data object and remove the history because 
        # it would be redundant
        # Note: This will get obsolete with an own implementation of deepcopy
        specs = copy.deepcopy(obj.specs)
        key = copy.deepcopy(obj.key)
        tag = copy.deepcopy(obj.tag)

        obj_without_meta = copy.deepcopy(obj)
        del obj_without_meta.key
        del obj_without_meta.tag
        del obj_without_meta.specs
        del obj_without_meta.history

        #transfer local copies
        obj_without_meta.key = key
        obj_without_meta.tag = tag
        obj_without_meta.specs = specs
        obj_without_meta.specs['node_specs'] = copy.deepcopy(node_specs)

        self.history = obj_without_meta #appends to history
