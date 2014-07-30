""" Extend array of feature values with some additional properties """
import numpy
import warnings
from pySPACE.resources.data_types import base


class FeatureVector(base.BaseData):
    """ Represent a feature vector (including meta information)

    This type is collected as
    :class:`~pySPACE.resources.dataset_defs.feature_vector.FeatureVectorDataset`.
    This type of dataset can be also used to create sets from FeatureVectors,
    e.g., from csv files.
    For data access in a node chain, data is
    loaded with the
    :class:`~pySPACE.missions.nodes.source.feature_vector_source.FeatureVectorSourceNode`
    as first node
    and saved with a
    :class:`~pySPACE.missions.nodes.sink.feature_vector_sink.FeatureVectorSinkNode`
    as the last node.
    For creating feature vectors from
    :class:`~pySPACE.resources.data_types.time_series.TimeSeries`
    objects in a processing chain, the nodes in the
    :mod:`~pySPACE.missions.nodes.feature_generation` module can be used.

    For historical reasons, the FeatureVector  object is a
    2d array where the first axes is not actively used.

    When creating a feature vector, the first mandatory argument with the data
    (*input_array*) can be any object,
    which can be cast to a two dimensional array,
    including lists and one-dimensional arrays.

    The second (recommended) argument are the *feature_names*.
    This is a list of strings which becomes an additional property of
    the feature vector class. For the MNIST dataset, feature names like
    ``Pixel001`` are used.
    For mappings between feature vectors and
    :class:`~pySPACE.resources.data_types.time_series.TimeSeries` objects,
    certain naming conventions should be met.
    The name parts should be separated with an underscore.
    The first part denotes an abbreviation of the feature type.
    The second part denotes the sensor/channel, the data was taken from.
    The third part is an information on the time part in seconds.
    the feature was taken from.
    An example of such a feature name is ``TD_C3_0.080sec``.

    If no feature names are given, the default name is
    ``feature_INDEX_0.000sec`` where INDEX is replaced by the respective index
    of the feature in the data array.

    .. note:: For memory issues, the feature_names are hashed and not stored
              individually for every samples.

    Additionally, a FeatureVector comes with the metadata from its
    :class:`base type <pySPACE.resources.data_types.base.BaseData>`.

    For accessing the array only without the meta information,
    please use the command

    .. code-block:: python

        x = data.view(numpy.ndarray)

    which hides this information.
    This is especially necessary, if you do not want to forward certain
    information. For example the feature names will not make any sense,
    when accessing only a subpart of the array.

    .. todo:: Implement a method _generate_tag for inherited data type (if desired)

    .. todo:: Change 2d-concept to 1d-concept.
              This includes modifying every node, which handles FeatureVector
              data.

    :Author: Jan Hendrik Metzen  (jhm@informatik.uni-bremen.de)
    :Created: 2009/01/30
    """
    
    def __new__(subtype, input_array, feature_names=None, tag=None):
        """ Create FeatureVector object

        Refer to general class documentation.
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = base.BaseData.__new__(subtype, numpy.atleast_2d(input_array))
        if obj.ndim > 2:
            input_array = obj[0]
            obj = base.BaseData.__new__(subtype, input_array)
        # handling transposed vectors
        if obj.shape[0] > 1:
            obj = obj.T
            warnings.warn("False order of dimensions in FeatureVector.")
        # add subclasses attributes to the created instance
        if feature_names is None:
            feature_names = ["feature_%s_0.000sec" % i 
                                    for i in range(obj.shape[1])]
        try:
            assert(len(feature_names)==obj.shape[1]),"Feature names (%s) do not match array dimensions (%s)! Fix this!"%(str(feature_names),str(obj.shape))
        except:
            warnings.warn("Length of feature names (%d) do not match array dimensions (%s)! Fix this! Occurring feature names:%s"%(len(feature_names),str(obj.shape),str(feature_names)))
        obj.feature_names = feature_names
        if not tag is None:
            obj.tag = tag
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        """ Inherit feature names from obj """
        # reset the attributes from passed original object
        super(FeatureVector, self).__array_finalize__(obj)
        
        if not obj is None and not type(obj) == numpy.ndarray:
            self.feature_names_hash = getattr(obj, 'feature_names_hash', None)
        else:
            # TODO: Do we need this?
            self.feature_names_hash = None
        
    def __str__(self):
        """ Connect array values and feature names for good string representation """
        str_repr = ""
        data=self.view(numpy.ndarray)
        for feature_name, feature_value in zip(self.feature_names,
                                               data[0,:]):
            if isinstance(feature_value, basestring):
                str_repr += "%s : %s \n" % (feature_name, feature_value)
            else : 
                str_repr += "%s : %.4f \n" % (feature_name, feature_value)
            
        return str_repr

    def __reduce__(self):
        # Refer to 
        # http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html
        # for infos about pickling ndarray subclasses
        object_state = list(super(FeatureVector, self).__reduce__())
        
        subclass_state = (self.feature_names)
        object_state[2].append(subclass_state)
        object_state[2] = tuple(object_state[2])
        return tuple(object_state)
    
    def __setstate__(self, state):
        if len(state) == 2:  # For compatibility with old FV implementation
            nd_state, own_state = state
            numpy.ndarray.__setstate__(self, nd_state)
        else:  # len == 3: new feature vector. TODO: catch weird cases?
            nd_state, base_state, own_state = state
            super(FeatureVector, self).__setstate__((nd_state, base_state))
        
        self.feature_names = own_state


    # In order to reduce the memory footprint, we do not store the feature names
    # once per instance but only once per occurrence. Instead we store a unique
    # hash once per instance that allows to retrieve the feature names.
    feature_names_dict = {}

    def get_feature_names(self):
        """ Extract feature names from hash """
        return FeatureVector.feature_names_dict[self.feature_names_hash]

    def set_feature_names(self, feature_names):
        """ Set feature names, using the hash and store hash if needed """
        if type(feature_names) is not list:
            warnings.warn("The feature names must be of type list. " +
                          "The current input is of type " +
                          str(type(feature_names)) +". Switching to list.")
            feature_names = list(feature_names)
        self.feature_names_hash = hash(str(feature_names))
        if not FeatureVector.feature_names_dict.has_key(self.feature_names_hash):
            FeatureVector.feature_names_dict[self.feature_names_hash] \
                                                                = feature_names

    def del_feature_names(self):
        """ Nothing happens, when feature names are deleted """
        pass

    feature_names = property(get_feature_names, set_feature_names, 
                             del_feature_names, 
                             "Property feature_names of FeatureVector.")
    
    @staticmethod
    def replace_data(old, data, **kwargs):
        """ Create new feature vector with the given data but the old metadata

        A factory method which creates a feature vector object with the given
        data and the metadata from the old feature vector
        """
        data = FeatureVector(data,
                   feature_names=kwargs.get('feature_names',
                                            old.feature_names),
                   tag=kwargs.get('tag', None))
        data.inherit_meta_from(old)
        return data

    def __eq__(self,other):
        """ Test for same type, dimensions, features names, and array values """
        if not type(self)==type(other):
            return False
        if not set(self.feature_names)==set(other.feature_names):
            return False
        if not self.shape==other.shape:
            return False
        if self.feature_names==other.feature_names:
            return numpy.allclose(self.view(numpy.ndarray), other.view(numpy.ndarray))
        else:
            # Comparison by hand
            for feature in self.feature_names:
                if not numpy.allclose(self[0][self.feature_names.index(feature)],
                                      other[0][other.feature_names.index(feature)]):
                    return False
            return True