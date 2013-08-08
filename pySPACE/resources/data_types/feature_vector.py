""" 1d array of feature values with some additional properties (e.g. feature names)

This type is collected as
:class:`~pySPACE.resources.dataset_defs.feature_vector.FeatureVectorDataset`,
loaded with :mod:`~pySPACE.missions.nodes.source.feature_vector_source`
and saved with :mod:`~pySPACE.missions.nodes.sink.feature_vector_sink`.

.. todo:: Implement a method _generate_tag for inherited data type (if desired)

:Author: Jan Hendrik Metzen  (jhm@informatik.uni-bremen.de)
:Created: 2009/01/30
"""

import numpy
import warnings
from pySPACE.resources.data_types import base

class FeatureVector(base.BaseData):
    """ Represent a feature vector.
    
    Refer to http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    for information about subclassing ndarray
    """
    
    def __new__(subtype, input_array, feature_names=None, tag=None):
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
        # reset the attributes from passed original object
        super(FeatureVector, self).__array_finalize__(obj)
        
        if not obj is None and not type(obj)==numpy.ndarray:
            self.feature_names_hash = getattr(obj, 'feature_names_hash', None)
        else:
            # TODO: Do we need this?
            self.feature_names_hash = None
        
    def __str__(self):
        str_repr =  ""
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
        if len(state) == 2: # For compatibility with old FV implementation
            nd_state, own_state = state
            numpy.ndarray.__setstate__(self, nd_state)
        else: #len == 3: new feature vector. TODO: catch weird cases?
            nd_state, base_state, own_state = state
            super(FeatureVector, self).__setstate__((nd_state, base_state))
        
        self.feature_names = own_state

    
    # In order to reduce the memory footprint, we do not store the feature names 
    # once per instance but only once per occurrence. Instead we store a unique 
    # hash once per instance that allows to retrieve the feature names.
    feature_names_dict = {}
    def get_feature_names(self):
        return FeatureVector.feature_names_dict[self.feature_names_hash]
    def set_feature_names(self, feature_names):
        self.feature_names_hash = hash(str(feature_names))
        if not FeatureVector.feature_names_dict.has_key(self.feature_names_hash):
            FeatureVector.feature_names_dict[self.feature_names_hash] \
                                                                = feature_names
    def del_feature_names(self):
        pass
    feature_names = property(get_feature_names, set_feature_names, 
                             del_feature_names, 
                             "Property feature_names of FeatureVector.")
    
    @staticmethod
    def replace_data(old, data, **kwargs):
        """ Create new feature vector with the given data but the old metadata.
        
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
        """ Same features (names) and values """
        if not type(self)==type(other):
            return False
        if not set(self.feature_names)==set(other.feature_names):
            return False
        if not self.shape==other.shape:
            return False
        if self.feature_names==other.feature_names:
            if (self.view(numpy.ndarray)-other.view(numpy.ndarray)).any():
                return False
            else:
                return True
        else:
            # Comparison by hand
            for feature in self.feature_names:
                if not self[0][self.feature_names.index(feature)]==self[0][other.feature_names.index(feature)]:
                    return False
            return True