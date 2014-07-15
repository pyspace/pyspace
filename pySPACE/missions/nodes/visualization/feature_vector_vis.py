""" Visualize :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector` elements"""
import itertools
import pylab
import numpy
try:
    import mdp.nodes
except:
    pass

from pySPACE.missions.nodes.base_node import BaseNode

class LLEVisNode(BaseNode):
    """ Show a 2d scatter plot of all :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector` based on Locally Linear Embedding (LLE) from MDP
    
    This node collects all training examples it obtains along with their
    label. It computes than an embedding of all these examples in a 2d space
    using the "Locally Linear Embedding" algorithm and plots a scatter plot of
    the examples in this space.
    
    **Parameters**
    
        :neighbors:
            The number of neighbor vectors that should be considered for each
            instance during locally linear embedding
            
            (*optional, default: 15*)
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        - 
            node : Time_Series_Source
        -
            node : All_Train_Splitter
        -
            node : Time_Domain_Features
        -
            node : LLE_Vis
            parameters :
                neighbors : 10
        -
            node : Nil_Sink

    Known Issues:
    This node will use pylab.show() to show the figure. There is no store
    method implemented yet. On Macs, pylab.show() might sometimes fail due to
    a wrong plotting backend. A possible workaround in that case is to
    manually set the plotting backend to 'MacOSX'. This has to be done before
    pylab is imported, so one can temporarily add "import matplotlib;
    matplotlib.use('MacOSX')" to the very beginning of launch.py.
                    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/07/07
    """
    input_types = ["FeatureVector"]

    def __init__(self, neighbors = 15, **kwargs):
        super(LLEVisNode, self).__init__(**kwargs)
                
        self.set_permanent_attributes(
                  neighbors = neighbors, 
                  # A set of colors that can be used to distinguish different classes
                  colors = set(["r", "b"]),
                  # A mapping from class label to its color in the plot
                  class_colors = dict(),
                  # Remembers the classes (colors) of the instances seen
                  instance_colors = [], 
                  #
                  instances = []
                  )  
        
        pylab.ion()
        figure = pylab.figure(figsize=(21, 11))
        figure.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top= 0.99,
                               wspace=0.2, hspace=0.2)
        pylab.draw()

    def is_trainable(self):
        """ Returns whether this node is trainable. """
        # Though this node is not really trainable, it returns true in order
        # to get trained. The reason is that during this training phase, 
        # it visualizes all samChannel_Visples that are passed as arguments
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _get_train_set(self, use_test_data):
        """ Returns the data that can be used for training """
        # We take data that is provided by the input node for training
        # NOTE: This might involve training of the preceding nodes
        train_set = self.input_node.request_data_for_training(use_test_data)
        
        # Add the data provided by the input node for testing to the
        # training set
        # NOTE: This node is not really learning but just collecting all
        #       examples. Because of that it must take
        #       all data for training (even when use_test_data is False) 
        train_set = itertools.chain(train_set,
                                    self.input_node.request_data_for_testing())
        return train_set
    
    
    def _train(self, data, label):
        """
        This node is not really trained but uses the labeled examples  to
        generate a scatter plot.
        """        
        # Determine color of this class if not yet done
        if label not in self.class_colors.keys():
            self.class_colors[label] = self.colors.pop() 
        
        # Stor the given example along with its class (encoded in the color)
        self.instances.append(data)
        self.instance_colors.append(self.class_colors[label])
        
        
    def _stop_training(self, debug=False):
        """ Stops the training, i.e. create the 2d representation
        
        Uses the Locally Linear Embedding algorithm to create a 2d 
        representation of the data and creates a 2d scatter plot.
        """
        instances = numpy.vstack(self.instances)
        
        # Compute LLE and project the data
        lle_projected_data = mdp.nodes.LLENode(k=self.neighbors,
                                               output_dim=2)(instances)

        # Create scatter plot of the projected data
        pylab.scatter(lle_projected_data[:,0], lle_projected_data[:,1],
                      c = self.instance_colors)
        pylab.show()
        
    def _execute(self, data):
        # We simply pass the given data on to the next node
        return data


_NODE_MAPPING = {"LLE_Vis": LLEVisNode}
