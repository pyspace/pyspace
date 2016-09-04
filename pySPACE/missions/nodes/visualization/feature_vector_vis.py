""" Visualize :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector` elements"""
import itertools
import os
import warnings
import pylab
import numpy
from collections import defaultdict
from pySPACE.resources.data_types.prediction_vector import PredictionVector
from pySPACE.tools.filesystem import create_directory

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


class MnistVizNode(BaseNode):
    """ Node for plotting MNIST Data

    **Parameters**
        :mode:
            One of *FeatureVector*, *PredictionVector*, and *nonlinear*.

            If *FeatureVector* is taken, the data is assumed to be in the
            28x28 format and can be visualized like the original data.

            If *PredictionVector* is chosen, the affine
            backtransformation approach is used. If possible,
            the visualization is enhanced by the average data found in
            the data history at the *history_index*.

            If *nonlinear* os used, a  nonlinear processing chain is assumed
            for calculating the backtransformation with derivatives
            with the sample at the

            If not specified, the input data type is used.

            (*recommended, default: input type*)

        :history_index:
            Index for determining the averaging data or the data for
            calculating the derivative from prediction vectors.
            To save the respective data, the *keep_in_history* parameter
            has to be used, in the node, which produces the needed data.
            This can be a Noop node at the beginning.

            By default the last stored sample is used.

            (*recommended, default: last sample*)

        :max_samples:
            In case of the *nonlinear* mode, a backtransformation
            graphic must be generated for every data sample.
            To reduce memory usage, only the first *max_*samples* training
            samples are used.

            (*optional, default: 10*)

    **Exemplary Call**

    .. code-block:: yaml

        - node : MnistViz
    """
    def __init__(self, mode=None, history_index=0, max_samples=10, **kwargs):
        super(MnistVizNode, self).__init__(**kwargs)
        self.set_permanent_attributes(
            averages=None,
            counter=None,
            mode=mode,
            history_index=history_index,
            inputs=None,
            max_samples=max_samples,
        )

    def _train(self, data, label):
        """ Average data with labels (no real training)"""
        if self.mode is None:
            self.mode = type(data).__name__
        if (self.mode == "PredictionVector") and data.has_history():
            new_data = data.history[self.history_index - 1]
            del(data)
            data = new_data
        if self.mode == "nonlinear":
            if self.inputs is None:
                self.inputs = []
            self.inputs.append(data)
        if self.mode in "FeatureVector" or (
                (self.mode == "PredictionVector")
                and not type(data) == PredictionVector):
            if self.averages is None or self.counter is None:
                self.averages = defaultdict(lambda : numpy.zeros((28, 28)))
                self.counter = defaultdict(float)
            # Average the given example along with its class
            data.view(numpy.ndarray)
            number_array = data.reshape((28,28))
            self.averages[label] += number_array
            self.counter[label] += 1
            if self.inputs is None:
                self.inputs = []
            if not len(self.inputs) == self.max_samples:
                self.inputs.append(number_array)

    def store_state(self, result_dir, index=None):
        """ Main method which generates and stores the graphics """
        if self.store:
            #set the specific directory for this particular node
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            #do we have an index-number?
            if not index is None:
                #add the index-number...
                node_dir += "_%d" % int(index)
            create_directory(node_dir)
            colors = ["white", "black", "blue", "red"]
            if self.mode == "FeatureVector":
                for label in self.averages:
                    self.averages[label] *= 1.0/self.counter[label]
                    #http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
                    pylab.figure(figsize=(4, 4), dpi=300)
                    pylab.contourf(self.averages[label], 50, cmap="jet",
                                   origin="image")
                    pylab.xticks(())
                    pylab.yticks(())
                    #pylab.colorbar()
                    f_name = str(node_dir)+str(os.sep)+str(label)+"average"
                    pylab.savefig(f_name + ".png", bbox_inches='tight')
                for index, input in enumerate(self.inputs):
                    pylab.figure(figsize=(4, 4), dpi=300)
                    pylab.contourf(input, 50, cmap="binary",
                                   origin="image")
                    pylab.xticks(())
                    pylab.yticks(())
                    #pylab.colorbar()
                    f_name = str(node_dir)+str(os.sep)+"sample"+str(index)
                    pylab.savefig(f_name + ".png", bbox_inches='tight')

            elif self.mode == "PredictionVector":
                trafos = self.get_previous_transformations()[-1]
                trafo = trafos[0]
                trafo.view(numpy.ndarray)
                covariance = trafos[1][1]
                trafo_covariance = numpy.dot(covariance, trafo.flatten())

                # covariance free picture
                number_array = trafo.reshape((28, 28))
                fig = pylab.figure(figsize=(4, 4), dpi=300)
                pylab.contourf(number_array, 50, cmap="jet", origin="image",
                               vmax=abs(number_array).max(),
                               vmin=-abs(number_array).max())
                pylab.xticks(())
                pylab.yticks(())
                #pylab.colorbar()
                if not self.averages is None:
                    for label in self.averages:
                        self.averages[label] *= 1.0/self.counter[label]
                        pylab.contour(
                            self.averages[label],
                            levels=[50],
                            colors=colors[self.averages.keys().index(label)],
                            linewidths=3,
                            origin="image")
                f_name = str(node_dir)+str(os.sep)+"classifier"
                pylab.savefig(f_name + ".png", bbox_inches='tight')
                pylab.close(fig)

                # covariance picture (similar code as before)
                number_array = trafo_covariance.reshape((28, 28))
                fig = pylab.figure(figsize=(4, 4), dpi=300)
                pylab.contourf(number_array, 50, cmap="jet", origin="image",
                               vmax=abs(number_array).max(),
                               vmin=-abs(number_array).max())
                pylab.xticks(())
                pylab.yticks(())
                #pylab.colorbar()
                if not self.averages is None:
                    for label in self.averages:
                        pylab.contour(
                            self.averages[label],
                            levels=[50],
                            linewidths=3,
                            colors=colors[self.averages.keys().index(label)],
                            origin="image")
                f_name = str(node_dir)+str(os.sep)+"classifier_cov"
                pylab.savefig(f_name + ".png", bbox_inches='tight')
                pylab.close(fig)
            elif self.mode == "nonlinear":
                from matplotlib.backends.backend_pdf import PdfPages
                import datetime
                with PdfPages(str(node_dir)+str(os.sep)+'sample_vis.pdf') as pdf:
                    index = 0
                    for sample in self.inputs:
                        index += 1
                        base_vector = sample.history[self.history_index-1]
                        trafos = self.get_previous_transformations(base_vector)[-1]
                        trafo = trafos[0]
                        trafo.view(numpy.ndarray)
                        covariance = trafos[1][1]
                        trafo_covariance = \
                            numpy.dot(covariance, trafo.flatten())
                        covariance_array = trafo_covariance.reshape((28, 28))

                        base_array = base_vector.reshape((28, 28))
                        trafo_array = trafo.reshape((28, 28))

                        #fig = pylab.figure(figsize=(5, 5), dpi=300)

                        #pylab.suptitle(sample.label)

                        # SUBPLOT 1: plot of the derivative
                        #pylab.subplot(2, 2, 1)
                        #pylab.title("Backtransformation")
                        fig = pylab.figure(figsize=(4, 4), dpi=300)
                        pylab.contourf(trafo_array, 50, cmap="jet",
                                       origin="image",
                                       vmax=abs(trafo_array).max(),
                                       vmin=-abs(trafo_array).max())
                        pylab.xticks(())
                        pylab.yticks(())
                        # pylab.colorbar()
                        pylab.contour(
                            base_array,
                            levels=[50],
                            colors=colors[1],
                            origin="image")

                        # store and clean
                        f_name = str(node_dir) + str(os.sep) + "classifier_" \
                            + str(index)
                        pylab.savefig(f_name + ".png", bbox_inches='tight')
                        pylab.close(fig)
                        fig = pylab.figure(figsize=(4, 4), dpi=300)

                        # SUBPLOT 2: plot of the derivative multiplied with covariance
                        # pylab.subplot(2,2,2)
                        # pylab.title("Backtransformation times Covariance")

                        pylab.contourf(covariance_array, 50, cmap="jet",
                                       origin="image",
                                       vmax=abs(covariance_array).max(),
                                       vmin=-abs(covariance_array).max())
                        pylab.xticks(())
                        pylab.yticks(())
                        # pylab.colorbar()
                        pylab.contour(
                            base_array,
                            levels=[50],
                            colors=colors[1],
                            origin="image")

                        # # SUBPLOT 2: plot of the original feature vector
                        # pylab.subplot(2,2,2)
                        # pylab.title("Original data")
                        #
                        # pylab.contourf(base_array, 50, cmap="binary", origin="image")
                        # pylab.xticks(())
                        # pylab.yticks(())
                        # pylab.colorbar()

                        # # SUBPLOT 3: plot of the difference between vectors
                        # pylab.subplot(2,2,3)
                        # pylab.title("Addition")
                        #
                        # pylab.contourf(trafo_array+base_array, 50, cmap="spectral", origin="image")
                        # pylab.xticks(())
                        # pylab.yticks(())
                        # pylab.colorbar()
                        #
                        # # SUBPLOT 4: plot of the difference between vectors
                        # pylab.subplot(2,2,4)
                        # pylab.title("Subtraction")
                        #
                        # pylab.contourf(base_array-trafo_array, 50, cmap="spectral", origin="image")
                        # pylab.xticks(())
                        # pylab.yticks(())
                        # pylab.colorbar()

                        # pdf.savefig(fig, bbox_inches='tight')
                                                # store and clean
                        f_name = str(node_dir) + str(os.sep) + \
                            "classifier_cov_" + str(index)
                        pylab.savefig(f_name + ".png", bbox_inches='tight')
                        pylab.close(fig)

                        if index == self.max_samples:
                            break

                    # d = pdf.infodict()
                    # d['Title'] = 'Sample visualization'
                    # # d['Author'] = ''
                    # # d['Subject'] = ''
                    # # d['Keywords'] = ''
                    # d['CreationDate'] = datetime.datetime.today()
                    # d['ModDate'] = datetime.datetime.today()
            pylab.close('all')

    def is_trainable(self):
        """ Labels are required for visualization """
        return True

    def is_supervised(self):
        """ Labels are required for visualization """
        return True

_NODE_MAPPING = {"LLE_Vis": LLEVisNode}
