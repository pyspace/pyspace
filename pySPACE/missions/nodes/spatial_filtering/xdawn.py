""" xDAWN and variants for enhancing event-related potentials """

import os
import cPickle
from copy import deepcopy, copy
import warnings

import numpy
from pySPACE.resources.dataset_defs.metric import BinaryClassificationDataset


try:
    import scipy
    if map(int, __import__("scipy").__version__.split('.')) < [0,8,0]:
        from scipy.linalg.decomp import qr
    else:
        from scipy.linalg import qr
except:
    pass

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.missions.nodes.spatial_filtering.spatial_filtering \
    import SpatialFilteringNode

from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.resources.dataset_defs.stream import StreamDataset

from pySPACE.tools.filesystem import create_directory
import logging

class XDAWNNode(SpatialFilteringNode):
    """ xDAWN spatial filter for enhancing event-related potentials.
    
    xDAWN tries to construct spatial filters such that the 
    signal-to-signal plus noise ratio is maximized. This spatial filter is 
    particularly suited for paradigms where classification is based on 
    event-related potentials.
    
    For more details on xDAWN, please refer to 
    http://www.icp.inpg.fr/~rivetber/Publications/references/Rivet2009a.pdf

    **References**

        ========= ==========================================================================================
        main      source: xDAWN
        ========= ==========================================================================================
        author      Rivet, B. and Souloumiac, A. and Attina, V. and Gibert, G.
        journal     Biomedical Engineering, IEEE Transactions on
        title       xDAWN Algorithm to Enhance Evoked Potentials: Application to Brain-Computer Interface
        year        2009
        month       aug.
        volume      56
        number      8
        pages       2035 -2043
        doi         10.1109/TBME.2009.2012869
        ISSN        0018-9294
        ========= ==========================================================================================



    **Parameters**
        :erp_class_label: Label of the class for which an ERP should be evoked. 
            For instance "Target" for a P300 oddball paradigm.
            
            (*recommended, default: 'Target'*)
    
        :retained_channels: Determines how many of the pseudo channels
            are retained. Default is None which means "all channels".

            (*optional, default: None*)
            
        :load_filter_path: An absolute path from which the spatial filters can
            be loaded. If not specified, these filters are learned from the 
            training data.

            (*optional, default: None*)

        :visualize_pattern: If value is true, a visualization of the learned
            spatial filters is stored.

            The visualisation is divided into two components.
            First of all each transformation is visualized separately.
            Since the visualization itself may not be so meaningful,
            there exists another combined visualization, which shows
            the filter (u_i) with the underlying spatial distribution
            (w_i, parameter names taken from paper).
            The number of filters equals the number of original channels.
            Normally only the first channels matter and the rest corresponds to
            different noise components.

            To avoid storing to many pictures, the *retained_channels*
            parameter is used to restrict the number.

            (*optional, default: False*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : xDAWN
            parameters:
                erp_class_label : "Target"
                retained_channels : 32
                store : True

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2011/07/05
    """
    def __init__(self, erp_class_label=None, retained_channels=None, 
                 load_filter_path=None, visualize_pattern=False,  **kwargs):
        # Must be set before constructor of superclass is called
        self.trainable = (load_filter_path is None)

        super(XDAWNNode, self).__init__(retained_channels=retained_channels,
                                        **kwargs)
        
        if erp_class_label is None:
            erp_class_label = "Target"
            self._log("No ERP class label given. Using default: 'Target'.",
                      level=logging.CRITICAL)
        filters = None
        # Load patterns from file if requested
        if not load_filter_path is None:
            filters_file = open(load_filter_path, 'r')
            filters = cPickle.load(filters_file)
            filters_file.close()
        
        self.set_permanent_attributes(
            # Label of the class for which an ERP should be evoked.
            erp_class_label=erp_class_label,
            # The channel names
            channel_names=None,
            # Matrices for storing data and stimuli  
            X=None,
            D=None,
            SNR=None,
            # The number of channels that will be retained
            retained_channels=retained_channels,
            # whether this node is trainable
            trainable=self.trainable,
            # After training is finished, this attribute will contain
            # the spatial filters that are used to project
            # the data onto a lower dimensional subspace
            filters=filters,
            # Determines whether the filters are stored after training
            visualize_pattern=visualize_pattern,
            xDAWN_channel_names=None)
        if self.visualize_pattern:
            self.set_permanent_attributes(store=True)
    
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return self.trainable
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return self.trainable

    def _train(self, data, label):
        """ Train node on given example *data* for class *label*. """
        # If this is the first data sample we obtain
        if self.channel_names is None:
            self.channel_names = data.channel_names
            if self.retained_channels in [None, 'None']:
                self.retained_channels = len(self.channel_names)
                
        if len(self.channel_names) < self.retained_channels:
            self.retained_channels = len(self.channel_names)
            self._log("To many channels chosen for the retained channels! "
                      "Replaced by maximum number.", level=logging.CRITICAL)

        # Iteratively construct Toeplitz matrix D and data matrix X
        if label == self.erp_class_label:
            D = numpy.diag(numpy.ones(data.shape[0]))
        else:
            D = numpy.zeros((data.shape[0], data.shape[0]))
            
        if self.X is None:
            self.X = deepcopy(data)
            self.D = D
        else:
            self.X = numpy.vstack((self.X, data))
            self.D = numpy.vstack((self.D, D))
    
    def _stop_training(self, debug=False):
        # The following if statement is needed only to account for
        # different versions of scipy
        if map(int, __import__("scipy").__version__.split('.')) >= [0, 9, 0]:
            # NOTE: mode='economy'required since otherwise the memory consumption is excessive
            # QR decompositions of X
            Qx, Rx = qr(self.X, overwrite_a=True, mode='economic')
            # QR decompositions of D
            Qd, Rd = qr(self.D, overwrite_a=True, mode='economic')
        else:
            # NOTE: econ=True required since otherwise
            #       the memory consumption is excessive
            # QR decompositions of X
            Qx, Rx = qr(self.X, overwrite_a=True, econ=True)
            # QR decompositions of D
            Qd, Rd = qr(self.D, overwrite_a=True, econ=True)
            
        #self.X = None # Free memory
        #self.D = None # Free memory             

        # Singular value decomposition of Qd.T Qx
        # NOTE: full_matrices=True required since otherwise we do not get 
        #       num_channels filters. 
        self.Phi, self.Lambda, self.Psi = \
                    numpy.linalg.svd(numpy.dot(Qd.T, Qx), full_matrices=True)
        self.Psi = self.Psi.T
       
        SNR = numpy.zeros(self.X.shape[1])
        # Construct the spatial filters
        for i in range(self.Psi.shape[1]):
            # Construct spatial filter with index i as Rx^-1*Psi_i
            ui = numpy.dot(numpy.linalg.inv(Rx), self.Psi[:,i])
            wi = numpy.dot(Rx.T, self.Psi[:,i]) 
            if i < self.Phi.shape[1]:
                ai = numpy.dot(numpy.dot(numpy.linalg.inv(Rd), self.Phi[:,i]),
                               self.Lambda[i])
            if i == 0:
                self.filters = numpy.atleast_2d(ui).T
                self.wi = numpy.atleast_2d(wi)
                self.ai = numpy.atleast_2d(ai)
            else:
                self.filters = numpy.hstack((self.filters,
                                             numpy.atleast_2d(ui).T))
                self.wi = numpy.vstack((self.wi, numpy.atleast_2d(wi)))
                if i < self.Phi.shape[1]:
                    self.ai = numpy.vstack((self.ai, numpy.atleast_2d(ai)))
            a = numpy.dot(self.D, ai.T)
            b = numpy.dot(self.X, ui)
#            b.view(numpy.ndarray)
#            bb = numpy.dot(b.T, b)
#            aa = numpy.dot(a.T, a)
            SNR[i] = numpy.dot(a.T, a)/numpy.dot(b.T, b)

        self.SNR = SNR
        self.D = None
        self.X = None

    def _execute(self, data):
        """ Apply the learned spatial filters to the given data point """
        if self.channel_names is None:
            self.channel_names = data.channel_names

        if self.retained_channels in [None, 'None']:
                self.retained_channels = len(self.channel_names)

        if len(self.channel_names)<self.retained_channels:
            self.retained_channels = len(self.channel_names)
            self._log("To many channels chosen for the retained channels! "
                      "Replaced by maximum number.", level=logging.CRITICAL)
        data_array=data.view(numpy.ndarray)
        # Project the data using the learned spatial filters
        projected_data = numpy.dot(data_array,
                                   self.filters[:, :self.retained_channels])
        
        if self.xDAWN_channel_names is None:
            self.xDAWN_channel_names = ["xDAWN%03d" % i 
                                        for i in range(self.retained_channels)]
        
        return TimeSeries(projected_data, self.xDAWN_channel_names,
                          data.sampling_frequency, data.start_time,
                          data.end_time, data.name, data.marker_name)

    def store_state(self, result_dir, index=None): 
        """ Stores this node in the given directory *result_dir* """
        if self.store:
            try:
                node_dir = os.path.join(result_dir, self.__class__.__name__)
                create_directory(node_dir)
                # This node only stores the learned spatial filters
                name = "%s_sp%s.pickle" % ("patterns", self.current_split)
                result_file = open(os.path.join(node_dir, name), "wb")
                result_file.write(cPickle.dumps((self.filters, self.wi,
                                                 self.ai), protocol=2))
                result_file.close()
                
                # Stores the signal to signal plus noise ratio resulted
                # by the spatial filter
                #fname = "SNR_sp%s.csv" % ( self.current_split)
                #numpy.savetxt(os.path.join(node_dir, fname), self.SNR,
                #    delimiter=',', fmt='%2.5e')
                
                # Store spatial filter plots if desired
                if self.visualize_pattern:
                    from pySPACE.missions.nodes.spatial_filtering.csp \
                        import CSPNode
                    # Compute, accumulate and analyze signal components
                    # estimated by xDAWN
                    vmin = numpy.inf
                    vmax = -numpy.inf
    
                    signal_components = []
                    lambda_sum = sum(self.Lambda)
                    complete_signal = numpy.zeros((self.wi.shape[1],
                                                   self.ai.shape[1]))
                    for filter_index in range(self.retained_channels):
                        #self.ai.shape[0]):
                        signal_component = numpy.outer(self.wi[filter_index, :], 
                                                       self.ai[filter_index, :])
                        vmin = min(signal_component.min(), vmin)
                        vmax = max(signal_component.max(), vmax)
                        
                        signal_components.append(signal_component)
                        complete_signal += signal_component
                    # Plotting
                    import pylab
                    for index, signal_component in enumerate(signal_components):
                        pylab.figure(0, figsize=(18,8))
                        pylab.gcf().clear()
                        
                        # Plot spatial distribution
                        ax=pylab.axes([0.0, 0.0, 0.2, 0.5])
                        CSPNode._plot_spatial_values(ax, self.wi[index, :], 
                                                     self.channel_names,
                                                     'Spatial distribution')
                        # Plot spatial filter
                        ax=pylab.axes([0.0, 0.5, 0.2, 0.5])
                        CSPNode._plot_spatial_values(ax, self.filters[:, index],
                                                     self.channel_names,
                                                     'Spatial filter')
                        # Plot signal component in electrode coordinate system 
                        self._plotTimeSeriesInEC(signal_component, vmin=vmin, 
                                                 vmax=vmax,
                                                 bb=(0.2, 1.0, 0.0, 1.0))
                        
                        pylab.savefig("%s%ssignal_component%02d.png" 
                                      % (node_dir, os.sep, index))
    
                    CSPNode._store_spatial_filter_plots(
                        self.filters[:, :self.retained_channels],
                        self.channel_names, node_dir)
                    # Plot entire signal
                    pylab.figure(0, figsize=(15, 8))
                    pylab.gcf().clear()
                    self._plotTimeSeriesInEC(
                        complete_signal,
                        file_name="%s%ssignal_complete.png" % (node_dir, os.sep))
                    pylab.savefig("%s%ssignal_complete.png" % (node_dir, os.sep))

            except Exception as e:
                print e
                raise
        super(XDAWNNode, self).store_state(result_dir)

    def _plotTimeSeriesInEC(self, values, vmin=None, vmax=None, 
                            bb=(0.0, 1.0, 0.0, 1.0), file_name=None):
        # Plot time series in electrode coordinate system, i.e. the values of
        # each channel at the position of the channel
        import pylab
        
        ec = self.getMetadata("electrode_coordinates")
        if ec is None:
            ec = StreamDataset.ec
        
        ec_2d = StreamDataset.project2d(ec)
        
        # Define x and y coordinates of electrodes in the order of the channels
        # of data
        x = numpy.array([ec_2d[key][0] for key in self.channel_names])
        y = numpy.array([ec_2d[key][1] for key in self.channel_names])
        
        # Determine min and max values
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()
        
        width = (bb[1] - bb[0])
        height = (bb[3] - bb[2])
        for channel_index, channel_name in enumerate(self.channel_names):
            ax = pylab.axes([x[channel_index]/(1.2*(x.max() - x.min()))*width +
                            bb[0] + width/2 - 0.025,
                            y[channel_index]/(1.2*(y.max() - y.min()))*height +
                            bb[2] + height/2 - 0.0375, 0.05, 0.075])
            ax.plot(values[channel_index, :], color='k', lw=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim((vmin, vmax))
            ax.text(values.shape[1]/2, vmax*.8, channel_name, 
                    horizontalalignment='center', verticalalignment='center')


class SparseXDAWNNode(XDAWNNode):
    """ Sparse xDAWN spatial filter for enhancing event-related potentials.
    
    xDAWN tries to construct spatial filters such that the 
    signal-to-signal plus noise ratio (SSNR) is maximized. This spatial filter
    is particularly suited for paradigms where classification is based on
    event-related potentials. In contrast to the standard xDAWN algorithm,
    this node tries to minimize the electrodes that have non-zero weights in
    the spatial filters while at the same time trying to maximize the 
    signal-to-signal plus noise ratio. This property is used for electrode 
    selection, i.e. only those electrodes need to be set that obtained non-zero
    weights.  
    
    For more details on Sparse xDAWN, please refer to 
    http://www.gipsa-lab.inpg.fr/~bertrand.rivet/references/RivetEMBC10.pdf
    
    .. todo:: Two more sentences about Sparse_XDAWN
    
    **Parameters**
    
        :`lambda_`: Determines the relative influence of the two objectives
            (maximization of SSNR and minimization of electrodes with non-zero
            weights). If `lambda_` is 0, only the SSNR is relevant (like in
            standard xDAWN). The larger `lambda_`, the weaker is the influence 
            of the SSNR. 
    
        :erp_class_label: Label of the class for which an ERP should be evoked. 
            For instance "Target" for a P300 oddball paradigm.
            
            (*recommended, default:'Target'*)
    
        :num_selected_electrodes: Determines how many electrodes keep a non-zero 
            weight.

    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : Sparse_xDAWN
            parameters :
                lambda_ : 0.1
                erp_class_label : "Target"
                num_selected_electrodes : 2
                store : True

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2011/08/22
    """
    
    def __init__(self, lambda_, erp_class_label='Target', num_selected_electrodes=None, 
                 **kwargs):
        if 'retained_channels' in kwargs:
            kwargs.pop('retained_channels')
        super(SparseXDAWNNode, self).__init__(erp_class_label=erp_class_label, 
                                                retained_channels=None, 
                                                load_filter_path=None, 
                                                visualize_pattern=False,
                                                 **kwargs)

        self.set_permanent_attributes(lambda_ = lambda_,
                                      num_selected_electrodes=num_selected_electrodes)

    def _stop_training(self, debug=False):
        if self.num_selected_electrodes is None:
            self.num_selected_electrodes = self.retained_channels
        # Estimate of the signal for class 1 (the erp_class_label class)
        A_1 = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(self.D.T, self.D)),
                                self.D.T),
                       self.X)
        # Estimate of Sigma 1 and Sigma X
        Sigma_1 = numpy.dot(numpy.dot(numpy.dot(A_1.T, self.D.T),
                                    self.D), A_1)
        Sigma_X = numpy.dot(self.X.T, self.X)

        # The objective function from the paper from Rivet et al.
        def objective_function(v_1, lambda_):
            a = numpy.dot(numpy.dot(v_1.T, Sigma_1), v_1) #0-d, skip trace!
            b = numpy.dot(numpy.dot(v_1.T, Sigma_X), v_1) #0-d, skip trace!
            c = numpy.linalg.norm(v_1, 1) / numpy.linalg.norm(v_1, 2)
            
            return a / b - lambda_*c
        
        # Compute the non-pruned weights
        v_1 = self._gradient_optimization(
            objective_function=lambda v_1: objective_function(v_1, self.lambda_),
            Sigma_1=Sigma_1, Sigma_X=Sigma_X, max_evals=25000)
        # Prune weight vector such that only self.num_selected_electrodes keep 
        # entries != 0 (those with the largest weight)
        threshold = sorted(numpy.absolute(v_1))[-self.num_selected_electrodes]
        v_1[numpy.absolute(v_1) < threshold] = 0
        v_1 /= numpy.linalg.norm(v_1, 2)

        # Determine indices and names of electrodes with non-zero weights
        self.selected_indices = list(numpy.where(numpy.absolute(v_1) > 0)[0])
        self.selected_channels = [self.channel_names[index]
                                  for index in self.selected_indices]

    def _gradient_optimization(self, objective_function, Sigma_1, Sigma_X,
                               max_evals=25000, **kwargs):
        best_f_value = -numpy.inf
        best_v_1 = None
        evals = 0
        rep = 0

        # Start several repetitions at random start states
        while True:
            rep += 1
            # Initialize electrode weight vector randomly
            v_1 = numpy.random.random(self.X.shape[1])
            v_1 /= numpy.linalg.norm(v_1, 2)

            # Set initial learning rate
            rho = 1.0
            
            # Gradient ascent until we are very close to a local maximum 
            while rho > 10**-5:
                # Some intermediate results
                a = numpy.dot(Sigma_X, v_1)
                b = numpy.dot(v_1.T, a)
                c = numpy.dot(Sigma_1, v_1)
                d = numpy.dot(v_1.T, c)
                
                e = numpy.dot(numpy.diag(numpy.sign(v_1)), numpy.ones(self.X.shape[1])) \
                                / numpy.linalg.norm(v_1, 2)
                f = numpy.dot(numpy.linalg.norm(v_1, 1) / (numpy.dot(v_1.T, v_1)**1.5), 
                             v_1)
                
                # Subgradient components
                sg1 = 2.0/b*(c - d/b*a)
                sg2 = e - f
                
                # Construct subgradient
                subgradient = sg1 - self.lambda_ * sg2
                               
                # Search for a learning rate such that following the gradient
                # does not bring us too far ahead of the optimum
                v_1_old = numpy.array(v_1)
                old_f_value = objective_function(v_1)
                while True:
                    evals += 1
                    # Update and renormalize weight vector v
                    v_1 += rho * subgradient
                    v_1 /= numpy.linalg.norm(v_1, 2)
                    
                    # Check if the current learning rate is too large
                    if objective_function(v_1) >= old_f_value:
                        # Not followed gradient too far, increase learning rate
                        # and break
                        rho /= 0.9
                        break
                
                    # Reduce learning rate and restore original v_1
                    rho *= 0.9                    
                    v_1 = numpy.array(v_1_old)
                    
                    # If the learning rate becomes too low, we break
                    if rho < 10**-5:
                        break
                    
                # Break if we have spent the allowed time searching the maximum
                if evals >= max_evals: break
                
            # Check if we have found a new optimum in this repetition    
            if objective_function(v_1) > best_f_value:
                best_f_value = objective_function(v_1)
                best_v_1 = v_1
            
            # Return if we have spent the allowed time searching the maximum
            if evals >= max_evals:
                return best_v_1
            
    def _execute(self, data):
        """ Project the data onto the selected channels. """
        projected_data = data[:, self.selected_indices]
                
        return TimeSeries(projected_data, self.selected_channels,
                          data.sampling_frequency, data.start_time,
                          data.end_time, data.name, data.marker_name)
            
    def store_state(self, result_dir, index=None): 
        """ Stores this node in the given directory *result_dir* """
        if self.store:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            create_directory(node_dir)
             
            # This node only stores which electrodes have been selected
            name = "%s_sp%s.txt" % ("electrode_selection", self.current_split)
            result_file = open(os.path.join(node_dir, name), "w")

            result_file.write(str(self.selected_channels))
            result_file.close()

    def get_filters(self):
        raise NotImplementedError("Sparse xDAWN is yet not fitting for ranking "
                                  "electrode selection.")


class SSNR(BaseNode):
    """ Helper-class that encapsulates SSNR related computations.
    
    Use as follows: add training examples one-by-one along with their labels
    using the method add_example. Once all training data has been added, metrics
    values can be computed using ssnr_as, ssnr_vs, and ssnr_vs_test
    
    """
    
    def __init__(self, erp_class_label, retained_channels=None):
        self.retained_channels = retained_channels
        self.erp_class_label = erp_class_label
        
        self.X = None # The data matrix (will be constructed iteratively)
        self.D = None # The Toeplitz matrix (will be constructed iteratively)
                
    def add_example(self, data, label):
        """ Add the example *data* for class *label*. """
        if self.retained_channels is None:
            self.retained_channels = data.shape[1]
        else:
            self.retained_channels = min(self.retained_channels, data.shape[1])
        
        # Iteratively construct Toeplitz matrix D and data matrix X
        if label == self.erp_class_label:
            D = numpy.diag(numpy.ones(data.shape[0]))
        else:
            D = numpy.zeros((data.shape[0], data.shape[0]))
            
        if self.X is None:
            self.X = deepcopy(data)
            self.D = D
        else:
            self.X = numpy.vstack((self.X, data))
            self.D = numpy.vstack((self.D, D))
                            
    def ssnr_as(self, selected_electrodes=None):
        """ SSNR for given electrode selection in actual sensor space. 
        
        If no electrode selection is given, the SSNR of all electrodes is 
        computed.
        """
        if selected_electrodes is None:
            selected_electrodes = range(self.X.shape[1])
            
        self.Sigma_1, self.Sigma_X = self._compute_Sigma(self.X, self.D)
        
        filters = numpy.zeros(shape=(self.X.shape[1], self.X.shape[1]))
        for electrode_index in selected_electrodes:
            filters[electrode_index, electrode_index] = 1

        # Return the SSNR that these filters would obtain on training data            
        return self._ssnr(filters, self.Sigma_1, self.Sigma_X)

    def ssnr_vs(self, selected_electrodes=None):
        """ SSNR for given electrode selection in virtual sensor space. 
        
        If no electrode selection is given, the SSNR of all electrodes is 
        computed.
        """
        if selected_electrodes is None:
            selected_electrodes = range(self.X.shape[1])
            
        self.Sigma_1, self.Sigma_X = self._compute_Sigma(self.X, self.D)
        
        # Determine spatial filter using xDAWN that would be obtained if
        # only the selected electrodes would be available
        partial_filters = \
            self._compute_xDAWN_filters(self.X[:, selected_electrodes], 
                                        self.D)
        # Expand partial filters to a filter for all electrodes (by setting
        # weights of inactive electrodes to 0)
        filters = numpy.zeros((self.X.shape[1], self.retained_channels))
        # Iterate over electrodes with non-zero weights
        for index, electrode_index in enumerate(selected_electrodes):
            # Iterate over non-zero spatial filters
            for j in range(min(filters.shape[1], partial_filters.shape[1])):
                filters[electrode_index, j] = partial_filters[index, j]

        # Return the SSNR that these filters would obtain on training data            
        return self._ssnr(filters, self.Sigma_1, self.Sigma_X)
    
    def ssnr_vs_test(self, X_test, D_test, selected_electrodes=None):
        """ SSNR for given electrode selection in virtual sensor space. 
        
        Note that the training of the xDAWN spatial filter for mapping to 
        virtual sensor space and the computation of the SSNR in this virtual 
        sensor space are done on different data sets. 
        
        If no electrode selection is given, the SSNR of all electrodes is 
        computed.
        """
        if selected_electrodes is None:
            selected_electrodes = range(self.X.shape[1])
                    
        # Determine spatial filter using xDAWN that would be obtained if
        # only the selected electrodes would be available
        partial_filters = \
            self._compute_xDAWN_filters(self.X[:, selected_electrodes], 
                                        self.D)
        # Expand partial filters to a filter for all electrodes (by setting
        # weights of inactive electrodes to 0)
        filters = numpy.zeros((self.X.shape[1], self.retained_channels))
        # Iterate over electrodes with non-zero weights
        for index, electrode_index in enumerate(selected_electrodes):
            # Iterate over non-zero spatial filters
            for j in range(min(filters.shape[1], partial_filters.shape[1])):
                filters[electrode_index, j] = partial_filters[index, j]

        # Return the SSNR that these filters would obtain on test data
        Sigma_1_test, Sigma_X_test = self._compute_Sigma(X_test, D_test)
        return self._ssnr(filters, Sigma_1_test, Sigma_X_test)
        
    def _compute_Sigma(self, X, D):
        if D is None:
            warnings.warn("No data given for sigma computation.")
        elif not(1 in D):
            warnings.warn("No ERP data (%s) provided." % self.erp_class_label)
        # Estimate of the signal for class 1 (the erp_class_label class)
        A_1 = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(D.T, D)), D.T), X)
        # Estimate of Sigma 1 and Sigma X
        Sigma_1 = numpy.dot(numpy.dot(numpy.dot(A_1.T, D.T), D), A_1)
        Sigma_X = numpy.dot(X.T, X)
        
        return Sigma_1, Sigma_X        
        
    def _ssnr(self, v, Sigma_1, Sigma_X):
        # Compute SSNR after filtering  with v.
        a = numpy.trace(numpy.dot(numpy.dot(v.T, Sigma_1), v))
        b = numpy.trace(numpy.dot(numpy.dot(v.T, Sigma_X), v))
        return a / b
    
    def _compute_xDAWN_filters(self, X, D):
        # Compute xDAWN spatial filters
                      
        # QR decompositions of X and D
        if map(int, __import__("scipy").__version__.split('.')) >= [0,9,0]:
            # NOTE: mode='economy'required since otherwise the memory 
            # consumption is excessive
            Qx, Rx = qr(X, overwrite_a=True, mode='economic')     
            Qd, Rd = qr(D, overwrite_a=True, mode='economic')
        else:
            # NOTE: econ=True required since otherwise the memory consumption 
            # is excessive 
            Qx, Rx = qr(X, overwrite_a=True, econ=True)      
            Qd, Rd = qr(D, overwrite_a=True, econ=True)       

        # Singular value decomposition of Qd.T Qx
        # NOTE: full_matrices=True required since otherwise we do not get 
        #       num_channels filters. 
        Phi, Lambda, Psi = numpy.linalg.svd(numpy.dot(Qd.T, Qx), 
                                           full_matrices=True)
        Psi = Psi.T

        # Construct the spatial filters
        for i in range(Psi.shape[1]):
            # Construct spatial filter with index i as Rx^-1*Psi_i
            ui = numpy.dot(numpy.linalg.inv(Rx), Psi[:,i])
            if i == 0:
                filters = numpy.atleast_2d(ui).T
            else:
                filters = numpy.hstack((filters, numpy.atleast_2d(ui).T))
                
        return filters


_NODE_MAPPING = {"xDAWN": XDAWNNode,
                 "Sparse_xDAWN": SparseXDAWNNode}
