""" Visualize data in time-amplitude or time-frequency representations

This module contains nodes that can be used to visualize
the data as time series. All these nodes use the functionality
of the VisualizationBaseNode.

"""
import numpy
import math

try:
    import pylab
except:
    pass
#import itertools
#from collections import defaultdict
import warnings
import os

try:
    from matplotlib import mlab
except:
    pass

#import logging
#from pySPACE.tools.memoize_generator import MemoizeGenerator

from pySPACE.missions.nodes.visualization.base import VisualizationBase
from pySPACE.tools.filesystem import  create_directory
from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.resources.dataset_defs.stream import StreamDataset


#delete me after conversion to vis supernode
from pySPACE.missions.nodes.base_node import BaseNode


class TimeSeriesPlotNode(VisualizationBase):
    """A node that allows to monitor the processing of time series
    
    This node plots the time series data either in one column for all channels
    or for a single selected channel. The node inherits the
    functionality of the VisualisationBase.
    
    See documentation of :mod:`VisualisationBase <pySPACE.missions.nodes.visualization.base>` 
    to view basic functionality.
    If not specified differently using the parameters below, the data is plotted
    in one matrix for each class (using pylab.matshow).
    
    **Parameters**

      :channel_names:
        If one channel_name is given, only information about
        this channel is plotted. If more channels are specified,
        they are plotted separately in one subplot 
        (forces separate_channels to True).
        
        (*optional, default: None*)
    
      :separate_channels:
        Each channel gets a separate subplot (amplitude vs time),
        either arranged in rows and columns or arranged according to
        position on the head (see parameter physiological_arrangement in base class).
        (*optional, default: False*)
        
      :class_difference:
        If this is True, the created plot shows the difference between the two classes.
        This option only works if 2 classes are present at the same time!
        (*optional, default: False*)
        
    .. image:: ../../graphics/time_series_plot.png
        :width: 1024   
        
    **Exemplary Call**
    
    .. code-block:: yaml

        -
             node : Time_Series_Plot
             parameters :
                  averaging : True
                  online : True
                  separate_channels: True

    :Author: Sirko Straube (sirko.straube@dfki.de)
    :Date of Last Revision: 2012/12/21                            
    """
    input_types = ["TimeSeries"]
    def __init__(self, channel_names=None, separate_channels = False, class_difference=False, **kwargs):       
        
        super(TimeSeriesPlotNode, self).__init__(**kwargs)
        
        #if the user specified more than one channel
        if channel_names and len(channel_names)>1:
            separate_channels = True
        
        self.set_permanent_attributes(channel_names=channel_names,
                                      separate_channels = separate_channels,
                                      min_value = pylab.inf,
                                      max_value = -pylab.inf,
                                      class_difference=class_difference)       
                    
    def reset(self):
        """
        Reset the state of the object to the clean state it had after its
        initialization
        """
        super(TimeSeriesPlotNode, self).reset()
        
    def _plotValues(self,
                    values,            #dict  TimeSeries values
                    plot_label,        #str   Plot-Label
                    fig_num,           #int   Figure-number for classify plots
                                       #      1: average,
                                       #      2: single trial,
                                       #      3: average accumulating
                    store_dir = None,  #str   Directory to store the plots
                    counter=0):        #int   Plotcounter for all trials
        
        if self.class_difference:
            #we have to compute the difference
            list_of_classes = values.keys()
            num_of_classes = len(list_of_classes)
            if num_of_classes != 2: 
                warnings.warn("TimeSeriesPlot:: Difference plots only possible for two classes  Plotting ignored!")
                return
            data1 = values[list_of_classes[0]].view(numpy.ndarray)
            data2 = values[list_of_classes[1]].view(numpy.ndarray)

            #construct new TimeSeries with differences
            data = TimeSeries(data1 - data2, 
                              channel_names=values[list_of_classes[0]].channel_names, 
                              sampling_frequency=values[list_of_classes[0]].sampling_frequency)
            #overwrite old incoming values
            values=dict(difference=data)

        sampling_frequency = values[values.keys()[0]].sampling_frequency
        list_of_classes = values.keys()
        num_of_classes = len(list_of_classes)
             
        #computing time points to show
        num_tpoints = values.values()[0].shape[0]
        all_tpoints = numpy.arange(0,
                                   num_tpoints * (1000/sampling_frequency),
                                   1000 / sampling_frequency) + self.timeshift    

        pylab.subplots_adjust(left  = 0.05,  # the left side of the subplots of the figure
                      right = 0.95,    # the right side of the subplots of the figure
                      bottom = 0.05,   # the bottom of the subplots of the figure
                      top = 0.95,      # the top of the subplots of the figure
                      wspace = 0.15,   # the amount of width reserved for blank space between subplots
                      hspace = 0.1,   # the amount of height reserved for white space between subplots                          
                      )

        f=pylab.figure(fig_num, figsize = (18,13))
        #assure that figure is displayed with an interactive backend
        if pylab.get_backend() in pylab.matplotlib.backends.interactive_bk:
                f.show()
        #plot all channels separately in one figure
        if self.separate_channels:
            self._plot_all_channels_separated(all_tpoints, values)
        else:
            for index, class_label in enumerate(list_of_classes):
           
                data = values[class_label].view(numpy.ndarray)
                
                axis = pylab.subplot(1, num_of_classes, index + 1)
                pylab.gca().clear()
                
                #plot just 1 channel
                if self.channel_names != None and len(self.channel_names)==1:
                    assert (self.channel_names[0] in values[class_label].channel_names),\
                            "TimeSeriesPlot::Channel requested for plotting is not available in data"
                    channel_index = values[class_label].channel_names.index(self.channel_names[0])                     
                    # Update minimal and maximal value
                    self.min_value = min(self.min_value, 
                                         min(data[:, channel_index].flatten()))
                    self.max_value = max(self.max_value, 
                                         max(data[:, channel_index].flatten()))
                    
                    title = pylab.title(str(self.channel_names) + ' ' + class_label)
                    pylab.plot(all_tpoints, data[:, channel_index])
    
                #plot all channels as matrix    
                else:
                    self._plot_all_channels(data, all_tpoints, values[class_label].channel_names)
                    title = pylab.title(class_label)
                
                title.set_fontsize(24)
                pylab.draw()
                          
        
        # Draw or store the figure
        if store_dir is None:
            pylab.draw()
        else:
            current_split=self.current_split
            if current_split != 0 and not\
            plot_label.endswith('_split_' + str(current_split)): #more than one split and first call
                plot_label = plot_label + '_split_' + str(current_split)
                
            f_name=str(store_dir) + str(os.sep) + str(plot_label)
            pylab.savefig(f_name + ".png")         
              
    def _plot_all_channels(self, data, tpoints, channel_names):
            
        if not self.class_difference:
            # Normalize the data to be plotted so that all values are 
            # between 0 and 1. This is useful since the same color 
            # in the two plots corresponds to the same value 
            min_value = min(data.ravel())
            max_value = max(data.ravel())
        
            plot_values = (data - min_value) / (max_value - min_value)
        else:
            plot_values=data
            
        im = pylab.matshow(plot_values.T, fignum = False, aspect = 4.0)

        ax = pylab.gca()
        #ax.set_xticks(pylab.linspace(0.0, plot_values.shape[0], 20))
        ax.set_xticklabels(map(lambda s: "%.2f" %s,tpoints))
        labels = ax.get_xticklabels()
        pylab.setp(labels, rotation=-45)
        
        ax.set_yticks(range(plot_values.shape[1]))
        ax.set_yticklabels(channel_names)
        pylab.xlabel("time (ms)")
        pylab.ylabel("channels")
        #pylab.colorbar(im)      
        pylab.draw()   
    
    def _plot_all_channels_separated(self, tpoints, values):          
        """ This function generates time series plot separated for each channel
        """
    #def _generate_time_series_plot(self, label, data):
        f=pylab.gcf()
        
        # in case of [phys_arrangement], write the figure title only once
        # and large in the upper left corner of the plot. this fails whenever
        # there are no more channels in that area, as the plot gets cropped
#        if self.physiological_arrangement:
#            h, l = pylab.gca().get_legend_handles_labels()
#            prop = pylab.matplotlib.font_manager.FontProperties(size='xx-large')
#            f.legend(h, l, prop=prop, loc=1)
#            text_x = .4
#            text_y = .4
#    
#            #if self.shrink_plots: text_y = 1.2
#            f.text(text_x, text_y, 'Channel-wise time series\n' +
#                       samples_per_condition_string,
#                       ha='center', color='black', size=32)
     
        for index, class_label in enumerate(values.keys()):
           
            data = values[class_label].view(numpy.ndarray)
            channel_names=values[class_label].channel_names

            line_color = 'bgrcmyk'[index]
            
            number_of_channels=len(channel_names)
            
            # Compute number of rows and cols for subplot-arrangement:
            # use 8 as upper limit for cols and compute rows accordingly
            if number_of_channels <= 8: 
                nr_of_cols = number_of_channels
            else: 
                nr_of_cols = 8
            nr_of_rows = (number_of_channels - 1) / 8 + 1
            
            # Set canvas size in inches. These values turned out fine, depending
            # on [physiological_arrangement]
            #if not self.physiological_arrangement:
            #    f.set_size_inches((5 * nr_of_cols,  3 * nr_of_rows))
            #else:
                #if not self.shrink_plots:
                #    figure.set_size_inches((3*11.7,  3*8.3))
            #    f.set_size_inches((4*11.7,  4*8.3))
            
            
            ec = self.getMetadata("electrode_coordinates")
            if ec is None:
                ec = StreamDataset.ec
            
            ec_2d = StreamDataset.project2d(ec)

            # plot everything channel-wise
            for channel_index in range(number_of_channels):
                channel_name  = channel_names[channel_index]
                skip_plot=False
                
                #skip_plot?
                if self.channel_names and (channel_name not in self.channel_names):
                    skip_plot=True
                        
                if not skip_plot:                   
                    f.add_subplot(nr_of_rows, nr_of_cols, channel_index + 1)
                
                    # actual plotting of the data. This can always be done
                    pylab.plot(tpoints, data[:, channel_index],
                               color=line_color)
                    channel_name  = channel_names[channel_index]
                    pylab.title(channel_name)
    
                    if self.physiological_arrangement:
                        x, y = ec_2d[channel_name]
                        w = .05
                        h = .045
                        pylab.gca().set_position([(x + 110) / 220, (y + 110) / 220, w, h])
                        
            pylab.draw()        
        
        
class SpectrumPlotNode(VisualizationBase):
    """ Construct spectrogram of the data using FFT
    
    This node uses the data for a power-spectral density (psd) computation
    plotted as time-frequency representation. The core function here is specgram 
    from matplotlib.mlab. The node inherits the
    functionality of the VisualisationBase.
    
    See documentation of :mod:`VisualisationBase <pySPACE.missions.nodes.visualization.base>` 
    to view basic functionality.
    
    If not specified differently using the parameters below, all channels
    are plotted in physiological arrangement separately for each class.
    
    .. note::
        If you average data, then the spectrogram is always computed on
        the average. Currently, the averaging of the power values is not
        implemented. 
        
    **Parameters**

      :channel_names:
        All classes are plotted directly in one figure, 
        if only one channel_name is specified (e.g., 'Pz').
        If more channels are specified, you get a plot of all channels
        for one class, i.e. for two classes two plots are returned.
        The default is that all channels are displayed which the user can
        explicitly specify using the keyword 'all'.
        
        (*optional, default: None*)
    
      :colorbar:
        Determine if the colorbar should be displayed.
        Currently, the colorbar is switched off when parameter 
        physiological_arrangement is True.
        
        (*optional, default: False*)
        
      :NFFT:
        The number of data points used in each block 
        for the FFT. Must be even; a power 2 is most efficient.
        
        (*optional, default: 128*)
      
      :noverlap:
        The number of points of overlap between blocks of the FFT.
        
        (*optional, default: 0*)
        
    **Exemplary Call**
    
    .. code-block:: yaml

        -
            node : Spectrum_Plot
            parameters :
                averaging : True
                online : True
                    
    :Author: Sirko Straube (sirko.straube@dfki.de)
    :Date of Last Revision: 2013/01/18                            
    """
    input_types = ["TimeSeries"]

    def __init__(self, 
                 channel_names=None,
                 colorbar = False, 
                 NFFT=128, 
                 noverlap=0,
                 **kwargs):
        super(SpectrumPlotNode, self).__init__(**kwargs)
        
        if channel_names=='all':
            channel_names=None
        elif type(channel_names) == str:
            channel_names=[channel_names]
        
        self.set_permanent_attributes(channel_names=channel_names, 
                                      colorbar=colorbar,
                                      NFFT=NFFT,
                                      noverlap=noverlap,
                                      min_value = pylab.inf,
                                      max_value = -pylab.inf)

    def _plotValues(self,
                    values,            #dict  TimeSeries values
                    plot_label,        #str   Plot-Label
                    fig_num,           #int   Figure-number for classify plots
                                       #      1: average,
                                       #      2: single trial,
                                       #      3: average accumulating
                    store_dir = None,  #str   Directory to store the plots
                    counter=0):        #int   Plotcounter for all trialsdef plot_values
                   
        sampling_frequency = values[values.keys()[0]].sampling_frequency
        list_of_classes = values.keys()
        num_of_classes = len(list_of_classes)
             
        #computing time points to show
        #num_tpoints = values.values()[0].shape[0]

        pylab.subplots_adjust(left  = 0.05,  # the left side of the subplots of the figure
                      right = 0.95,    # the right side of the subplots of the figure
                      bottom = 0.05,   # the bottom of the subplots of the figure
                      top = 0.95,      # the top of the subplots of the figure
                      wspace = 0.15,   # the amount of width reserved for blank space between subplots
                      hspace = 0.1,   # the amount of height reserved for white space between subplots                          
                      )

        #plot all channels separately in one figure
        #plot just 1 channel
        if self.channel_names != None and len(self.channel_names)==1:
            
            f=pylab.figure(fig_num, figsize = (18,13))

            #assure that figure is displayed with an interactive backend
            if pylab.get_backend() in pylab.matplotlib.backends.interactive_bk:
                f.show()
            
            for index, class_label in enumerate(list_of_classes):
                
                assert (self.channel_names[0] in values[class_label].channel_names),\
                    "SpectrumPlot::Channel requested for plotting is not available in data"
                       
                channel_index = values[class_label].channel_names.index(self.channel_names[0])  
                
                #operations like splicing only on view of object
                data = values[class_label].view(numpy.ndarray)                
                
                pylab.subplot(1, num_of_classes, index + 1)
                title = pylab.title(str(self.channel_names) + ' ' + class_label)
                
                self._plot_spectrum(data[:,channel_index], sampling_frequency)
                
            # Draw or store the figure
            if store_dir is None:
                pylab.draw()
            else:
                current_split=self.current_split
                if current_split != 0 and not\
                plot_label.endswith('_split_' + str(current_split)): #more than one split and first call
                    plot_label = plot_label + '_split_' + str(current_split)
                    
                f_name=str(store_dir) + str(os.sep) + str(plot_label)

                pylab.savefig(f_name + ".png")                                      

        #plot more than one channel in one figure for each label
        else:
            
            for index, class_label in enumerate(list_of_classes): 
                title = pylab.title(class_label)
                
                if self.channel_names==None: #this means all channels are plotted
                    self.channel_names=values[class_label].channel_names
                
                f=pylab.figure(fig_num, figsize = (18,13))
                
                #assure that figure is displayed with an interactive backend
                if pylab.get_backend() in pylab.matplotlib.backends.interactive_bk:
                    f.show()
                
                number_of_channels=len(self.channel_names)
                # Compute number of rows and cols for subplot-arrangement:
                # use 8 as upper limit for cols and compute rows accordingly
                if number_of_channels <= 8: 
                    nr_of_cols = number_of_channels
                else: 
                    nr_of_cols = 8
                    nr_of_rows = (number_of_channels - 1) / 8 + 1
                    
                data = values[class_label].view(numpy.ndarray)
                
                
                ec = self.getMetadata("electrode_coordinates")
                if ec is None:
                    ec = StreamDataset.ec
                
                ec_2d = StreamDataset.project2d(ec)
                
                # plot everything channel-wise
                for channel_index in range(number_of_channels):
                    channel_name  = self.channel_names[channel_index]

                    f.add_subplot(nr_of_rows, nr_of_cols, channel_index + 1)
                    
                    pylab.title(channel_name)
                    
                    # actual plotting of the data
                    self._plot_spectrum(data[:,channel_index], sampling_frequency)
                        
                    if self.physiological_arrangement:
                        x, y = ec_2d[channel_name]
                        w = .05
                        h = .045
                        pylab.gca().set_position([(x + 110) / 220, (y + 110) / 220, w, h])
                        
                pylab.draw()
                
                # Draw or store the figure
                if store_dir is None:
                    pylab.draw()
                else:
                    current_split=self.current_split
                    if current_split != 0 and not\
                    plot_label.endswith('_split_' + str(current_split)): #more than one split and first call
                        plot_label = plot_label + '_split_' + str(current_split)
                        
                    f_name=str(store_dir) + str(os.sep) + str(plot_label) + '_' + class_label
    
                    pylab.savefig(f_name + ".png") 
        
 #       title.set_fontsize(24)
 #       pylab.draw()      
        
    def _plot_spectrum(self, data, sampling_frequency):
        (Pxx, freqs, bins) = mlab.specgram(data,
                                           Fs=sampling_frequency,
                                           NFFT=self.NFFT,
                                           noverlap=self.noverlap)
                    
        if numpy.any(Pxx[0,0] == 0):
            self._log("SpectrumPlot::Instance has power 0 in a frequency band, skipping...")
            return

        # Update minimal and maximal value
        self.min_value = min(self.min_value, min(Pxx.flatten()))
        self.max_value = max(self.max_value, max(Pxx.flatten()))
        
        Z = numpy.flipud(Pxx)
        extent = 0, numpy.amax(bins), freqs[0], freqs[-1]
        
        pylab.imshow(Z, None, extent=extent, vmin=self.min_value,
                     vmax=self.max_value)
        pylab.axis('auto')
        pylab.xlabel("Time(s)")
        pylab.ylabel("Frequency(Hz)")
        
        if self.colorbar and not self.physiological_arrangement:
            pylab.colorbar()
        
        return (Pxx, freqs, bins)


class ScatterPlotNode(BaseNode):
    """Creates a scatter plot of the given channels for the given point in time
    
    This node creates scatter_plot of the values of all vs. all specified 
    channels for the given point in time (plot_ms).
    
    **Parameters**

        :plot_ms:  The point of time, for which the scatter plots are drawn.
                   For instance, if plot_ms = 200, all the values of the 
                   selected channels are collected that were measured
                   200ms after the window start and the scatter plots for 
                   these values are drawn

        :channels:  If channels is  not None, only scatter plots for
                    these specified channels are plotted. If channels is not
                    specified, scatter plots for the first 7 available 
                    channels are drawn. 
     
    .. note:: The maximal number of channels has to be less than 8 since
                more than a 7*7 matrix of plots is hard to get plotted
                into one window. 

    .. image:: ../../graphics/scatter_plot.png
        :width: 1024

    **Exemplary Call**

    .. code-block:: yaml

        -   node : ScatterPlot
            parameters :
                plot_ms : 2
    """
    figure_number = 0
    input_types = ["TimeSeries"]

    def __init__(self, plot_ms, channels=None, **kwargs):
        super(ScatterPlotNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(
            plot_ms=plot_ms,
            channels=channels,
            # An attribute that stores the number of channels
            number_of_channels = None, # Set lazily
            # A set of colors that can be used to distinguish different classes
            colors=set(["r", "b"]),
            # A mapping from class label to its color in the plot
            class_colors=dict())
        
        self.figure_number = ScatterPlotNode.figure_number 
        ScatterPlotNode.figure_number += 1      
        
        pylab.ion()
        figure = pylab.figure(self.figure_number,
                              figsize=(21, 11))
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

    def _train(self, data, label):
        """
        This node is not really trained but uses the labeled examples  to
        generate a scatter plot.
        """
        #Determine the number of channels if not yet done
        if self.number_of_channels == None:
            # If no channels are specified -> use the first seven
            # For more than 7*7 subplot, plotting is too slow
            if self.channels == None:
                self.channels = data.channel_names[:7]
            elif len(self.channels) > 7:
                self.channels = self.channels[:7]
            
            self.number_of_channels = len(self.channels)
            self.plot_index = data.shape[0] * self.plot_ms \
                                    / (data.end_time - data.start_time)
        
        # Determine color of this class if not yet done
        if label not in self.class_colors.keys():
            self.class_colors[label] = self.colors.pop() 
        
        pylab.ioff()
        pylab.figure(self.figure_number)
        print self.figure_number

        # For all pairs of channels
        #for ch1 in range(self.number_of_channels):
        #    for ch2 in range(self.number_of_channels):
        for index1, channel_name1 in enumerate(self.channels):
            for index2, channel_name2 in enumerate(self.channels):
                channel_index1 = data.channel_names.index(channel_name1)
                channel_index2 = data.channel_names.index(channel_name2)
                # Determine the relevant scatter plot
                pylab.subplot(self.number_of_channels,
                              self.number_of_channels,
                              index1 * self.number_of_channels + index2 + 1)
                
                pylab.text(0.1, 0.9, 
                           "%s vs. %s" % (channel_name1, channel_name2),
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform = pylab.gca().transAxes)
                
                if index1 == self.number_of_channels \
                    and index2 == self.number_of_channels:
                    pylab.ion()
                # Plot the point projected on the respective subspace
                pylab.plot([data[self.plot_index, channel_index1]],
                           [data[self.plot_index, channel_index2]],
                           self.class_colors[label] + "o")

        pylab.draw()
        
    def _stop_training(self, debug=False):
        pass
        
    def _execute(self, data):
        # We simply pass the given data on to the next node
        return data


class HistogramPlotNode(BaseNode):
    """ Creates a histogram of the given channels for the given point in time   
   
    This node creates histograms of the values of all specified channels for 
    the given point in time (plot_ms). The value range is restricted to the 
    specified value_range, values outside of this range are not plotted.
    
    **Parameters**

        :plot_ms:  The point of time, for which the histograms are drawn.
                   For instance, if plot_ms = 200, all the values of the 
                   selected channels are collected that were measured
                   200ms after the window start and the histograms for these
                   values are drawn

        :value_range:  A pair (tuple) that specifies the range of
                       values that are plotted in the histogram. Values
                       outside this range are not drawn.     

        :channels:  If channels is  not None, only histograms for
                    these specified channels are plotted. If channels is not
                    specified, histograms for all available channels
                    are plotted.

    .. image:: ../../graphics/histogram.png
        :width: 1024

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : HistogramPlot
            parameters :
                plot_ms : 2
                value_range : [-10, 10]

    """
    input_types = ["TimeSeries"]

    def __init__(self, plot_ms, value_range, channels=None, **kwargs):
        super(HistogramPlotNode, self).__init__(**kwargs)
                
        self.set_permanent_attributes(
            plot_ms=plot_ms,
            value_range=value_range,
            channels=channels,
            # An attribute that stores the number of channels
            number_of_channels=None, # Set lazily
            # A mapping from class  label to samples for this class
            class_samples=dict(),
            # A set of colors that can be used to distinguish different classes
            colors=set(["r", "b"]))
    
    def is_trainable(self):
        """ Returns whether this node is trainable. """
        # Though this node is not really trainable, it returns true in order
        # to get trained. The reason is that during this training phase, 
        # it visualizes all samples that are passed as arguments
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True

    def _train(self, data, label):
        """
        This node is not really trained but uses the labeled examples  to
        generate a histogram
        """
        #Determine the number of channels if not yet done
        if self.number_of_channels == None:
            # If no channels are specified -> plot all
            if self.channels == None:
                self.channels = data.channel_names
                
            self.number_of_channels = len(self.channels)
            self.plot_index = data.shape[0] * self.plot_ms \
                                    / (data.end_time - data.start_time)
            # Intialize Plotting            
            pylab.ion()
            figure = pylab.figure(figsize=(21, 11))
            figure.subplots_adjust(left=0.03, bottom=0.03,
                                   right=0.97, top= 0.97,
                                   wspace=0.2, hspace=0.2)
            
            # Set title of subplots
            pylab.ioff()
            for i, channel_name in enumerate(self.channels):
                pylab.subplot(int(math.ceil(float(self.number_of_channels)/2)),
                              2,
                              i + 1)
                pylab.text(0.5, 0.5, channel_name,
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform = pylab.gca().transAxes)
            pylab.ion()
            
        if label not in self.class_samples.keys():
            self.class_samples[label] = [[] for i in range(self.number_of_channels)]

        plot_column = data[self.plot_index, :]
        for index, channel_name in enumerate(self.channels):
            channel_index = data.channel_names.index(channel_name)
            self.class_samples[label][index].append(plot_column[channel_index])

    def _stop_training(self, debug=False):
        pylab.ioff()
        for class_label, class_samples in self.class_samples.iteritems():
            color = self.colors.pop()
            for i, channel_samples in enumerate(class_samples):
                pylab.subplot(int(math.ceil(float(self.number_of_channels )/2)),
                              2,
                              i+1)
                pylab.hist(channel_samples, bins = 25, fc=color, 
                           histtype = 'stepfilled', normed = True, alpha = 0.5,
                           range = self.value_range,
                           label = class_label)
        for i in range(self.number_of_channels):
            pylab.subplot(int(math.ceil(float(self.number_of_channels )/2)),
                          2,
                          i+1)      
            pylab.legend()
        pylab.ion()
        pylab.draw()
        
    def _execute(self, data):
        # We simply pass the given data on to the next node
        return data


_NODE_MAPPING = {"Time_Series_Plot": TimeSeriesPlotNode,
                 "Spectrum_Plot": SpectrumPlotNode,
                 "Scatter_Plot": ScatterPlotNode,
                 "Histogram_Plot": HistogramPlotNode}
