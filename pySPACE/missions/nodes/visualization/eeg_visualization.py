""" Visualization of :mod:`time series <pySPACE.resources.data_types.time_series>`  based on EEG signals to combine it with mapping to real sensor positions """
import logging
import os, pylab, numpy, warnings

from pySPACE.missions.nodes.visualization.base import VisualizationBase
from pySPACE.resources.dataset_defs.stream import StreamDataset
from matplotlib.mlab import griddata

class ElectrodeCoordinationPlotNode(VisualizationBase):
    """ Node for plotting EEG time series as topographies.
     
    This node uses time series data and plots snapshots of the activity in the
    brain, i.e. in the electrode configuration space. The node inherits the
    functionality of the VisualisationBase.
    
    Therefore, see documentation of :mod:`VisualisationBase <pySPACE.missions.nodes.visualization.base>` 
    to view basic functionality.
    
   **Parameters**
     
    :Layout Options:
    
        :contourlines:
            If set to True, contour lines are added to the plot.
        
            (*optional, default: False*)
        
        :nose_ears:
            Mark nose as triangle and ears as bars (common for EEG plots)
    
            (*optional, default: False*)
                        
        :smooth_corners:
            If true, the generated graphics will be rectangular, i.e., the
            corners around the round head shape are filled. This is achieved by
            adding sham electrodes to the corners, who's signals are calculated
            as the means of neighboring electrodes.
        
            .. note:: This is only for visualization purposes and does not
                reflect the true data! Be careful with use in scientific
                publications!
        
            (*optional, default: False*)
    
        :add_info:
            If set to True, additional information (e.g. number of trial) is
            displayed.
                
            (*optional, default: False*)
        
        :figlabels:
            If this option is True, channel names and channel coordinates.
        
            (*optional, default: False*)
        
        :single_plot:
            All results per window are plotted into one figure with columns
            being classes and rows being time points.
    
            (*optional, default: False*)
    
    :Influence the way how the data is plotted:
    
        :clip:
            If set to True, the values are clipped to maximum and minimum
            defined by parameter limits. This is only working if limits are
            defined. 

            (*optional, default: False*)
                
            
        :limits:
            Here, the user can set the limits for the color in the contour plot
            (e.g. [-1.0,1.0]). If this option is not set, the colorbar is
            normalized according to the data.

            (*optional, default: False*)
    
    **Exemplary Call**
    
    .. code-block:: yaml

        - 
            node : Time_Series_Source
        -
             node : Electrode_Coordination_Plot
             parameters :
                  figlabels : True
                  create_movie : True
                  time_stamps : [200, 400]
                  timeshift : -200
                  smooth_corners : False
        - 
             node: Nil_Sink
             
    Here is an alternative call:
    
    .. code-block:: yaml

        -  
            node : Electrode_Coordination_Plot
            parameters :
                 single_trial : True
                 accum_avg : True
                 separate_training_and_test : True
                 add_info : True
                 time_stamps : [400]
                 limit2class : Target
        
    :Author: Sirko Straube (sirko.straube@dfki.de)
    :Date of Last Revision: 2013/01/01
    
    .. todo:: Depending on which plot backend is used, the resizing of the
                figure does not work well. Currently this is not well supported
                by matplotlib. Change when support is improved. The key command
                is fig.set_size_inches([a,b]) and the dpi property.
    """
    input_types = ["TimeSeries"]

    def __init__(self,
                 clip = False,
                 contourlines = False,
                 limits = False,
                 nose_ears = False,
                 smooth_corners = False,
                 add_info=False,
                 single_plot=False,
                 figlabels=False,
                 **kwargs):
        
        super(ElectrodeCoordinationPlotNode, self).__init__(**kwargs)
        
        if limits:
            limits = [float(i) for i in limits] #make sure limits consists of floats
            
        # define electrode grid.
        xi = numpy.linspace(-125, 125, 200)
        yi = numpy.linspace(-100, 100, 200)

        self.set_permanent_attributes(xi=xi,
                                      yi=yi,
                                      clip = clip,
                                      contourlines = contourlines,
                                      limits = limits,
                                      nose_ears = nose_ears,
                                      smooth_corners = smooth_corners,
                                      add_info = add_info,
                                      single_plot = single_plot,
                                      figlabels = figlabels,
                                      time_checked = False)
 
    def _plotValues(self,
                    values,            #dict  TimeSeries values
                    plot_label,        #str   Plot-Label
                    fig_num,           #int   Figure-number for classify plots
                                       #      1: average,
                                       #      2: single trial,
                                       #      3: average accumulating
                    store_dir = None,  #str   Directory to store the plots
                    counter=0):        #int   Plotcounter for all trials
        #compute sampling_frequency and classes to plot
        sampling_frequency = values[values.keys()[0]].sampling_frequency
        list_of_classes = values.keys()
        num_of_classes = len(list_of_classes)
        #autoscale color bar or use user scaling
        if self.limits:
            levels = self._compute_levels()
        else:
            levels = None
            #compute maximum and minimum for colorbar scaling if not existing
            vmax = float(max(numpy.array(v).max() for v in values.itervalues()))
            vmin = float(min(numpy.array(v).min() for v in values.itervalues()))
            levels = self._compute_levels(limits=[vmin, vmax])
            #normalizer=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        
        #computing time points to show
        num_tpoints = values.values()[0].shape[0]
        all_tpoints = numpy.arange(0,
                                   num_tpoints * (1000/sampling_frequency),
                                   1000 / sampling_frequency
                                   ) + self.timeshift
        
        if self.time_stamps == [-1]:
            tpoints = all_tpoints
        else: #check if desired time points are existing and confirm
            if not self.time_checked:
                for t in self.time_stamps:
                    if not t in all_tpoints:
                        warnings.warn("Electrode_Coordination_Plot:: At least" \
                                      " one desired time stamp not available!" \
                                      " Legal time stamps are " \
                                      + str(all_tpoints) + ". Switching to " \
                                      "next legal time point. Please check " \
                                      "for consistency!")
                        
                        if t < 0:
                            new_t = self.timeshift
                        else:
                            new_t = range(0, t+1, int(1000/sampling_frequency))[-1]
                        
                            #if we obtain an empty list reset to timeshift
                            if new_t == []:
                                new_t = self.timeshift
                            else:
                                new_t = new_t+self.timeshift
                        
                        #finally check for too high or low values
                        if new_t < self.timeshift:
                            new_t = self.timeshift
                        elif new_t > all_tpoints[-1]:
                            new_t = all_tpoints[-1]
                
                        self.time_stamps[self.time_stamps.index(t)] = new_t
                self.time_checked = True #has to be performed only once
                
            tpoints = numpy.array(self.time_stamps)
        
        num_of_tpoints = len(tpoints)
        
        # selecting formatting and clearing figure
        default_size = [8., 6.]
        if self.single_plot:
            num_of_rows = num_of_tpoints
            if num_of_rows > 4:
                default_size[0] = default_size[0]/(int((num_of_rows+3)/4))
                default_size[1] = default_size[1]/(int((num_of_rows+3)/4))
        else:
            num_of_rows = 1

        f=pylab.figure(fig_num, figsize=[default_size[0]*num_of_classes, default_size[1]*num_of_rows])
        if pylab.get_backend() in pylab.matplotlib.backends.interactive_bk:
            f.show()

        if counter%20 == 19: #clear every 20th trial
            pylab.figure(fig_num).clear()
        
        # Iterate  over the time window
        for time_index in range(num_of_tpoints):
            
            pylab.subplots_adjust(left=0.025, right=0.8) #shift a bit to the left
            
            if self.single_plot:
                pl_offset=time_index
            else:
                pl_offset=0

            for index, class_label in enumerate(list_of_classes):
                current_plot_num=(num_of_classes*pl_offset)+index+1
                pylab.subplot(num_of_rows, num_of_classes, current_plot_num)
                pylab.gca().clear()
        
                # Get the values for the respective class
                data = values[class_label].view(numpy.ndarray)

                ec = self.get_metadata("electrode_coordinates")
                if ec is None:
                    ec = StreamDataset.ec
                    
                # observe channels
                channel_names = [channel for channel in values[class_label].channel_names if channel in ec.keys()]
                fcn = [channel for channel in values[class_label].channel_names if not channel in ec.keys()]
                if not fcn == []:
                    self._log("Unsupported channels ignored:%s ."%str(fcn),
                                level=logging.CRITICAL)
                if channel_names == []:
                    self._log("No channel for plotting left.",
                              level=logging.CRITICAL)
                    return

                ec_2d = StreamDataset.project2d(ec)
                
                # Define x and y coordinates of electrodes in the order of the channels
                # of data
                x = numpy.array([ec_2d[key][0] for key in channel_names])
                y = numpy.array([ec_2d[key][1] for key in channel_names])
            
#                x = numpy.array([self.electrode_coordinates[key][0] * numpy.cos(self.electrode_coordinates[key][1]/180*numpy.pi)
#                                        for key in channel_names])
#                y = numpy.array([self.electrode_coordinates[key][0] * numpy.sin(self.electrode_coordinates[key][1]/180*numpy.pi) 
#                                        for key in channel_names])
                
                # The values of the electrodes at this point of time 
                pos=list(all_tpoints).index(tpoints[time_index])
                
                z = data[pos, :]
        
                if self.smooth_corners:
                    x,y,z = self._smooth_corners(x,y,z, data, channel_names, pos)
                
                #griddata returns a masked array
                #you can get the data via zi[~zi.mask]            
                zi = griddata(x, y, z, self.xi, self.yi)
                
                #clip values
                if self.clip and self.limits:
                    zi=numpy.clip(zi, self.limits[0], self.limits[1]) #minimum and maximum
                    
                # contour the gridded data,
                # plotting dots at the nonuniform data points.
                
                cs=pylab.contourf(self.xi, self.yi, zi, 15, cmap=pylab.cm.jet, levels=levels)
                if self.contourlines:
                    pylab.contour(self.xi, self.yi, zi, 15, linewidths=0.5, colors='k', levels=levels)
                
                if self.figlabels:
                    # plot data points.
                    if not self.smooth_corners:
                        pylab.scatter(x, y, c='b', s=5, marker='o')
                    else:
                        # dont plot invented electrode positions
                        pylab.scatter(x[:-4], y[:-4], c='b', s=5, marker='o')
                    # Add channel labels
                    for label, position in ec_2d.iteritems():
                        if label in channel_names:
                            pylab.text(position[0], position[1], label)
                        
                if self.add_info:
                    if counter:
                        if len(list_of_classes) > 1:
                            if index == 0:
                                pylab.text(-120, -98, 'Trial No. ' + str(counter), fontsize=12)
                        else:
                            pylab.text(-120, -98, 'Trial No. ' + str(counter), fontsize=12)
                
                if self.nose_ears:
                    #nose
                    ytips=[87.00,87.00, 97]
                    xtips=[-10.00,10.00, 0]
                    pylab.fill(xtips,ytips, facecolor='k', edgecolor='none')
                    
                    #left
                    xtips=[-108.0,-113.0,-113.0,-108.0]
                    ytips=[-10.0,-10.0,10.0,10.0]
                    pylab.fill(xtips,ytips, facecolor='k', edgecolor='none')

                    #right
                    xtips=[108.0,114.0,113.0,108.0]
                    ytips=[-10.0,-10.0,10.0,10.0]
                    pylab.fill(xtips,ytips, facecolor='k', edgecolor='none')
                
                pylab.xlim(-125, 125)
                pylab.ylim(-100, 100)
                if not self.single_plot or time_index==0: #if single_plot=True do only for the first row
                    pylab.title(class_label, fontsize=20)
                pylab.setp(pylab.gca(), xticks=[], yticks=[])
                pylab.draw()
            
            caxis = pylab.axes([.85, .1, .04, .75])
            cb = pylab.colorbar(mappable=cs, cax=caxis)
            # TODO: The label read 'Amplitude ($\mu$V)'
            #       Removed the unit. Or can we really still assume 
            #       a (correct) \muV scale after all preprocessing?
            cb.ax.set_ylabel(r'Amplitude', fontsize=16)
    
            # Position of the time axes
            ax = pylab.axes([.79, .94, .18, .04])
            pylab.gca().clear()
            
            pylab.bar(tpoints[time_index], 1.0,  width=1000.0/sampling_frequency)
            pylab.xlim(tpoints[0], tpoints[-1])
            pylab.xlabel("time (ms)", fontsize=12)
            pylab.setp(ax, yticks=[],xticks=[all_tpoints[0], tpoints[time_index], all_tpoints[-1]])
            
            # Draw or store the figure
            if store_dir is None:
                pylab.draw()
                #pylab.show()
            elif self.single_plot and not current_plot_num==(num_of_rows*num_of_classes): #save only if everything is plotted
                pylab.draw()
                #pylab.show()
            else:
                current_split=self.current_split
                if current_split != 0 and not\
                plot_label.endswith('_split_' + str(current_split)): #more than one split and first call
                    plot_label = plot_label + '_split_' + str(current_split)
                    
                f_name=str(store_dir) + str(os.sep) + str(plot_label) + "_" + str(int(tpoints[time_index]))
                pylab.savefig(f_name + ".png")
            
        if self.store_data:
            import pickle
            f_name=str(store_dir) + str(os.sep) + str(plot_label)
            pickle.dump(values, open(f_name + ".pickle",'w'))

        
    def _smooth_corners(self, x, y, z, data, channel_names, time_index):
        """ Add sham electrodes to the corners of the coordinate system """
        # invent new corner electrodes using x and y positions of the margin
        # electrodes of the 64 electrode cap. Data is mean of neighbouring
        # electrodes based, again, on the 64 electrode cap.
        # 
        # frontleft FL positioned at [x(FT9), y(Fp1)]
        
        x=numpy.append(x,x[numpy.where(numpy.array(channel_names)=='FT9')])
        y=numpy.append(y,y[numpy.where(numpy.array(channel_names)=='Fp1')])
        nz =  data[time_index, numpy.where(numpy.array(channel_names)=='Fp1')] +\
              data[time_index, numpy.where(numpy.array(channel_names)=='AF7')] +\
              data[time_index, numpy.where(numpy.array(channel_names)=='F7')] +\
              data[time_index, numpy.where(numpy.array(channel_names)=='FT9')]
        z=numpy.append(z, 0.1 * nz / 4.0)
        # frontright FR positioned at [x(FT10), y(Fp1)]
        x=numpy.append(x,x[numpy.where(numpy.array(channel_names)=='FT10')])
        y=numpy.append(y,y[numpy.where(numpy.array(channel_names)=='Fp1')])
        nz =  data[time_index, numpy.where(numpy.array(channel_names)=='Fp2')] +\
              data[time_index, numpy.where(numpy.array(channel_names)=='AF8')] +\
              data[time_index, numpy.where(numpy.array(channel_names)=='F8')] +\
              data[time_index, numpy.where(numpy.array(channel_names)=='FT10')]
        z=numpy.append(z, 0.1 * nz / 4.0)
        # backleft BL positioned at [x(FT9), y(Oz)]
        x=numpy.append(x,x[numpy.where(numpy.array(channel_names)=='FT9')])
        y=numpy.append(y,y[numpy.where(numpy.array(channel_names)=='Oz')])
        nz =  data[time_index, numpy.where(numpy.array(channel_names)=='TP9')] +\
              data[time_index, numpy.where(numpy.array(channel_names)=='P7')] +\
              data[time_index, numpy.where(numpy.array(channel_names)=='PO9')]
        z=numpy.append(z, 0.1 * nz / 4.0)
        # backright BR positioned at [x(FT10), y(Oz)]
        x=numpy.append(x,x[numpy.where(numpy.array(channel_names)=='FT10')])
        y=numpy.append(y,y[numpy.where(numpy.array(channel_names)=='Oz')])
        nz =  data[time_index, numpy.where(numpy.array(channel_names)=='TP10')] +\
              data[time_index, numpy.where(numpy.array(channel_names)=='P8')] +\
              data[time_index, numpy.where(numpy.array(channel_names)=='PO10')]
        z=numpy.append(z, 0.1 * nz / 4.0)
        
        return x,y,z
        
    def _compute_levels(self, limits=None):
        if not limits:
            if self.limits:
                limits=self.limits
            else: #should never be reached
                return None
                
        rel_precision = int(('%1.e'%limits[0])[-3:]) #determine the precision of the values
        epsilon=pow(10, rel_precision-5) #add a small amount to make sure that clipping works
        step = (limits[1]-limits[0])/100 #split into 100 steps
        levels=numpy.arange(limits[0],limits[1]+epsilon,step)
        levels=numpy.round(levels, decimals=abs(rel_precision)+2)
        levels[0]=levels[0]-epsilon
        levels[-1]=levels[-1]+epsilon
        return levels
        

_NODE_MAPPING = {"Electrode_Coordination_Plot": ElectrodeCoordinationPlotNode}
