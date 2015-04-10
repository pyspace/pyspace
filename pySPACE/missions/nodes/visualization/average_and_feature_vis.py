""" Visualize average of :mod:`time series <pySPACE.resources.data_types.time_series>` and time domain features

Additional features can be added to the visualization.

"""
import os
import cPickle
import pylab
import matplotlib.font_manager
from matplotlib import colors

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.resources.dataset_defs.stream import StreamDataset
from pySPACE.tools.filesystem import create_directory


def convert_feature_vector_to_time_series(feature_vector, sample_data):
    """ Parse the feature name and reconstruct a time series object holding the equivalent data 
    
    In a feature vector object, a feature is determined by the feature
    name and the feature value. When dealing with time domain features, the
    feature name is a concatenation of the (pseudo-) channel
    name and the time within an epoch in seconds. A typical feature name
    reads, e.g., "TD_F7_0.960sec".
    """
        
    # channel name is what comes after the first underscore
    feat_channel_names = [chnames.split('_')[1]
                          for chnames in
                          feature_vector.feature_names]
    # time is what comes after the second underscore
    feat_times = [int(float((chnames.split('_')[2])[:-3])
                      * sample_data.sampling_frequency)
                     for chnames in feature_vector.feature_names]
    
    # generate new time series object based on the exemplary "sample_data"
    # all filled with zeros instead of data
    new_data = TimeSeries(pylab.zeros(sample_data.shape),
                          channel_names=sample_data.channel_names,
                          sampling_frequency=sample_data.sampling_frequency,
                          start_time=sample_data.start_time,
                          end_time=sample_data.end_time,
                          name=sample_data.name,
                          marker_name=sample_data.marker_name)
                          
    # try to find the correct place (channel name and time)
    # to insert the feature values
    for i in range(len(feature_vector)):
        try:
            new_data[feat_times[i],
                     new_data.channel_names.index(feat_channel_names[i])] = \
                feature_vector[i]
        except ValueError:
            import warnings
            warnings.warn("\n\nFeatureVis can't find equivalent to Feature "+
                          feature_vector.feature_names[i] + 
                          " in the time series.\n")
    
    return new_data


class AverageFeatureVisNode(BaseNode):
    """ Visualize time domain features in the context of average time series.
    
    This node is supposed to visualize features from any feature
    selection algorithm in the context of the train. This data is some kind of
    time series, either channelwise "plain" EEG time series or somehow
    preprocessed data, e.g. the time series of CSP pseudo channels.
        
    The purpose is to investigate two main issues:

    1. By comparing the mean time series of standard and target time windows,
       is it understandable why certain features have been selected?
    2. Comparing the time series from one set to the selected features from
       some other set, are the main features robust? 
    
    If no features are passed to this node, it will still visualize average
    time series in any case. Only the time series that are labeled as training
    data will be taken into account. The reason is that the primary aim of
    this node is to visualize the features on the very data they were chosen
    from, i.e., the training data. If instead all data is to be plotted (e.g.,
    at the end of a preprocessing flow), one would in the worst case have to
    run the node chain twice. In the extra run for the visualization, an
    All_Train_Splitter would be used prior to this node.
            
    This is what this node will plot:

    - In a position in the node chain where the current data object is a time
      series, it will plot the average of all training samples if the
      current time series.
    - If the current object is not a time series, this node will go back in
      the data object's history until it finds a time series. This time series
      will then be used.
    - If a path to a features.pickle is passed using the load_feature_path
      variable, then this features will be used for plotting.
    - If no load_feature_path is set, this node will check if the current data
      object has a data.predictor.features entry. This will be the case if the
      previous node has been a classifier. If so, these features will be used.
    - If features are found in neither of the aforementioned two locations, no
      features will be plotted. The average time series however will still be
      plotted.
    
    
    **Parameters**
    
        :load_feature_path:
            Path to the stored pickle file containing the selected features.
            So far, LibSVM and 1-Norm SVM nodes can deliver this output.
            Defaults to 'None', which implies that no features are plotted.
            The average time series are plotted anyway.
            
            (*optional, default: None*)
    
        :error_type:
            Selects which type of error is into the average time series plots:
            
                :None: No errors
                :'SampleStdDev': +- 1 Sample standard deviation.
                    This is, under Gaussian assumptions, the area, in which 68% of the
                    samples lie.
                    
                :'StdError': +- 1 Standard error of the mean.
                    This is the area in which, under Gaussian assumptions, the sample
                    mean will end up in 68% of all cases.
            
            If multiples of this quantities are desired, simply use them as prefix
            in the strings.
            With Multiplier 2, the above percentages change to 95%
            With Multiplier 3, the above percentages change to 99.7%
            
            Here are examples for valid entries:
            '2SampleStdDev', None, 'StdError', '2StdError', '1.7StdError'
            
            (*optional, default: '2StdError'*)
    
    
        :axflip:
            If axflip is True, the y-axes of the averaged time series plots
            are reversed. This makes the plots look the way to which psychologists
            (and even some neuro scientists) are used.
            
            (*optional, default: False*)
    
        :alternative_scaling:
            If False, the values from the loaded feature file (i.e. the "w" in the
            SVM notation) are directly used for both graphical feature
            representation and rating of "feature importance". If True, instead
            the product of these values and the difference of the averaged time
            domain feature values of both classes is used: importance(i) = w(i) *
            (avg_target(i) - avg_standard(i)) On the one hand, using the feature
            averages implicitly assumes normally distributed features. On the
            other hand, this computation takes into account the fact that
            different features have different value ranges. The eventual
            classification with SVMs is done by evaluating the
            sum_i{  w(i) * feature(i) }.
            In that sense, the here defined importance measures the average
            contribution of a certain feature to the classification function.
            As such, and that's the essential point, it makes the values
            comparable.
            
            (*optional, default: False*)
    
        :physiological_arrangement:
            If False all time series plots are arranged in a matrix of plots. If
            set to True, the plots are arranged according to the arrangement of
            the electrodes on the scalp. Obviously, this only makes sense if the
            investigated time series are not spatially filtered. CSP pseudo
            channels, e.g., can't be arranged on the scalp.
            
            (*optional, default: False*)
    
        :shrink_plots:
            Defaults to False and is supposed to be set to True, whenever channels
            from the 64 electrode cap are investigated jointly with electrodes
            from 128 cap that do not appear on the 64 cap. Omits overlapping of
            the plots in physiological arrangement.
            
            (*optional, default: False*)
    
        :important_feature_thresh:
            Gives a threshold below which features are not considered important.
            Only important features will appear in the plots.
            Defaults to 0, i.e. all non-zero features are important.
            This parameter collides with  percentage_of_features; the stricter
            restriction applies.
            
            (*optional, default: 0.0*)
            
        :percentage_of_features:
            Define the percentage of features to be drawn in the plots.
            Defaults to 100, i.e. all features are to be used.
            This parameter collides with  important_feature_thresh; the stricter
            restriction applies. Thus, even in the default case, most of the time
            less than 100% of the features will be drawn due to the non-zero
            condition of the important_feature_thresh parameter.
            Note that the given percentage is in relation to the total number of
            features; not in relation to the number of features a classifier has
            used in some sense.
            
            (*optional, default: 100*)
            
    
        :emotiv:
            Use the emotiv parameter if the data was acquired wit the emotiv EPOC
            system. This will just change the position of text in the plots - it's
            not visible otherwise.
            
            (*optional, default: False*)

    **Known Issues**
    
    The title of physiologically arranged time series plots vanishes, if no
    frontal channels are plotted, because the the plot gets trimmed and so
    gets the title.
    
    **Exemplary Call**

    .. code-block:: yaml

        - 
            node : AverageFeatureVis
            parameters : 
                load_feature_path : "/path/to/my/features.pickle"
                alternative_scaling : True
                physiological_arrangement : True
                axflip : True
                shrink_plots : False
                important_feature_thresh : 0.3
                percentage_of_features : 20
                error_type : "2SampleStdDev"
                
            
    
    :Author: David Feess (David.Feess@dfki.de)
    :Created: 2010/02/10
    :Reviewed: 2011/06/24
    """
    input_types = ["TimeSeries", "PredictionVector"]

    def __init__(self,
                 load_feature_path='None',
                 axflip= False,
                 alternative_scaling=False,
                 physiological_arrangement=False,
                 shrink_plots=False,
                 important_feature_thresh=0.0,
                 percentage_of_features=100,
                 emotiv=False,
                 error_type='2StdError',
                 **kwargs):
                 
        # Must be set before constructor of superclass is set
        self.trainable = True
        
        super(AverageFeatureVisNode, self).__init__(store=True, **kwargs)
        
        # try to read the file containing the feature information
        feature_vector = None
        try:
            feature_file = open(load_feature_path, 'r')
            feature_vector = cPickle.load(feature_file)
            feature_vector = feature_vector[0] #[0]!!
            feature_file.close()
        except:
            print "FeatureVis: No feature file to load."+\
                  " Will search in current data object."
                          
        self.set_permanent_attributes(
            alternative_scaling=alternative_scaling,
            physiological_arrangement=physiological_arrangement,
            feature_vector=feature_vector,
            important_feature_thresh=important_feature_thresh,
            percentage_of_features=percentage_of_features,
            shrink_plots=shrink_plots,
            max_feature_val=None,            
            feature_time_series=None,
            number_of_channels=None,
            samples_per_window=None,
            sample_data=None,
            own_colormap=None,
            error_type=error_type,
            mean_time_series=dict(),
            time_series_histo=dict(),
            error=dict(),
            mean_classification_target=None,
            samples_per_condition=dict(),
            normalizer=None,
            trainable=self.trainable,
            corr_important_feats=dict(),
            corr_important_feat_names=None,
            labeled_corr_matrix=dict(),
            channel_names=None,
            indexlist=None,
            ts_plot=None,
            histo_plot=None,
            corr_plot=dict(),
            feature_development_plot=dict(),
            axflip=axflip,
            emotiv=emotiv
        )


    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return self.trainable
    

    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return self.trainable

    def _execute(self, data):
        """ Nothing to be done here """
        return data
    

    def get_last_timeseries_from_history(self, data):
        n = len(data.history)
        for i in range(n):
            if type(data.history[n-1-i]) == TimeSeries:
                return data.history[n-1-i]
        raise LookupError('FeatureVis found no TimeSeries object to plot.' +
            ' Add "keep_in_history : True" to the last node that produces' +
            ' time series objects in your node chain!')

    def _train(self, data, label):
        """
        Add the given data point along with its class label 
        to the training set, i.e. update 'mean' time series and append to
        the complete data.
        """
        # Check i
        # This is needed only once
        if self.feature_vector == None and self.number_of_channels == None:
            # if we have a prediction vector we are at a processing stage
            # after a classifier. The used features can than be found in
            # data.predictor.features.
            try: 
                self.feature_vector = data.predictor.features[0] #[0]!!
                print "FeatureVis: Found Features in current data object."
            # If we find no data.predictor.features, simply go on without
            except:
                print "FeatureVis: Found no Features at all."


        # If the current object is no time series, get the last time series
        # from history. This will raise an exception if there is none.
        if type(data) != TimeSeries:
            data = self.get_last_timeseries_from_history(data)
        
        # If this is the first data sample we obtain
        if self.number_of_channels == None:
            # Count the number of channels & samples per window
            self.number_of_channels = data.shape[1]
            self.sample_data = data
            self.samples_per_window = data.shape[0]
            self.channel_names = data.channel_names
            
        # If we encounter this label for the first time
        if label not in self.mean_time_series.keys():
            # Create the class mean time series lazily
            self.mean_time_series[label] = data
            self.time_series_histo[label] = []
        else:
            # If label exists, just add data 
            self.mean_time_series[label] += data

        self.time_series_histo[label].append(data)
        
        # Count the number of samples per class
        self.samples_per_condition[label] = \
                     self.samples_per_condition.get(label, 0) + 1
    

    def _stop_training(self, debug=False):
        """
        Finish the training, i.e. for the time series plots: take the
        accumulated time series and divide by the number of samples per
        condition.
        For the
        """
        # Compute avg
        for label in self.mean_time_series.keys():
            self.mean_time_series[label] /= self.samples_per_condition[label]
            self.time_series_histo[label] = \
                pylab.array(self.time_series_histo[label])
 
            # Compute error of desired type - strip the numerals:
            if self.error_type is not None:
                if self.error_type.strip('0123456789.') == 'SampleStdDev':
                    self.error[label] = \
                     pylab.sqrt(pylab.var(self.time_series_histo[label],0))
                elif self.error_type.strip('0123456789.') == 'StdError':
                    self.error[label] = \
                     pylab.sqrt(pylab.var(self.time_series_histo[label],0)) /\
                     pylab.sqrt(pylab.shape(self.time_series_histo[label])[0])
            
                multiplier = float(''.join([nr for nr in self.error_type 
                                            if (nr.isdigit() or nr == ".")]))
                self.error[label] = multiplier * self.error[label]
            
        # other plots only if features where passed
        if (self.feature_vector != None):
            self.feature_time_series = \
                convert_feature_vector_to_time_series(self.feature_vector,
                                                      self.sample_data)
            
            # in the alternative scaling space, the feature "importance" is
            # determined by the feature values
            # weighted by the expected difference in time series values 
            # between the two classes (difference of avg std and avg target)
            # The standard P3 and LRP cases are handeled separately to make
            # sure that the sign of the difference is consistent
            if self.alternative_scaling:
                if all(
                    [True if label_iter in ['Target', 'Standard'] else False
                               for label_iter in self.mean_time_series.keys()]):
                    self.feature_time_series*=(
                        self.mean_time_series['Target']-
                                            self.mean_time_series['Standard'])
                elif all(
                    [True if label_iter in ['LRP', 'NoLRP'] else False
                               for label_iter in self.mean_time_series.keys()]):
                    self.feature_time_series*=(
                        self.mean_time_series['LRP']-
                                            self.mean_time_series['NoLRP'])
                else:
                    self.feature_time_series*=(
                       self.mean_time_series[self.mean_time_series.keys()[0]]-
                       self.mean_time_series[self.mean_time_series.keys()[1]])
                    print "AverageFeatureVis (alternative_scaling): " +\
                      "Present classes don't match the standards " +\
                      "(Standard/Target or LRP/NoLRP). Used the difference "+\
                      "%s - %s" % (self.mean_time_series.keys()[0],
                       self.mean_time_series.keys()[1]) +" for computation "+\
                       "of the alternative scaling."

            
            # greatest feature val that occures is used for the normalization
            # of the color-representation of the feature values
            self.max_feature_val = \
                (abs(self.feature_time_series)).max(0).max(0)
            self.normalizer = colors.Normalize(vmin=-self.max_feature_val,
                                               vmax= self.max_feature_val)            
            cdict={  'red':[(0.0, 1.0, 1.0),(0.5, 1.0, 1.0),(1.0, 0.0, 0.0)],
                   'green':[(0.0, 0.0, 0.0),(0.5, 1.0, 1.0),(1.0, 0.0, 0.0)],
                    'blue':[(0.0, 0.0, 0.0),(0.5, 1.0, 1.0),(1.0, 1.0, 1.0)]}
            self.own_colormap = \
                colors.LinearSegmentedColormap('owncm', cdict, N=256)
            
            # sort the features with descending importance
            self.indexlist=pylab.transpose(self.feature_time_series.nonzero())
            indexorder = abs(self.feature_time_series
                             [abs(self.feature_time_series) > 
                                    self.important_feature_thresh]).argsort()

            self.indexlist = self.indexlist[indexorder[-1::-1]] #reverse order
            self.indexlist = map(list,self.indexlist[
                :len(self.feature_vector)*self.percentage_of_features/100])
            
            self.histo_plot = self._generate_histo_plot()
            
            try:
                # try to generate a plot of the feature crosscorrelation
                # matrix. Might fail if the threshold is set such that no
                # features are left.
                for label in self.mean_time_series.keys():
                    self.labeled_corr_matrix[label] = \
                        self._generate_labeled_correlation_matrix(label)
                    self.corr_plot[label] = \
                        self._get_corr_plot(self.corr_important_feats[label],
                                            label)

                # if 2 class labels exist, also compute the difference in the
                # cross correlation between the classes.
                if len(self.corr_important_feats.keys()) == 2:
                    self.corr_plot['Diff'] = self._get_corr_plot((
                        self.corr_important_feats
                            [self.corr_important_feats.keys()[0]]
                      - self.corr_important_feats
                            [self.corr_important_feats.keys()[1]]),
                        self.corr_important_feats.keys()[0] + ' - ' + \
                            self.corr_important_feats.keys()[1])
            except TypeError:
                import warnings
                warnings.warn("\n\nFeatureVis doesn't have enough important" +
                              " features left for correlation plots..."+
                              " Check threshold.\n")
                              
        # Compute avg time series plot anyway         
        self.ts_plot = self._generate_time_series_plot()
    

    def store_state(self, result_dir, index=None):
        """ Stores all generated plots in the given directory *result_dir* """
        if self.store:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            if not index == None:
                node_dir += "_%d" % int(index)

            create_directory(node_dir)
            
            if (self.ts_plot != None):
                name = 'timeseries_sp%s.pdf' % self.current_split
                self.ts_plot.savefig(os.path.join(node_dir, name),
                                     bbox_inches="tight")
            
            if (self.histo_plot != None):
                name = 'histo_sp%s.pdf' % self.current_split
                self.histo_plot.savefig(os.path.join(node_dir, name),
                                        bbox_inches="tight")
            
            for label in self.labeled_corr_matrix.keys():
                name = 'Feature_Correlation_%s_sp%s.txt' % (label,
                                                           self.current_split)
                pylab.savetxt(os.path.join(node_dir, name),
                              self.labeled_corr_matrix[label], fmt='%s',
                              delimiter='  ')
                name = 'Feature_Development_%s_sp%s.pdf' % (label,
                                                           self.current_split)
                self.feature_development_plot[label].savefig(
                    os.path.join(node_dir, name))
            
            for label in self.corr_plot.keys():
                name = 'Feature_Correlation_%s_sp%s.pdf' % (label,
                                                            self.current_split)
                self.corr_plot[label].savefig(os.path.join(node_dir, name))
            
            pylab.close("all")
    
    
    def _generate_labeled_correlation_matrix(self, label):
        """ Concatenates the feature names to the actual correlation matrices.
            This is for better overview in stored txt files later on."""
        labeled_corr_matrix = pylab.array([])
        for i in pylab.array(self.corr_important_feats[label]):
            if len(labeled_corr_matrix) == 0:
                labeled_corr_matrix = [[('% .2f' % j).rjust(10) for j in i]]
            else:
                labeled_corr_matrix = pylab.vstack((labeled_corr_matrix,
                                    [[('% .2f' % j).rjust(10) for j in i]]))
        
        labeled_corr_matrix = pylab.c_[self.corr_important_feat_names,
                                       labeled_corr_matrix]
        labeled_corr_matrix = pylab.vstack((pylab.hstack(('          ',
                                           self.corr_important_feat_names)),
                                           labeled_corr_matrix))
        
        return labeled_corr_matrix
    

    def _generate_time_series_plot(self):
        """ This function generates the actual time series plot"""
        # This string will show up as text in the plot and looks something
        # like "Target: 123; Standard:634"
        samples_per_condition_string = \
            ";  ".join([("%s: " + str(self.samples_per_condition[label]))
                       % label for label in self.mean_time_series.keys()])
        
        figTS = pylab.figure()
        
        # Compute number of rows and cols for subplot-arrangement:
        # use 8 as upper limit for cols and compute rows accordingly
        if self.number_of_channels <= 8: nr_of_cols = self.number_of_channels
        else: nr_of_cols = 8
        nr_of_rows = (self.number_of_channels - 1) / 8 + 1
        
        # Set canvas size in inches. These values turned out fine, depending
        # on [physiological_arrengement] and [shrink_plots]
        if not self.physiological_arrangement:
            figTS.set_size_inches((5 * nr_of_cols,  3 * nr_of_rows))
            ec_2d = None
        else:
            if not self.shrink_plots:
                figTS.set_size_inches((3*11.7,  3*8.3))
            else:
                figTS.set_size_inches((4*11.7,  4*8.3))
                
            ec = self.get_metadata("electrode_coordinates")
            if ec is None:
                ec = StreamDataset.ec
            ec_2d = StreamDataset.project2d(ec)
                
        # plot everything channel-wise
        for i_chan in range(self.number_of_channels):
            figTS.add_subplot(nr_of_rows, nr_of_cols, i_chan + 1)
            
            # actual plotting of the data. This can always be done
            for tslabel in self.mean_time_series.keys():
                tmp_plot=pylab.plot(self.mean_time_series[tslabel][:, i_chan],
                           label=tslabel)
                cur_color = tmp_plot[0].get_color()
                if self.error_type != None:
                    for sample in range(self.samples_per_window):
                        current_error = self.error[label][sample, i_chan]
                        pylab.bar(sample-.35, 2*current_error, width=.7, 
                         bottom=self.mean_time_series[tslabel][sample, i_chan]
                                                            -current_error,
                                      color=cur_color, ec=None, alpha=.3)
                         
            
            # plotting of features; only if features present
            if (self.feature_time_series != None):
                # plot those nice grey circles
                pylab.plot(self.feature_time_series[:, i_chan],
                           'o', color='0.5', label='Feature', alpha=0.5)
                
                for sample in range(self.samples_per_window):
                    if [sample, i_chan] in self.indexlist:
                        # write down value...
                        pylab.text(sample, 
                                   self.feature_time_series[sample, i_chan],
                                   '%.2f' %
                                   self.feature_time_series[sample, i_chan],
                                   ha='center', color='black',
                                   size='xx-small')
                        # ...compute the corresponding color-representation...
                        marker_color = \
                                self.own_colormap(self.normalizer(\
                                   self.feature_time_series[sample, i_chan]))
                        # ...and draw vertical boxes at the feature's position
                        pylab.axvspan(sample - .25, sample + .25,
                                      color=marker_color,
                                      ec=None, alpha=.8)
                                      

            # more format. and rearrangement in case of [phys_arrengement]
            self._format_subplots('mean time series', i_chan,
                                  samples_per_condition_string, ec_2d)
            
        # in case of [phys_arrengement], write the figure title only once
        # and large in the upper left corner of the plot. this fails whenever
        # there are no more channels in that area, as the plot gets cropped
        if self.physiological_arrangement:
            h, l = pylab.gca().get_legend_handles_labels()
            prop = matplotlib.font_manager.FontProperties(size='xx-large')
            figTS.legend(h, l, prop=prop, loc=1)
            if not self.emotiv:
                text_x = .1
                text_y = .92
            else:
                text_x = .4
                text_y = .4
            
            if self.shrink_plots: text_y = 1.2
            figTS.text(text_x, text_y, 'Channel-wise mean time series\n' +
                       samples_per_condition_string,
                       ha='center', color='black', size=32)
            
        
        return figTS
    

    def _format_subplots(self, type, i_chan, samples_per_condition_string, ec=None):
        """ Some time series plot formatting. Mainly writes the channel names
        into the axes, sets titles and rearranges the axes for 
        physiological_arrengement. Also flips axes if desired by setting
        axflip = True """
        # Plot zero line into every single subplot
        pylab.plot(range(1, self.samples_per_window + 1),
                   pylab.zeros(self.samples_per_window), 'k--')

        # current axis limits:
        pylab.gca().set_xlim(xmin=0 - .5, xmax=self.samples_per_window + .5)
        if self.axflip:
            tempax = pylab.gca()
            # reverse ylim
            tempax.set_ylim(tempax.get_ylim()[::-1])
                    
        if not self.physiological_arrangement:
            # every subplot gets a title
            pylab.title('Channel-wise ' + type + '  -  ' +
                        samples_per_condition_string,
                        ha='center', color='black', size='x-small')    
            # and a small legend
            prop = matplotlib.font_manager.FontProperties(size='x-small')
            pylab.legend(prop=prop)
            
        else:
            # do the physiological arrangement
            x, y = ec[self.channel_names[i_chan]]
            w = .07
            h = .065
            if self.shrink_plots:
                w *= 1.2
                h *= 0.9
                x *= 4.0/3.0
                y *= 4.0/3.0
            pylab.gca().set_position([(x + 110) / 220, (y + 110) / 220, w, h])
                  
        # get current axis limits...
        cal = pylab.gca().axis()
        # ... and place channel name at a nice upper left position
        pylab.text((.85 * cal[0] + .15 * cal[1]), (.8 * cal[3] + .2 * cal[2]),
                    self.channel_names[i_chan], ha='center', color='black',
                    size='xx-large')
    

    def _generate_histo_plot(self):
        """ This function generates the actual histogram plot"""
        fighist = pylab.figure()
        
        nr_of_feats = len(self.indexlist)
        # again, number of subplot columns is 8 at most while
        # using as many rows as necessary
        if nr_of_feats <= 8: nr_of_cols = nr_of_feats
        else: nr_of_cols = 8
        nr_of_rows = (len(self.indexlist) - 1) / 8 + 1
        fighist.set_size_inches((5 * nr_of_cols,  3 * nr_of_rows))

        important_features = dict()
        important_feature_names = pylab.array([])
        itercount = 1
        for feat_index in self.indexlist:
            # backgroundcolor for the feature importance text
            bg_color = self.own_colormap(self.normalizer(\
                        self.feature_time_series[tuple(feat_index)]))
            fighist.add_subplot(nr_of_rows, nr_of_cols, itercount)
            itercount += 1
            # plot the actual histogram
            pylab.hist([self.time_series_histo[label]
                        [:, feat_index[0], feat_index[1]]
                        for label in self.mean_time_series.keys()],
                        bins=20, normed=True , histtype='step')
            # write feature importance as fig.text
            cal = pylab.gca().axis()
            pylab.text((.23 * cal[0] + .77 * cal[1]),
                       (.8 * cal[3] + .2 * cal[2]), '%.2f' %
                       self.feature_time_series[feat_index[0], feat_index[1]],
                       fontsize='xx-large',
                       bbox=dict(fc=bg_color, ec=bg_color, alpha=0.6, pad=14))
            # Title uses feature name
            pylab.title('Channel %s at %dms' %
                        (self.channel_names[feat_index[1]],
                         float(feat_index[0]) / 
                         self.sample_data.sampling_frequency * 1000),
                        fontsize='x-large')
            
            # initialize, if no important features known yet
            if important_features.values() == []:
                for label in self.mean_time_series.keys():
                    important_features[label] = \
                        pylab.array(self.time_series_histo[label][:,
                                    feat_index[0], feat_index[1]])
            # stack current important feature with previous
            else:
                for label in self.mean_time_series.keys():
                    important_features[label] = pylab.vstack(
                        (important_features[label],
                         pylab.array(self.time_series_histo[label]
                         [:, feat_index[0], feat_index[1]])))
            # memorize feature name
            important_feature_names = \
             pylab.append(important_feature_names, \
             [('%s' % self.channel_names[feat_index[1]]).ljust(4, '_')\
             + ('%dms' % (float(feat_index[0]) / \
             self.sample_data.sampling_frequency * 1000)).rjust(6, '_')])
        
        self.corr_important_feat_names = important_feature_names
        for label in important_features.keys():
            self.corr_important_feats[label] = \
                pylab.corrcoef(important_features[label])
        # Draw the "feature development" plots of the important features
        self._generate_feature_development_plots(important_features)
        return fighist
    

    def _generate_feature_development_plots(self, important_features):
        """ This function generates the actual histogram plot"""
        # Everything is done class-wise
        for label in important_features.keys():
            # Axis limits are determined by the global maxima
            (minVal, maxVal) = (important_features[label].min(0).min(0),
                                important_features[label].max(0).max(0))                                
            nr_chans = pylab.shape(important_features[label])[0]
                                
            myFig = pylab.figure()
            myFig.set_size_inches((40,nr_chans))
            
            for i_chan in range(nr_chans):
                ax = myFig.add_subplot(nr_chans, 1, i_chan+1)
                
                # cycle line colors
                if (pylab.mod(i_chan,2) == 0): myCol = '#000080'
                else: myCol = '#003EFF'
                # plot features and black zero-line
                pylab.plot(important_features[label][i_chan,:],color=myCol)
                pylab.plot(range(len(important_features[label][i_chan,:])),
                    pylab.zeros(len(important_features[label][i_chan,:])),
                    'k--')
                pylab.ylim((minVal,maxVal))
                xmax = pylab.shape(important_features[label])[1]
                pylab.xlim((0,xmax))
                
                # draw vertical dashed line every 20 epochs
                for vertical_line_position in range(0,xmax+1,20):
                    pylab.axvline(x=vertical_line_position,
                                  ymin=0, ymax=1, color='k', linestyle='--')
                    
                # write title above uppermost subplot
                if i_chan+1 == 1:
                    pylab.title('Feature development: Amplitudes of %s Epochs'
                                % label, fontsize=40)
                # adjust the axes, i.e. remove upper and right,
                # shift the left to avoid overlaps,
                # and lower axis only @ bottom subplot
                if i_chan+1 < nr_chans:
                    self._adjust_spines(ax,['left'],i_chan)
                else:
                    self._adjust_spines(ax,['left', 'bottom'],i_chan)
                    pylab.xlabel('Number of Epoch', fontsize=36)
                # Write feature name next to the axis
                pylab.ylabel(self.corr_important_feat_names[i_chan],
                             fontsize=20, rotation='horizontal')
            # remove whitespace between subplots etc.
            myFig.subplots_adjust(bottom=0.03,left=0.08,right=0.97,
                                  top=0.94,wspace=0,hspace=0)

            self.feature_development_plot[label] = myFig
    

    def _get_corr_plot(self, corr_matrix, label):
        """ Plot the current correlation matrix as filled contour plot
            and return figure instance. """
        figCorrelation = pylab.figure()
        pylab.imshow(corr_matrix, interpolation='nearest', vmin=-1, vmax=1)
        pylab.gca().set_ylim((len(corr_matrix) - 0.5, -0.5))
        pylab.gca().set_yticks(range(len(corr_matrix)))
        pylab.gca().set_yticklabels(self.corr_important_feat_names,
                                    size='xx-small')
        pylab.gca().set_xticks(range(len(corr_matrix)))
        pylab.gca().set_xticklabels(self.corr_important_feat_names,
                                    rotation='vertical', size='xx-small')
        pylab.gca().set_title(label)
        pylab.colorbar()
        
        return figCorrelation
    

#    def _get_electrode_coordinates(self, key):
#        """ Convert the polar coordinates of the electrode positions
#        to cartesian coordinates for the positioning of the physiologically
#        arranged plots. As the position specification also requires a height
#        and width, these values are also passed. Height and width are tuned
#        manually such that the resulting plots look nice. """
#        # coordinate transformation
#        x = (self.ec[key][0] *
#             pylab.cos(self.ec[key][1] / 180 * pylab.pi) + 110) / 220
#        y = (self.ec[key][0] *
#             pylab.sin(self.ec[key][1] / 180 * pylab.pi) + 110) / 220
#        w = .07
#        h = .065
#        if self.shrink_plots:
#            w *= 1.2
#            h *= 0.9
#            x *= 4.0/3.0
#            y *= 4.0/3.0
#        
#        return [x, y, w, h]
    

    def _adjust_spines(self,ax,spines,i_chan):
        """ Essentially, removes most of the axes in the feature development
        plots. Also produces the alternating shift of the left axes. """
        for loc, spine in ax.spines.iteritems():
            if loc in spines:
                if ((loc=='left') and (pylab.mod(i_chan,2) == 0)):
                    spine.set_position(('outward',5))
                else:
                    spine.set_position(('outward',30)) # outward by 10 points
            else:
                spine.set_color('none') # don't draw spine
                
        ax.yaxis.set_ticks_position('left')
        
        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(30)
        else: ax.xaxis.set_ticks([]) # no xaxis ticks
