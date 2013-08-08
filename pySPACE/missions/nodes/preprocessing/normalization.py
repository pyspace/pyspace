""" Normalize :class:`~pySPACE.resources.data_types.time_series.TimeSeries` channel-wise

Normalize can mean to translate the values of each channel so that its mean
becomes zero and/or to scale each channels values so that the standard deviation
becomes 1.

"""

import numpy
import logging
import scipy.signal

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries


class DetrendingNode(BaseNode):
    """ Eliminate trend on the selected channels with the given function (default: mean)

    This method takes one function that is used for detrending
    of the selected channels of each time series instance that is
    passed through this node (e.g. the matplotlib.mlab.detrend_mean function).

    **Parameters**
        :detrend_method:
            Method being applied to each selected channel, to detrend.

            (*optional, default: "eval(__import__('pylab').detrend_mean)"*)

        :selected_channels:
            List of channel names, were the method is applied to

            (*optional, default: None = all channels*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Detrending

    :Authors: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/01/06
    """
    def __init__(self, detrend_method=__import__('pylab').detrend_mean,
                 selected_channels=None,**kwargs):
        super(DetrendingNode, self).__init__(**kwargs)

        if selected_channels is not None:
            raise DeprecationWarning("The parameter 'selected_channels' of "
                                     "DetrendingNode is deprecated and will be"
                                     " removed. Use a separate "
                                     "ChannelNameSelectorNode instead.")

        self.set_permanent_attributes(detrend_method=detrend_method,
                                      selected_channels=selected_channels,
                                      selected_channel_indices=None)

    def _execute(self, data):
        """ Apply the detrending method to the given data x and return a new time series """
        #Determine the indices of the channels which will be filtered
        x = data.view(numpy.ndarray)
        if self.selected_channel_indices is None:
            self.selected_channel_names = self.selected_channels \
                if not self.selected_channels is None else data.channel_names
            self.selected_channel_indices = \
                    [data.channel_names.index(channel_name)
                     for channel_name in self.selected_channel_names]

        #Do the actual detrending
        detrended_data = numpy.zeros(x.shape)
        for channel_index in self.selected_channel_indices:
            detrended_data[:, channel_index] = \
                self.detrend_method(x[:, channel_index])

        result_time_series = TimeSeries.replace_data(data, detrended_data)

        return result_time_series


class LocalStandardizationNode(BaseNode):
    """ Z-Score transformation (zero mean, variance of one)

    Each channel of the resulted multidimensional time series has a mean of 0
    and a standard deviation of 1.

    **Parameters**

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Standardization

    :Author: Anett Seeland (Anett.Seeland@dfki.de)
    """
    def __init__(self, **kwargs):
        super(LocalStandardizationNode, self).__init__(**kwargs)

    def _execute(self, x):
        """
        Apply z-score transformation to the given data and return a modified
        time series.
        """
        data = x.view(numpy.ndarray)
        #Do the z-score transformation
        std = numpy.std(data-numpy.mean(data, axis=0), axis=0)
        std = check_zero_division(self, std,  tolerance=10**-15, data_ts=x)
        return TimeSeries.replace_data(x, (data-numpy.mean(data, axis=0)) / std)


class MaximumStandardizationNode(BaseNode):
    """ Standardize by subtracting the mean and dividing by maximum value

    Each channel of the resulted multidimensional time series has a mean of 0
    and the maximum value will be 1.

    **Parameters**

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : MaximumStandardization

    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de),
              Yohannes Kassahun (kassahun@informatik.uni-bremen.de)
    """
    def __init__(self, **kwargs):
        super(MaximumStandardizationNode, self).__init__(**kwargs)

    def _execute(self, x):
        data = x.view(numpy.ndarray)
        mean = numpy.mean(data, axis=0)
        data -= mean

        max_values = numpy.abs(numpy.max(data, axis=0))

        max_values = check_zero_division(self, max_values, tolerance=10**-15,
                                         data_ts=x)

        return TimeSeries.replace_data(x, data/max_values)


class MemoryStandardizationNode(BaseNode):
    """ Z-Score transformation with respect to the last `order` windows

    mean = 1/(order+1) sum_{i=0}^{order} mean[time_{-i}]
    std = sqrt(1/(order+1) sum_{i=0}^{order} var[time_{-i}])

    If the standard deviation is to high because of artifacts,
    we only use the current standard deviation and it is not put into memory.

    **Parameters**
        :order:
            Number of previously occurred time windows being considered
            for calculation of standard deviation and mean

            (*optional, default: 0*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Memory_Standardization
            parameters:
                order : 3

    :Author: Mario Krell (Mario.Krell@dfki.de)
    """
    def __init__(self,
                 order = 0,
                 **kwargs):
        super(MemoryStandardizationNode, self).__init__(**kwargs)
        self.set_permanent_attributes(order=int(order), memory= None)

    def _execute(self, x):
        """
        Apply memory z-score transformation to the given data
        and return a new time series.
        """
        data = x.view(numpy.ndarray)
        # calculate the important measures in an array format
        mean = numpy.mean(data, axis=0)
        var = numpy.var(data, axis=0)

        # initialize the memory when it is first used
        if self.memory is None:
            self.memory=dict()
            self.memory['mean'] = list()
            self.memory['var'] = list()
            for i in range(self.order):
                self.memory['mean'].append(mean)
                self.memory['var'].append(var)

        # extend the memory by the new mean and variance if the
        # std<200 and not zero to exclude big artifacts
        if (var < 40000).all() and (var > 10**-9).all():
            self.memory['var'].append(var)
            self.memory['mean'].append(mean)

        # calculate the mean of the current mean and variance
        # and the latest (order) means and variances in memory
        # and delete the last, because it is no longer needed
            var = numpy.mean(self.memory['var'],axis=0)
            mean = numpy.mean(self.memory['mean'],axis=0)
            self.memory['var'].pop(0)
            self.memory['mean'].pop(0)
        std = numpy.sqrt(var)

        # # code for easy viszualization
        # statistic = numpy.vstack((mean,std))
        # statistic = TimeSeries.replace_data(data, statistic)
        # statistic.sampling_frequency = 1
        # return statistic

        #Do the modified z-score transformation
        std = check_zero_division(self, std,  tolerance=10**-15, data_ts=x)

        return TimeSeries.replace_data(x, (data-numpy.mean(data, axis=0))/std)


class DevariancingNode(BaseNode):
    """ Apply devariancing method on training data and use the result for scaling

    This method takes one function that is used for devariancing
    of the selected channels of the whole data set
    (e.g. a scaling the channels so that they have all standard deviation 1).

    .. note::
        The scaling factors are calculated on a training set
        and remain equal over the whole execution time.
        They do not change with each window.

    .. todo:: Check the sense of this and documentation!
    .. todo:: Find meaningful method!

    **Parameters**
        :devariance_method:
            Method being applied to each selected channel of the signal, to
            devariance.

        :selected_channels:
            List of channel names, were the method is applied to. If nothing
            is specified, all channels are used.

            (*optional, default: None*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Devariancing

    :Authors: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2009/01/06
    """
    def __init__(self,
                 devariance_method,
                 selected_channels= None,
                 **kwargs):
        super(DevariancingNode, self).__init__(**kwargs)

        self.set_permanent_attributes(devariance_method = devariance_method,
                                      selected_channels = selected_channels)

    def is_trainable(self):
        """ Returns whether this node is trainable """
        return True

    def _train(self, x):
        """
        The node gathers all data point of the selected channels
        during training
        """
        # Created lazily a data structure that gathers all values
        # of the selected channels
        if self.selected_channels == None:
            #Determine the indices of the channels which will be filtered
            self.selected_channels =  \
                self.selected_channels if self.selected_channels != None \
                                       else x.channel_names
            self.channel_values = dict(zip(self.selected_channels,
                               [[] for i in range(len(self.selected_channels))]))

        # Store all values of the selected channels in self.channel_values
        for channel in self.selected_channels:
            channel_index = x.channel_names.index(channel)
            self.channel_values[channel].extend(x[:, channel_index])


    def _stop_training(self):
        """
        Uses the devariance method to determine scaling factors for each
        selected channel
        """
        self.scale_factors = dict()
        for channel in self.selected_channels:
            self.scale_factors[channel] = 1.0 / \
                             self.devariance_method(self.channel_values[channel])

    def _execute(self, x):
        """
        Apply devariancing to the given data and return the modified time series
        """
        #Determine the indices of the channels which will be filtered
        selected_channel_indices = [x.channel_names.index(channel_name)
                                      for channel_name in self.selected_channels]

        #Do the actual devariancing, i.e. multiply with the scale factor
        devarianced_data = numpy.zeros(x.shape)
        for channel in self.selected_channels:
            channel_index = x.channel_names.index(channel)
            devarianced_data[:, channel_index] = x[:, channel_index] * \
                                                      self.scale_factors[channel]

        return TimeSeries.replace_data(x, devarianced_data)


class SubsetNormalizationNode(BaseNode):
    """ Z-Score transformation by the mean and variance of a subset of the samples

    Thus, in LRP scenarios, e.g., one could
    perform a shift driven by the first x samples of the window. This would
    arrange the "baseline", such that "pre LRP" is zero. In EEG data
    processing, this procedure is commonly referred to as
    "baseline correction"

    The resulting :class:`~pySPACE.resources.data_types.time_series.TimeSeries`
    are **not** regularized in the sense that they
    are not in (0,1).

    **Parameters**
        :subset:
            List of samples being used to calculate the mean.

        :devariance:
            Also divide by the variance

            (*optional, default: False*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : SubsetNormalization
            parameters :
                subset : eval(range(50)) # or any list

    :Author: Mario Krell (mario.krell@dfki.de)
    """
    def __init__(self, subset, devariance=False, **kwargs):

        super(SubsetNormalizationNode, self).__init__(**kwargs)
        self.set_permanent_attributes(subset=subset,
                                      devariance=devariance)

    def _execute(self, data):
        """ Perform a shift and normalization according
        (whole_data - mean(specific_samples)) / std(specific_samples)
        """
        if self.devariance:
            # code copy from LocalStandardizationNode
            std = numpy.std(data[self.subset],axis=0)
            std = check_zero_division(self, std, tolerance=10**-15, data_ts=data)

            return TimeSeries.replace_data(data,
                        (data-numpy.mean(data[self.subset], axis=0)) / std)
        else:
            return TimeSeries.replace_data(data, \
                        data-numpy.mean(data[self.subset], axis=0))


class EuclideanNormalizationNode(BaseNode):
    """ Scale all channels to norm one

    Together with detrending this is equivalent to standardization.

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : Euclidian_Feature_Normalization

    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2011/07/28
    """
    def __init__(self,
                 selected_channels= None,
                 **kwargs):
        super(EuclideanNormalizationNode, self).__init__(**kwargs)
        self.set_permanent_attributes(selected_channels = selected_channels,
                                      selected_channel_indices = None)

    def _execute(self, x):
        """
        Apply the detrending method to the given data x
        and return a new time series.
        """
        #Determine the indices of the channels which will be filtered
        if None == self.selected_channel_indices:
            self.selected_channel_names = self.selected_channels \
                          if self.selected_channels != None else x.channel_names
            self.selected_channel_indices = \
                    [x.channel_names.index(channel_name) for channel_name in \
                                                    self.selected_channel_names]
        #Do the actual detrending
        detrended_data = numpy.zeros(x.shape)
        for channel_index in self.selected_channel_indices:
            temp = x[:, channel_index]
            temp = temp*numpy.float64(1)/numpy.linalg.norm(temp)
            detrended_data[:, channel_index] = temp
        result_time_series = TimeSeries.replace_data(x, detrended_data)

        return result_time_series


class DcRemovalNode(BaseNode):
    """ Perform a realtime DC removal on the selected channels

    A node that uses the realtime DC removal method by IIR filtering with

    .. math::

        H(z)=\\frac{1-z^{-1}}{1-\\alpha z^{-1}}

    The filters internal state is preserved between blocks.

    For details:

    .. seealso:: R. Lyons, "Understanding Digital Signal Processing", 2nd ed.,
                 13.23.2 Real-Time DC Removal, p.553

    **Parameters**

        :alpha:
            See parameter in formula

        :selected_channels:
            List of channel names, were the method is applied to

            (*optional, default: None = all channels*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : DcRemoval
            parameters :
                alpha : 0.95

    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2011/08/18
    """
    def __init__(self,
                 alpha=0.95,
                 selected_channels=None,
                 **kwargs):
        super(DcRemovalNode, self).__init__(**kwargs)

        a = [1, -1 * alpha]
        b = [1, -1]

        # storage for filter conditions
        internal_state = None

        self.set_permanent_attributes(alpha=alpha,
                                      selected_channels=selected_channels,
                                      a=a,
                                      b=b,
                                      internal_state=internal_state,
                                      selected_channel_indices=None)

    def _execute(self, data):
        """ Apply the :func:`scipy.signal.lfilter` function with the DC removal coefficients on the data """
        #Determine the indices of the channels which will be filtered
        if self.selected_channel_indices is None:
            self.selected_channel_names = self.selected_channels \
                if self.selected_channels is not None else data.channel_names
            self.selected_channel_indices = [
                data.channel_names.index(channel_name)
                for channel_name in self.selected_channel_names]

        # create initial filter conditions for all channels
        if self.internal_state is None:
            self.internal_state = dict()
            for channel_index in xrange(data.shape[1]):
                self.internal_state[channel_index] = \
                    scipy.signal.lfiltic(self.b, self.a, [0, 0])

        # do the actual removal
        cleaned_data = numpy.zeros(data.shape)
        for channel_index in self.selected_channel_indices:
            (cleaned_data[:, channel_index],
             self.internal_state[channel_index]) = \
                scipy.signal.lfilter(self.b, self.a, data[:, channel_index],
                                     zi=self.internal_state[channel_index])

        result_time_series = TimeSeries.replace_data(data, cleaned_data)

        return result_time_series


def check_zero_division(self, data,  tolerance=10**-15, data_ts=None):
    """ Increase to small values in *data* and give warning

    This global function of this module checks whether numpy array
    'data' contains values below the specified 'tolerance'.
    If so, the value of 1 is added to
    this value and a warning is put into the log file.
    Returned is the modified array.

    **Parameters**
        :self:
            Needs the node object for proper logging.

        :data:
            Numpy array which is checked for values below certain tolerance.

        :tolerance:
            Values in data below this value are corrected.

            (*optional, default: 10^-9*)

        :data_ts:
            Actual TimeSeries object to improve information in logging.
            If not specified, the logging message is simplified.

            (*optional, default: None*)
    """
    if sum(data < tolerance):  # elements close to zero
        if not data_ts is None:  # better warning output
            # filter the channel names where std equals zero
            zero_channels = [data_ts.channel_names[index] for (index,elem) in \
                             enumerate(data) if elem < tolerance]
            self._log("Normalization:: Warning: Prevented division by zero during normalization "  \
                      "for channel(s) %s in time interval [%.1f,%.1f]!" % (str(zero_channels),
                                        data_ts.start_time, data_ts.end_time), level=logging.WARNING)
        else:
            self._log("Normalization:: Warning: Prevented division by zero during normalization!", level=logging.WARNING)

        data = [elem + (elem < tolerance) for (index, elem) in enumerate(data)]

    return data


_NODE_MAPPING = {"Standardization": LocalStandardizationNode,
                "Dc_Removal": DcRemovalNode,
                }

