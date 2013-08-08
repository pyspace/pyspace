""" Generate features based on wavelet transformation

.. note:: Currently there is only a simple wrapper around pywt
          http://www.pybytes.com/pywavelets/
          So one has to install this package.
          Scipy does not seem to give a real support for wavelets.
"""
from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.feature_vector import FeatureVector
import warnings
import logging
import numpy
try:
    import pywt
    pywt_error = False
except:
    pywt_error = True

class PywtWaveletNode(BaseNode):
    """ Extract features based on the discrete wavelet transform from pywavelets

    The components of the wavelet transform are returned as new features.

    .. note::
        This node is only a wrapper around the
        pywavelet package (http://www.pybytes.com/pywavelets/)

    **Parameters**

        :wavelet:
            Name of the wanted wavelet, including the number of taps

            The pywt documentation currently names the following possibilities::

                :haar family: haar
                :db family:   db1, db2, db3, db4, db5, db6, db7, db8, db9, db10,
                              db11, db12, db13, db14, db15, db16, db17, db18,
                              db19, db20
                :sym family:  sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9,
                              sym10, sym11, sym12, sym13, sym14, sym15, sym16,
                              sym17, sym18, sym19, sym20
                :coif family: coif1, coif2, coif3, coif4, coif5
                :bior family: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4,
                              bior2.6, bior2.8, bior3.1, bior3.3, bior3.5,
                              bior3.7, bior3.9, bior4.4, bior5.5, bior6.8
                :rbio family: rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4,
                              rbio2.6, rbio2.8, rbio3.1, rbio3.3, rbio3.5,
                              rbio3.7, rbio3.9, rbio4.4, rbio5.5, rbio6.8
                :dmey family: dmey

            which refer to Haar, Daubechies, Symlets, Coiflets, Biorthogonal,
            Reverse biorthogonal and Discrete Meyer wavelets
            being used to transform the time series.

            (*recommended, default: 'haar'*)

        :mode:
            One of the pywt extension modes::

                ['zpd', 'cpd', 'sym', 'ppd', 'sp1', 'per']

            which mean zero, constant, symmetric, periodic and smooth padding and
            periodization. The latter uses the minimum number
            of coefficients in comparison to periodic padding.

            (*optional, default: 'sym'*)

    **Exemplary Call**

    .. code-block:: yaml

        -
            node : DWT
            parameters :
                mode : 'cpd'
                wavelet: 'haar'

    :Input: TimeSeries
    :Output: FeatureVector
    :requires: pywavelets
    :Author: Mario Michael Krell (mario.krell@dfki.de)
    :Created: 2012/12/04
    """

    def __init__(self, wavelet = None, mode = 'sym',
                 *args, **kwargs):
        super(PywtWaveletNode, self).__init__(*args, **kwargs)

        if not wavelet in pywt.wavelist():
            warnings.warn("No or wrong wavelet specified (%s). Using 'haar'."%str(wavelet))
            wavelet = "haar"

        if not mode in pywt.MODES.modes:
            warnings.warn("Wrong mode (%s) specified. Using 'sym'."%mode)
            mode = 'sym'

        self.set_permanent_attributes(wavelet=wavelet,
                                      mode=mode,
                                      feature_names=None,
                                      channel_names=None)

    def _execute(self, x):
        """ Extract the wavelet features from the given data x

        The feature names will get an A for approximation and
        D for details coefficients.
        So on example name is: *Waveletname_Channelname_DetailsIndex*,
        where index is the position of the coefficient in the
        transformed list. Details is *A* or *D*.
        Each channel is processed separately and its name is used in *channelname*.
        """
        y = x.view(numpy.ndarray)

        new_data=[]
        if self.feature_names is None:
            feature_names=[]
        if self.channel_names is None:
            self.channel_names = x.channel_names
        for n, channel in enumerate(self.channel_names):
            cA, cD = pywt.dwt(y[n], wavelet=self.wavelet, mode=self.mode)
            for index, value in enumerate(cA):
                new_data.append(value)
                if self.feature_names is None:
                    feature_names.append(self.wavelet+"_"+channel+"_A"+str(index))
            for index, value in enumerate(cD):
                new_data.append(value)
                if self.feature_names is None:
                    feature_names.append(self.wavelet+"_"+channel+"_D"+str(index))
        if self.feature_names is None:
            self.feature_names = feature_names

        feature_vector =\
            FeatureVector(numpy.atleast_2d(new_data).astype(numpy.float64),
                      self.feature_names)

        return feature_vector

_NODE_MAPPING = {"DWT": PywtWaveletNode}