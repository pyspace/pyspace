#!/usr/bin/env python
""" Unittests which test filtering nodes

.. todo:: Implement tests for VarianceFilterNode

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/08/22


"""


import unittest

import numpy
import scipy
import time
import pylab

import logging

if __name__ == '__main__':
    import sys
    import os
    # The root of the code
    file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(file_path[:file_path.rfind('pySPACE')-1])

    suite = unittest.TestLoader().loadTestsFromName('test_filtering')
    unittest.TextTestRunner(verbosity=2).run(suite)


import pySPACE.missions.nodes.preprocessing.filtering as filtering
import pySPACE.tests.utils.data.test_data_generation as test_data_generation

test_ts_generator = test_data_generation.TestTimeSeriesGenerator()
noise_generator = test_data_generation.GaussianNoise()

def _fourierSpectrum(time_series, channel):
    fourier_spectrum = numpy.absolute(numpy.fft.fft(time_series[25:-25, channel])) # The first and the last data points are not trustworthy for filtering
    return fourier_spectrum[:len(fourier_spectrum)/2]

class SimpleLowPassFilterTestCase(unittest.TestCase):
    """ Test for SimpleLowPassFilterNode
    """

    def setUp(self):
        self.sampling_frequency = 100
        self.time_series \
            = test_ts_generator.generate_test_data(channels=8,
                                                   time_points=8096,
                                                   sampling_frequency=self.sampling_frequency,
                                                   function \
                                                    = test_data_generation.ChannelDependentSine(sampling_frequency=self.sampling_frequency))

    def test_specgram_low_pass(self):
        time_points = 10000
        sampling_frequency = 100.0
        data = numpy.zeros((time_points,1))
        for time_index in range(time_points):
            data[time_index,0] = numpy.sin(2.0 * numpy.pi * 5.0 * (time_index / sampling_frequency))
            data[time_index,0] += numpy.sin(2.0 * numpy.pi * 15.0 * (time_index / sampling_frequency))
            data[time_index,0] += numpy.sin(2.0 * numpy.pi * 25.0 * (time_index / sampling_frequency))
            data[time_index,0] += numpy.sin(2.0 * numpy.pi * 35.0 * (time_index / sampling_frequency))
            data[time_index,0] += numpy.sin(2.0 * numpy.pi * 45.0 * (time_index / sampling_frequency))

        #Generate a time series build out of the data
        from pySPACE.resources.data_types.time_series import TimeSeries
        test_data = TimeSeries(input_array = data,
                               channel_names = ["test_channel_1"],
                               sampling_frequency = sampling_frequency,
                               start_time = 0,
                               end_time = float(time_points) / sampling_frequency)

        lpf_node = filtering.SimpleLowPassFilterNode(cutoff_frequency = 20.0)
        filtered_time_series = lpf_node.execute(test_data)

        lpf_node_fir = filtering.FIRFilterNode([20.0])
        filtered_time_series_fir = lpf_node_fir.execute(test_data)

#        import pylab
#        pylab.figure(0)
#        pylab.specgram(test_data[:, 0], Fs = sampling_frequency)
#        pylab.colorbar()
#        pylab.figure(1)
#        pylab.specgram(filtered_time_series[:, 0], Fs = sampling_frequency)
#        pylab.colorbar()
#        pylab.figure(2)
#        pylab.specgram(filtered_time_series_fir[:, 0], Fs = sampling_frequency)
#        pylab.colorbar()
#        pylab.show()

    def test_specgram_band_pass(self):
        time_points = 10000
        sampling_frequency = 100.0
        data = numpy.zeros((time_points,1))
        for time_index in range(time_points):
            data[time_index,0] = numpy.sin(2.0 * numpy.pi * 5.0 * (time_index / sampling_frequency))
            data[time_index,0] += numpy.sin(2.0 * numpy.pi * 15.0 * (time_index / sampling_frequency))
            data[time_index,0] += numpy.sin(2.0 * numpy.pi * 25.0 * (time_index / sampling_frequency))
            data[time_index,0] += numpy.sin(2.0 * numpy.pi * 35.0 * (time_index / sampling_frequency))
            data[time_index,0] += numpy.sin(2.0 * numpy.pi * 45.0 * (time_index / sampling_frequency))

        pass_band=(20.,30.)

        #Generate a time series build out of the data
        from pySPACE.resources.data_types.time_series import TimeSeries
        test_data = TimeSeries(input_array = data,
                               channel_names = ["test_channel_1"],
                               sampling_frequency = sampling_frequency,
                               start_time = 0,
                               end_time = float(time_points) / sampling_frequency)

        lpf_node = filtering.FFTBandPassFilterNode(pass_band=pass_band)
        filtered_time_series = lpf_node.execute(test_data)

        lpf_node_fir = filtering.FIRFilterNode(pass_band=pass_band)
        filtered_time_series_fir = lpf_node_fir.execute(test_data)

        lpf_node_fir2 = filtering.FIRFilterNode(pass_band=pass_band,window='hann')
        filtered_time_series_fir2 = lpf_node_fir2.execute(test_data)

        lpf_node_iir = filtering.IIRFilterNode(pass_band=pass_band,stop_band_rifle=90)
        filtered_time_series_iir = lpf_node_iir.execute(test_data)

    def test_low_pass_filtering(self):
        lpf_node = filtering.SimpleLowPassFilterNode(cutoff_frequency =  4.0)

        filtered_time_series = lpf_node.execute(self.time_series)

        self.assert_(id(self.time_series) != id(filtered_time_series)) # The object should be different!

        #Channels with frequencies significantly below the cutoff frequencies should remain nearly unchanged
        self.assertAlmostEqual(max(_fourierSpectrum(filtered_time_series, 0)) /
                                   max(_fourierSpectrum(self.time_series, 0)),
                                   1.0,
                                   places = 1)

        pylab.plot(_fourierSpectrum(filtered_time_series, 0))

        #Channels with frequencies significantly above the cutoff frequencies should be nearly completely removed
        self.assertAlmostEqual(max(_fourierSpectrum(filtered_time_series, 7))
                               /max(_fourierSpectrum(self.time_series, 7)),
                                  0.0,
                                  places  = 1)


class FFTBandPassFilterTestCase(unittest.TestCase):
    """ Test for FFTBandPassFilterNode
    """

    def setUp(self):
        self.sampling_frequency = 100
        self.time_series \
            = test_ts_generator.generate_test_data(channels=8,
                                                   time_points=8096,
                                                   sampling_frequency=self.sampling_frequency,
                                                   function \
                                                    = test_data_generation.ChannelDependentSine(sampling_frequency=self.sampling_frequency))

    def test_fft_band__pass_filtering(self):
        bpf_node = filtering.FFTBandPassFilterNode(pass_band = (3.0,5.0))

        filtered_time_series = bpf_node.execute(self.time_series)

        self.assert_(id(self.time_series) != id(filtered_time_series)) # The object should be different!

        #Channels with frequencies within the pass band should remain nearly unchanged
        self.assertAlmostEqual(max(_fourierSpectrum(filtered_time_series, 3)) /
                                   max(_fourierSpectrum(self.time_series, 3)),
                                   1.0,
                                   places = 1)

        #Channels with frequencies significantly outside the pass band should be nearly completely removed
        self.assertAlmostEqual(max(_fourierSpectrum(filtered_time_series, 6))
                               /max(_fourierSpectrum(self.time_series, 6)),
                                  0.0,
                                  places  = 1)


class FIRFilterTestCase(unittest.TestCase):
    """ Test for the band pass filter node
    """

    def setUp(self):
        self.logger = logging.getLogger("TestLogger")

        self.pass_band = [10,20]

        self.filter_node = filtering.FIRFilterNode(self.pass_band)

        time_points = 1000
        channels = 10
        self.data = test_ts_generator.generate_test_data(channels,time_points,function=test_data_generation.Delta())


    def tearDown(self):
        self.filter_node = None


    def testTimeShifting(self):
        time_points = 10
        channels = 1
        import pySPACE.tests.utils.data.test_data_generation as tdg
        counter = tdg.Counter()
        test_data = test_ts_generator.generate_test_data(channels=channels,
                                                         time_points=time_points,
                                                         function=counter)

        filter_node = filtering.FIRFilterNode([10,20],time_shift="middle",taps=3)

        filter_node.initialize_data_dependencies(test_data)

        filter_node.filter_kernel = numpy.array([1,1,1])/3.

        filtered_data = filter_node.execute(test_data)

        # the result should be
        #[[ 0.33333333]
        # [ 1.        ]
        # [ 2.        ]
        # [ 3.        ]
        # [ 4.        ]
        # [ 5.        ]
        # [ 6.        ]
        # [ 8.        ]
        # [ 7.        ]
        # [ 5.66666667]]
        desired_result=numpy.array([1.0/3,1,2,3,4,5,6,7,8,17./3])
        desired_result.shape=(10,1)

        #print filtered_data
        #print desired_result
        self.assertTrue(numpy.allclose(filtered_data,desired_result))


    # Functions for plotting
    # Plot frequency and phase response
def mfreqz(b,a=1):
    pylab.subplot(211)
    plot_freqz_response(b,a)
    pylab.subplot(212)
    plot_phase_response(b,a)

def plot_freqz_response(b,a=1):
    w,h = scipy.signal.freqz(b,a)
    h_dB = 20 * pylab.log10 (abs(h))
    pylab.plot(w/max(w),h_dB)
    pylab.ylim(-150, 5)
    pylab.ylabel('Magnitude (db)')
    pylab.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    pylab.title(r'Frequency response')

def plot_freqz_response_lin(b,a=1):
    w,h = scipy.signal.freqz(b,a)
    pylab.plot(w/max(w),abs(h))
    pylab.ylim(0, 1.1)
    pylab.ylabel('Magnitude')
    pylab.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    pylab.title(r'Frequency response')

def plot_phase_response(b,a=1):
    w,h = scipy.signal.freqz(b,a)
    h_Phase = pylab.unwrap(numpy.arctan2(numpy.imag(h),numpy.real(h)))
    pylab.plot(w/max(w),h_Phase)
    pylab.ylabel('Phase (radians)')
    pylab.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    pylab.title(r'Phase response')
    pylab.subplots_adjust(hspace=0.5)

def plot_kernel(b,a=1):
    pylab.plot(range(0,len(b)),b)
    pylab.ylabel('Coefficient value')
    pylab.xlabel(r'tap')
    pylab.title(r'Kernel structure')

def plot_time_series_response(x,b,a=1):
    import scipy.signal
    y = scipy.signal.lfilter(b,a,x)
    x_zero_tapped = numpy.append(numpy.zeros(len(y)-len(x)),x)
    pylab.subplot(211)
    pylab.plot(range(0,len(x_zero_tapped)),x_zero_tapped)
    pylab.subplot(212)
    pylab.plot(range(0,len(y)),y)
    return y


# Plot step and impulse response
def impz(b,a=1):
    l = len(b)
    impulse = numpy.repeat(0.,l); impulse[0] =1.
    x = numpy.arange(0,l)
    response = scipy.signal.lfilter(b,a,impulse)
    pylab.subplot(211)
    pylab.stem(x, response)
    pylab.ylabel('Amplitude')
    pylab.xlabel(r'n (samples)')
    pylab.title(r'Impulse response')
    pylab.subplot(212)
    step = numpy.cumsum(response)
    pylab.stem(x, step)
    pylab.ylabel('Amplitude')
    pylab.xlabel(r'n (samples)')
    pylab.title(r'Step response')
    pylab.subplots_adjust(hspace=0.5)

def plot_fir_filter(b,a=1):
    pylab.subplot(211)
    plot_kernel(b,a)
    pylab.subplot(212)
    plot_freqz_response(b,a)
    #pylab.subplot(313)
    #plot_phase_response(b)

#class FilterPlayground(unittest.TestCase):
#
#    def testFilterPlayground(self):
#        # create a signal
#        # sampled at 5000Hz
#        sampling_rate = 4096
#        # for 2s
#        end_time = 1.
#        # time points
#        x = numpy.arange(0,end_time,1.0/sampling_rate,dtype=numpy.float64)
#        theta = 2 * numpy.pi * x
#
#        # the actual signal
#        y = numpy.sin(theta*10) + numpy.sin(theta*20) + numpy.sin(theta*300)
#
#        e = 2 * scipy.randn(sampling_rate)
#
#        print x.shape
#        print y.shape
#        print e.shape
#
#        y = y+e
#
#        pylab.plot(x,abs(y))
#        pylab.savefig("signal")
#        pylab.figure()
#
#        # perform 1024 tap fft of the signal
#        fft_order = 4096
#        Y = scipy.fft(y,fft_order)
#        # and calculate at which points the fft is sampled
#        F = pylab.fftfreq(fft_order, d=1.0)
#
#        pylab.rc('text', usetex=True)
#        pylab.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
#        pylab.ylabel(r'$|X(j\omega)|/N$')
#        Y_normalized = Y* 1./fft_order
#        pylab.plot(F,abs(Y_normalized))
#        pylab.savefig("fft_1")
#
#        downsampling_factor = 8
#        y_down = numpy.arange(0,end_time/downsampling_factor,end_time/sampling_rate,dtype=numpy.float64)
#
#        for i in numpy.arange(0,end_time*sampling_rate/downsampling_factor):
#            y_down[i]=y[i*downsampling_factor]
#
#        print y_down.shape
#        # perform 1024 tap fft of the signal
#        fft_order_down = fft_order/downsampling_factor
#        Y_down = scipy.fft(y_down,fft_order_down)
#        # and calculate at which points the fft is sampled
#        F_down = pylab.fftfreq(fft_order_down, d=1.0)
#
#        pylab.figure()
#        pylab.rc('text', usetex=True)
#        pylab.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
#        pylab.ylabel(r'$|X(j\omega)|/N$')
#        Y_down_normalized = Y_down * 1./fft_order_down
#        pylab.plot(F_down,abs(Y_down_normalized))
#        pylab.savefig("fft_2")

        # filtering out frequencies above 15 Hz
        #fc = 256.
        #wc = fc * 2. / sampling_rate
        #taps = 256

        #filter_coeffs = scipy.signal.firwin(taps,wc)
        # plot filter response
        #pylab.figure(figsize=(8,8))
        #plot_fir_filter(filter_coeffs)
        #pylab.savefig("filter")

        #pylab.show()
        #filter the signal
        #y01 = scipy.signal.lfilter(filter_coeffs,1,y)

        #Y01=fft(y01,fft_order)
        #pylab.plot(F,Y01)



