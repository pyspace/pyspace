""" Reader objects and main class for continuous data (time series)

Depending on the storage format, the fitting reader is loaded and takes care
of reading the files.

.. todo:: unify with analyzer collection!
        eeg source and analyzer sink node should work together
        this connection should be documented when tested
"""

import os
import glob
import re
import numpy
import scipy
from scipy.io import loadmat
import warnings
import csv
from pySPACE.missions.support.windower import MarkerWindower
import logging

from pySPACE.resources.dataset_defs.base import BaseDataset
from pySPACE.missions.support.WindowerInterface import AbstractStreamReader


class StreamDataset(BaseDataset):
    """ Wrapper for dealing with stream datasets like raw EEG datasets
    
    For loading EEG data you need the
    :class:`~pySPACE.missions.nodes.source.time_series_source.Stream2TimeSeriesSourceNode`
    as described in :ref:`tutorial_node_chain_operation`.

    If ``file_name`` is given in the meta data, the corresponding file is
    loaded, otherwise ``storage_format`` is used.
    Some formats are already supported, like EEG data in the .eeg/.vhdr/.vmrk 
    format and other streaming data in edf or csv format. It is also possible to
    load EEGLAB format (.set/.fdt) which itself can import a variety of 
    different EEG formats (http://sccn.ucsd.edu/eeglab/).

    **csv**

    Labels can be coded with the help of an extra channel as a column
    in the csv-file or an extra file.
    Normally the label is transformed immediately to the label
    or this is done later on with extra algorithms.

    The file suffix should be *csv*.

    Special Parameters in the metadata:

        :sampling_frequency:
            Frequency of the input data (corresponds to
            1/(number of samples of one second))

            (*optional, default: 1*)

        :marker:
            Name of the marker channel. If it is not found,
            no marker is forwarded.

            (*optional, default: 'marker'*)

        :marker_file:
            If the marker is not a column in the data file,
            an external csv file in the same folder can be specified
            with one column with the heading named like the *marker*
            parameter and one column named *time* with increasing
            numbers, which correspond to the index in the data file.
            (First sample corresponds index one.)
            Here, the relative path is needed as for file_name.

            (*optional, default: None*)

    **BP_eeg**

    Here the standard BrainProducts format is expected with the corresponding
    *.vhdr* and *.vmrk* with the same base name as the *.eeg* file.

    **set**
    
    EEGLABs format with two files (extension .set and .fdt) is expected.

    **edf**

    When using the European Data Format there are two different specifications
    that are supported:
    Plain EDF (see `EDF Spec <http://www.edfplus.info/specs/edf.html>`_) and
    EDF+ (see `EDF+ Spec <http://www.edfplus.info/specs/edfplus.html>`_).

    When using EDF there is no annotation- or marker-channel inside the data-
    segment. You can process the data originating from a EDF file but be sure,
    that you don't have any marker-information at hand, to later cut
    the continuous data into interesting segments.

    EDF+ extended the original EDF-Format by an annotations-channel
    (named 'EDF+C') and added a feature to combine non-continuous
    data segments (named 'EDF+D') in one file.
    The EDF+C Format is fully supported i.e. the annotations-channel is
    parsed and is forwarded in combination with the corresponding data
    so that the data can later be cut into meaningful segments (windowing).
    Files, which make use of the EDF+D option, can be streamed - BUT: The
    information about different segments in the file is completely ignored!
    The file is treated as if it contains EDF+C data. The full support for
    EDF+D files may be integrated in a future release.

    In any case, the file suffix should be *edf*.
    
    .. warning:: Currently only one streaming dataset can be loaded
        as testing data.

    .. todo:: Implement loading of training and testing data.
    
    **Parameters**
    
        :dataset_md:
            A dictionary with all the meta data.
            
            (*optional, default: None*)
            
        :dataset_dir: 
            The (absolute) directory of the dataset.
            
            (*obligatory, default: None*)
            
    :Author:  Johannes Teiwes (johannes.teiwes@dfki.de)
    :Date: 2010/10/13
    :refactored: 2013/06/10 Johannes Teiwes and Mario Michael Krell
    """
    def __init__(self, dataset_md=None, dataset_dir=None, **kwargs):
        super(StreamDataset, self).__init__(dataset_md=dataset_md)
        self.dataset_dir = dataset_dir
        if not self.meta_data.has_key('storage_format'):
            warnings.warn(
                str("Storage Format not set for current dataset in %s" %
                    dataset_dir))

        if self.meta_data.has_key("file_name"):
            data_files = [os.path.join(dataset_dir,self.meta_data["file_name"])]
            if not "storage_format" in self.meta_data:
                self.meta_data["storage_format"] = \
                    os.path.splitext(data_files[0])[1].lower()

        elif self.meta_data.has_key('storage_format'):
            self.meta_data["storage_format"] = \
                self.meta_data['storage_format'].lower()
            # mapping of storage format to file suffix
            suffix = self.meta_data['storage_format']
            if "eeg" in suffix:
                suffix = "eeg"
            # searching files
            data_files = glob.glob(os.path.join(
                dataset_dir, str("*.%s" % suffix)))
            if len(data_files) == 0:
                raise IOError, str("Cannot find any .%s file in %s" %
                    (suffix, dataset_dir))
            if len(data_files) != 1:
                raise IOError, str("Found more than one *.%s file in %s" %
                    (suffix, dataset_dir))
        else:
            # assume .eeg files
            data_files = glob.glob(dataset_dir + os.sep + "*.eeg")
            if(len(data_files) == 0):
                data_files = glob.glob(dataset_dir + os.sep + "*.dat")

            assert len(data_files) == 1, \
                "Error locating eeg-data files (.eeg/.dat)"

        self.data_file = data_files[0]
        self.reader = None
        
        ec_files = glob.glob(dataset_dir + os.sep + "*.elc")
        assert len(ec_files) <= 1, "More than one electrode position file found!"
        if len(ec_files)==1:
            try:
                ec = {}
                ec_file = open(ec_files[0], 'r')
                while (ec_file.readline() != "Positions"):
                    pass
                
                for line in ec_file:
                    if line == "Labels":
                        break
                    pair = line.split(":")
                    ec[pair[0]] = \
                        numpy.array([int(x) for x in pair[1].split(" ")])
                
                nas = ec["NAS"]
                lpa = ec["LPA"]
                rpa = ec["RPA"]
                origin = (rpa + lpa) * 0.5
                vx = nas - origin
                vx = vx / numpy.linalg.norm(vx)
                vz = numpy.cross(vx, lpa - rpa)
                vz = vz / numpy.linalg.norm(vz)
                vy = numpy.cross(vz, vx)
                vy = vy / numpy.linalg.norm(vy)
                rotMat = numpy.linalg.inv(numpy.matrix([vx, vy, vz]))
                transMat = numpy.dot(-rotMat, origin)
                
                for k, v in self.ec.iteritems():
                    ec[k] = numpy.dot(transMat, numpy.dot(v, rotMat))
                    
                self.meta_data["electrode_coordinates"] = ec
                self._log("Loaded dataset specific electrode position file", logging.INFO)
            except Exception, e:
                print e
                #self.meta_data["electrode_coordinates"] = StreamDataset.ec
            finally:
                file.close()
    
    # Spherical electrode coordinates (x-axis points to the right, 
    # y-axis to the front, z-axis runs through the vertex; 3 params: r (radius)
    # set to 1 on standard caps, theta (angle between z-axis and line connecting
    # point and coordinate origin, < 0 in left hemisphere, > 0 in right 
    # hemisphere) and phi (angle between x-axis and projection of the line 
    # connecting the point and coordinate origin on the xy plane, > 0 for front
    # right and back left quadrants, < 0 for front left and back right)) are 
    # exported from analyzer2 (generic export; saved in header file) and
    # converted to Cartesian coordinates via
    # x = r * sin(rad(theta)) * cos(rad(phi))
    # y = r * sin(rad(theta)) * sin(rad(phi))
    # z = r * cos(rad(theta))
    # electrodes FP1/Fp1 and FP2/Fp2 have same coordinates
    ec = {  'CPP5h': (-0.72326832569043442, -0.50643793379675761, 0.46947156278589086),
            'AFF1h': (-0.11672038362490393, 0.83050868362971098, 0.5446390350150272),
            'O2': (0.30901699437494745, -0.95105651629515353, 6.123233995736766e-17),
            'O1': (-0.30901699437494745, -0.95105651629515353, 6.123233995736766e-17),
            'FCC6h': (0.82034360384187455, 0.1743694158206236, 0.5446390350150272),
            'TPP8h': (0.86385168719631511, -0.47884080932566353, 0.15643446504023092),
            'PPO10h': (0.69411523801289432, -0.69411523801289421, -0.1908089953765448),
            'TP7': (-0.95105651629515353, -0.3090169943749474, 6.123233995736766e-17),
            'CPz': (2.293803827831453e-17, -0.37460659341591201, 0.92718385456678742),
            'CCP4h': (0.54232717509597328, -0.18673822182292288, 0.8191520442889918),
            'TP9': (-0.87545213915725872, -0.28445164312142457, -0.3907311284892736),
            'TP8': (0.95105651629515353, -0.3090169943749474, 6.123233995736766e-17),
            'FCC5h': (-0.82034360384187455, 0.1743694158206236, 0.5446390350150272),
            'CPP2h': (0.16769752048474765, -0.54851387399083462, 0.8191520442889918),
            'FFC1h': (-0.16769752048474765, 0.54851387399083462, 0.8191520442889918),
            'TPP7h': (-0.86385168719631511, -0.47884080932566353, 0.15643446504023092),
            'PO10': (0.54105917752298882, -0.7447040698476447, -0.3907311284892736),
            'FTT8h': (0.96671406082679645, 0.17045777155400837, 0.19080899537654492),
            'Oz': (6.123233995736766e-17, -1.0, 6.123233995736766e-17),
            'AFF2h': (0.11672038362490393, 0.83050868362971098, 0.5446390350150272),
            'CCP3h': (-0.54232717509597328, -0.18673822182292288, 0.8191520442889918),
            'CP1': (-0.35777550984135725, -0.37048738597260156, 0.85716730070211233),
            'CP2': (0.35777550984135725, -0.37048738597260156, 0.85716730070211233),
            'CP3': (-0.66008387202973706, -0.36589046498407451, 0.6560590289905075),
            'CP4': (0.66008387202973706, -0.36589046498407451, 0.6560590289905075),
            'CP5': (-0.87157241273869712, -0.33456530317942912, 0.35836794954530016),
            'CP6': (0.87157241273869712, -0.33456530317942912, 0.35836794954530016),
            'FFT7h': (-0.86385168719631511, 0.47884080932566353, 0.15643446504023092),
            'FTT7h': (-0.96671406082679645, 0.17045777155400837, 0.19080899537654492),
            'PPO5h': (-0.5455036073850148, -0.7790598895575418, 0.30901699437494745),
            'AFp1': (-0.13661609910710645, 0.97207405517694545, 0.19080899537654492),
            'AFp2': (0.13661609910710645, 0.97207405517694545, 0.19080899537654492),
            'FT10': (0.87545213915725872, 0.28445164312142457, -0.3907311284892736),
            'POO9h': (-0.44564941557132876, -0.87463622477252034, -0.1908089953765448),
            'POO10h': (0.44564941557132876, -0.87463622477252034, -0.1908089953765448),
            'T8': (1.0, -0.0, 6.123233995736766e-17),
            'FT7': (-0.95105651629515353, 0.3090169943749474, 6.123233995736766e-17),
            'FT9': (-0.87545213915725872, 0.28445164312142457, -0.3907311284892736),
            'FT8': (0.95105651629515353, 0.3090169943749474, 6.123233995736766e-17),
            'FFC3h': (-0.48133227677866169, 0.53457365038161042, 0.69465837045899737),
            'P10': (0.74470406984764481, -0.54105917752298871, -0.3907311284892736),
            'AF8': (0.58778525229247325, 0.80901699437494734, 6.123233995736766e-17),
            'T7': (-1.0, -0.0, 6.123233995736766e-17),
            'AF4': (0.36009496929665602, 0.89126632448749754, 0.27563735581699916),
            'AF7': (-0.58778525229247325, 0.80901699437494734, 6.123233995736766e-17),
            'AF3': (-0.36009496929665602, 0.89126632448749754, 0.27563735581699916),
            'P2': (0.28271918486560565, -0.69975453766943163, 0.6560590289905075),
            'P3': (-0.5450074457687164, -0.67302814507021891, 0.50000000000000011),
            'CPP4h': (0.48133227677866169, -0.53457365038161042, 0.69465837045899737),
            'P1': (-0.28271918486560565, -0.69975453766943163, 0.6560590289905075),
            'P6': (0.72547341102583851, -0.63064441484306177, 0.27563735581699916),
            'P7': (-0.80901699437494745, -0.58778525229247314, 6.123233995736766e-17),
            'P4': (0.5450074457687164, -0.67302814507021891, 0.50000000000000011),
            'P5': (-0.72547341102583851, -0.63064441484306177, 0.27563735581699916),
            'P8': (0.80901699437494745, -0.58778525229247314, 6.123233995736766e-17),
            'P9': (-0.74470406984764481, -0.54105917752298871, -0.3907311284892736),
            'PPO2h': (0.11672038362490393, -0.83050868362971098, 0.5446390350150272),
            'F10': (0.74470406984764481, 0.54105917752298871, -0.3907311284892736),
            'TPP9h': (-0.87463622477252045, -0.4456494155713287, -0.1908089953765448),
            'FTT9h': (-0.96954172390250215, 0.1535603233115839, -0.1908089953765448),
            'CCP5h': (-0.82034360384187455, -0.1743694158206236, 0.5446390350150272),
            'AFF6h': (0.5455036073850148, 0.7790598895575418, 0.30901699437494745),
            'FFC2h': (0.16769752048474765, 0.54851387399083462, 0.8191520442889918),
            'FCz': (2.293803827831453e-17, 0.37460659341591201, 0.92718385456678742),
            'FCC2h': (0.1949050434465294, 0.19490504344652934, 0.96126169593831889),
            'CPP1h': (-0.16769752048474765, -0.54851387399083462, 0.8191520442889918),
            'FTT10h': (0.96954172390250215, 0.1535603233115839, -0.1908089953765448),
            'Fz': (4.3297802811774658e-17, 0.70710678118654746, 0.70710678118654757),
            'TTP8h': (0.96671406082679645, -0.17045777155400837, 0.19080899537654492),
            'FFT9h': (-0.87463622477252045, 0.4456494155713287, -0.1908089953765448),
            'Pz': (4.3297802811774658e-17, -0.70710678118654746, 0.70710678118654757),
            'FFC4h': (0.48133227677866169, 0.53457365038161042, 0.69465837045899737),
            'C3': (-0.70710678118654746, -0.0, 0.70710678118654757),
            'C2': (0.39073112848927372, -0.0, 0.92050485345244037),
            'C1': (-0.39073112848927372, -0.0, 0.92050485345244037),
            'C6': (0.92718385456678731, -0.0, 0.37460659341591218),
            'C5': (-0.92718385456678731, -0.0, 0.37460659341591218),
            'C4': (0.70710678118654746, -0.0, 0.70710678118654757),
            'TTP7h': (-0.96671406082679645, -0.17045777155400837, 0.19080899537654492),
            'FC1': (-0.35777550984135725, 0.37048738597260156, 0.85716730070211233),
            'FC2': (0.35777550984135725, 0.37048738597260156, 0.85716730070211233),
            'FC3': (-0.66008387202973706, 0.36589046498407451, 0.6560590289905075),
            'FC4': (0.66008387202973706, 0.36589046498407451, 0.6560590289905075),
            'FC5': (-0.87157241273869712, 0.33456530317942912, 0.35836794954530016),
            'FC6': (0.87157241273869712, 0.33456530317942912, 0.35836794954530016),
            'FCC1h': (-0.1949050434465294, 0.19490504344652934, 0.96126169593831889),
            'CPP6h': (0.72326832569043442, -0.50643793379675761, 0.46947156278589086),
            'F1': (-0.28271918486560565, 0.69975453766943163, 0.6560590289905075),
            'F2': (0.28271918486560565, 0.69975453766943163, 0.6560590289905075),
            'F3': (-0.5450074457687164, 0.67302814507021891, 0.50000000000000011),
            'F4': (0.5450074457687164, 0.67302814507021891, 0.50000000000000011),
            'F5': (-0.72547341102583851, 0.63064441484306177, 0.27563735581699916),
            'F6': (0.72547341102583851, 0.63064441484306177, 0.27563735581699916),
            'F7': (-0.80901699437494745, 0.58778525229247314, 6.123233995736766e-17),
            'F8': (0.80901699437494745, 0.58778525229247314, 6.123233995736766e-17),
            'F9': (-0.74470406984764481, 0.54105917752298871, -0.3907311284892736),
            'FFT8h': (0.86385168719631511, 0.47884080932566353, 0.15643446504023092),
            'FFT10h': (0.87463622477252045, 0.4456494155713287, -0.1908089953765448),
            'Cz': (0.0, 0.0, 1.0),
            'FFC5h': (-0.72326832569043442, 0.50643793379675761, 0.46947156278589086),
            'FCC4h': (0.54232717509597328, 0.18673822182292288, 0.8191520442889918),
            'TP10': (0.87545213915725872, -0.28445164312142457, -0.3907311284892736),
            'POz': (5.6364666119006729e-17, -0.92050485345244037, 0.39073112848927372),
            'CPP3h': (-0.48133227677866169, -0.53457365038161042, 0.69465837045899737),
            'FFC6h': (0.72326832569043442, 0.50643793379675761, 0.46947156278589086),
            'PPO1h': (-0.11672038362490393, -0.83050868362971098, 0.5446390350150272),
            'Fpz': (6.123233995736766e-17, 1.0, 6.123233995736766e-17),
            'POO2': (0.13661609910710645, -0.97207405517694545, 0.19080899537654492),
            'POO1': (-0.13661609910710645, -0.97207405517694545, 0.19080899537654492),
            'I1': (-0.28651556797120703, -0.88180424668940116, -0.37460659341591207),
            'I2': (0.28651556797120703, -0.88180424668940116, -0.37460659341591207),
            'PPO9h': (-0.69411523801289432, -0.69411523801289421, -0.1908089953765448),
            'FP1': (-0.30901699437494745, 0.95105651629515353, 6.123233995736766e-17),
            'OI2h': (0.15356032331158395, -0.96954172390250215, -0.1908089953765448),
            'FP2': (0.30901699437494745, 0.95105651629515353, 6.123233995736766e-17),
            'CCP6h': (0.82034360384187455, -0.1743694158206236, 0.5446390350150272),
            'FCC3h': (-0.54232717509597328, 0.18673822182292288, 0.8191520442889918),
            'PO8': (0.58778525229247325, -0.80901699437494734, 6.123233995736766e-17),
            'PO9': (-0.54105917752298882, -0.7447040698476447, -0.3907311284892736),
            'PO7': (-0.58778525229247325, -0.80901699437494734, 6.123233995736766e-17),
            'PO4': (0.36009496929665602, -0.89126632448749754, 0.27563735581699916),
            'PO3': (-0.36009496929665602, -0.89126632448749754, 0.27563735581699916),
            'Fp1': (-0.30901699437494745, 0.95105651629515353, 6.123233995736766e-17),
            'Fp2': (0.30901699437494745, 0.95105651629515353, 6.123233995736766e-17),
            'PPO6h': (0.5455036073850148, -0.7790598895575418, 0.30901699437494745),
            'CCP2h': (0.1949050434465294, -0.19490504344652934, 0.96126169593831889),
            'Iz': (5.6773636985816068e-17, -0.92718385456678742, -0.37460659341591207),
            'AFF5h': (-0.5455036073850148, 0.7790598895575418, 0.30901699437494745),
            'TPP10h': (0.87463622477252045, -0.4456494155713287, -0.1908089953765448),
            'OI1h': (-0.15356032331158395, -0.96954172390250215, -0.1908089953765448),
            'CCP1h': (-0.1949050434465294, -0.19490504344652934, 0.96126169593831889)
            }
    
    def store(self, result_dir, s_format="multiplexed"):
        """ Not yet implemented! """
        raise NotImplementedError("Storing of StreamDataset is currently not supported!")
    
    @staticmethod
    def project2d(ec_3d):
        """
        Take a dictionary of 3d Cartesian electrode coordinates and return a 
        dictionary of their 2d projection in Cartesian coordinates.
        """
        keys = []
        x = []
        y = []
        z = []
        for k, v in ec_3d.iteritems():
            keys.append(k)
            x.append(v[0])
            y.append(v[1])
            z.append(v[2])
            
        x = numpy.array(x)
        y = numpy.array(y)
        z = numpy.array(z)
        
        z = z - numpy.max(z)
        # get spherical coordinates: normally this can be done via:
        # phi = deg(atan2(y,x)); if < -90 -> + 180, if > 90 -> - 180
        # theta = deg(arccos(z/r)); if x < 0 -> * (-1) 
        hypotxy = numpy.hypot(x, y)
        r = numpy.hypot(hypotxy, z)
        phi = numpy.arctan2(z, hypotxy)
        theta = numpy.arctan2(y, x)
        
        phi = numpy.maximum(phi, 0.001)
        
        r2 = r / numpy.power(numpy.cos(phi), 0.2)
        
        x = r2 * numpy.cos(theta) * 60
        y = r2 * numpy.sin(theta) * 60
        
        ec_2d = {}
        for i in xrange(0, len(keys)):
            ec_2d[keys[i]] = (x[i], y[i])
            
        return ec_2d

    def set_window_defs(self, window_definition, nullmarker_stride_ms=1000, 
                        no_overlap=False, data_consistency_check=False):
        """ Takes the window definition dictionary for later reading

        The parameters are later on mainly forwarded to the
        :class:`~pySPACE.missions.support.windower.MarkerWindower`.
        To find more about these parameters, check out its documentation.
        """
        self.window_definition = window_definition
        self.nullmarker_stride_ms = nullmarker_stride_ms
        self.no_overlap = no_overlap
        self.data_consistency_check = data_consistency_check

    def get_data(self, run_nr, split_nr, train_test):
        if not (run_nr, split_nr, train_test) == (0, 0, "test"):
            return self.data[(run_nr, split_nr, train_test)]
        if self.meta_data.has_key('storage_format'):
            if "bp_eeg" in self.meta_data['storage_format']:
                # remove ".eeg" suffix
                self.reader = EEGReader(self.data_file[:-4],
                                        blocksize=100)
            elif "set" in self.meta_data['storage_format']:
                self.reader = SETReader(self.data_file[:-4])
            elif "edf" in self.meta_data['storage_format']:
                self.reader = EDFReader(self.data_file)
            elif "csv" in self.meta_data['storage_format']:
                sf = self.meta_data.get("sampling_frequency", 1)
                try:
                    mf = os.path.join(self.dataset_dir,
                                      self.meta_data["marker_file"])
                except KeyError:
                    mf = None
                if "marker" in self.meta_data:
                    marker = self.meta_data["marker"]
                else:
                    marker = "marker"
                self.reader = CsvReader(self.data_file, sampling_frequency=sf,
                                        marker=marker, marker_file=mf)
        else:
            self.reader = EEGReader(self.data_file, blocksize=100)

        # Creates a windower that splits the training data into windows
        # based in the window definitions provided
        # and assigns correct labels to these windows
        self.marker_windower = MarkerWindower(
            self.reader, self.window_definition,
            nullmarker_stride_ms=self.nullmarker_stride_ms,
            no_overlap=self.no_overlap,
            data_consistency_check=self.data_consistency_check)
        return self.marker_windower


def parse_float(param):
    """ Work around to catch colon instead of floating point """
    try:
        return float(param)
    except ValueError, e:
        warnings.warn("Failed float conversion from csv file.")
        return float(param.replace(".","").replace(",","."))


def get_csv_handler(file_handler):
    """Helper function to get a DictReader from csv"""
    try:
        dialect = csv.Sniffer().sniff(file_handler.read(2048))
        file_handler.seek(0)
        return csv.DictReader(file_handler, dialect=dialect)
    except csv.Error, e:
        class excel_space(csv.excel):
            delimiter = ' '
        warnings.warn(str(e))
        csv.register_dialect("excel_space", excel_space)
        file_handler.seek(0)
        return csv.DictReader(file_handler, dialect=excel_space)


class CsvReader(AbstractStreamReader):
    """ Load time series data from csv file

    **Parameters**

        :file_path:
            Path of the file to be loaded.

            (*optional, default: 'data.csv'*)

        :sampling_frequency:
            Underlying sampling frequency of the data in Hz

            (*optional, default: 1*)

        :marker:
            Name of the marker channel. If it is not found,
            no marker is forwarded.

            (*optional, default: 'marker'*)

        :marker_file:
            If the marker is not a column in the data file,
            an external csv file in the same folder can be specified
            with one column with the heading named like the *marker*
            parameter and one column named *time* with increasing
            numbers, which correspond to the index in the data file.
            (first time point gets zero.)
            Here the absolute path is needed.

            (*optional, default: None*)
    """
    def __init__(self, file_path, sampling_frequency=1, marker="marker",
                 marker_file=None):
        try:
            self.file = open(file_path, "r")
        except IOError as io:
            warnings.warn("Failed to open file at [%s]" % file_path)
            raise io

        self._dSamplingInterval = sampling_frequency
        self.marker = marker
        self._markerids = dict()
        self._markerNames = dict()
        self.callbacks = list()
        self.new_marker_id = 1
        self.time_index = 1

        try:
            if not marker_file is None:
                marker_file = open(marker_file, "r")
        except IOError:
            warnings.warn("Failed to open marker file at [%s]. Now ignored."
                          % marker_file)

        self._markerids["null"] = 0
        self._markerNames[0] = "null"

        self.DictReader = get_csv_handler(self.file)

        self.first_entry = self.DictReader.next()
        self._channelNames = self.first_entry.keys()

        self.MarkerReader = None
        if not marker_file is None:
            self.MarkerReader = get_csv_handler(marker_file)

        if not self.MarkerReader is None:
            self.update_marker()
            if self.next_marker[0] == self.time_index:
                self.first_marker = self.next_marker[1]
                self.update_marker()
            else:
                self.first_marker = ""
        elif self.marker in self._channelNames:
            self.first_marker = self.first_entry.pop(self.marker)
        else:
            self.first_marker = ""

    @property
    def dSamplingInterval(self):
        """ actually the sampling frequency """
        return self._dSamplingInterval

    @property
    def stdblocksize(self):
        """ standard block size (int) """
        return 1

    @property
    def markerids(self):
        """ mapping of markers/events in stream and unique integer (dict)

        The dict has to contain the mapping 'null' -> 0 to use the
        nullmarkerstride option in the windower.
        """
        return self._markerids

    @property
    def channelNames(self):
        """ list of channel/sensor names """
        return self._channelNames

    @property
    def markerNames(self):
        """ inverse mapping of markerids (dict) """
        return self._markerNames

    def regcallback(self, func):
        """ register a function as consumer of the stream """
        self.callbacks.append(func)

    def read(self, nblocks=1):
        """ Read *nblocks* of the stream and pass it to registers functions """
        n = 0
        while nblocks == -1 or n < nblocks:
            if not self.first_entry is None:
                samples, marker = self.first_entry, self.first_marker
                self.first_entry = None
            else:
                try:
                    samples = self.DictReader.next()
                except IOError:
                    break
                if not self.MarkerReader is None:
                    if self.next_marker[0] == self.time_index:
                        marker = self.next_marker[1]
                        self.update_marker()
                    else:
                        marker = ""
                elif self.marker in self._channelNames:
                    marker = samples.pop(self.marker)
                else:
                    marker = ""
            # add marker to dict
            if not marker == "" and not marker in self._markerids:
                self._markerids[marker] = self.new_marker_id
                self._markerNames[self.new_marker_id] = marker
                self.new_marker_id += 1
            # convert marker to array
            markers = numpy.ones(1)*(-1)
            if not marker == "":
                markers[0] = self._markerids[marker]
            # convert samples to array
            array_samples = numpy.zeros((len(self.channelNames),1))
            for index, channel in enumerate(self.channelNames):
                array_samples[index] = parse_float(samples[channel])
            n += 1
            for c in self.callbacks:
                c(array_samples, markers)
        self.time_index += 1
        return n

    def update_marker(self):
        """Update `next_marker` from `MarkerReader` information"""
        try:
            next = self.MarkerReader.next()
            self.next_marker = (next["time"], next[self.marker])
        except IOError:
            pass

class EDFReader(AbstractStreamReader):
    """ Read EDF-Data

    On Instantiation it will automatically assign the value
    for the blocksize coded in the edf-file to its own
    attribute 'stdblocksize'.
    The Feature, that different signals can have different
    sampling rates is eliminated in a way, that every value
    of a lower sampled signal is repeated so that it fits
    the highest sampling rate present in the dataset. This
    is needed to have the same length for every signal
    in the returned array.
    """

    def __init__(self, abs_edffile_path):
        """Initializes module and opens specified file."""
        try:
            self.edffile = open(abs_edffile_path, "r")
        except IOError as io:
            warnings.warn(str("failed to open file at [%s]" % abs_edffile_path))
            raise io

        # variables to later overwrite
        # the properties from AbstractStreamReader
        self.callbacks = list()
        self._dSamplingInterval = 0
        self._stdblocksize = 0
        self._markerids = dict()
        self._channelNames = dict()
        self._markerNames = dict()

        # gains, frequency for each channel
        self.gains = []
        self.phy_min = []
        self.dig_min = []
        self.frequency = []
        self.num_channels = 0
        self.num_samples = []
        self.edf_plus = False
        self.edf_header_length = 0
        self.annotations = None
        self.num_samples_anno = None
        self.timepoint = 0.0

        self.generate_meta_data()

    def __str__(self):
        return ("EDFReader Object (%d@%s)\n" + \
            "\tEDF File:\t %s\n" + \
            "\tFile Format:\t %s\n" + \
            "\tBlocksize:\t %d\n" + \
            "\tnChannels:\t %d\n"
            "\tfrequency:\t %d [Hz] (interval: %d [ns])\n") % (
                os.getpid(), os.uname()[1],
                os.path.realpath(self.edffile.name),
                "EDF+" if self.edf_plus else "EDF",
                self.stdblocksize, len(self.channelNames),
                self.dSamplingInterval, 1000000/self.dSamplingInterval)

    @property
    def dSamplingInterval(self):
        return self._dSamplingInterval

    @property
    def stdblocksize(self):
        return self._stdblocksize

    @property
    def markerids(self):
        return self._markerids

    @property
    def channelNames(self):
        return self._channelNames[:-1] if self.edf_plus else self._channelNames

    @property
    def markerNames(self):
        return self._markerNames

    def read_edf_header(self):
        """Read edf-header information"""
        m = dict()
        m["version"] = self.edffile.read(8)
        m["subject_id"] = self.edffile.read(80).strip()
        m["recording_id"] = self.edffile.read(80).strip()
        m["start_date"] = self.edffile.read(8)
        m["start_time"] = self.edffile.read(8)
        m["num_bytes_header"] = int(self.edffile.read(8).strip())
        m["edf_c_d"] = self.edffile.read(44).strip()
        m["num_data_records"] = self.edffile.read(8)
        m["single_record_duration"] = float(self.edffile.read(8))
        m["num_channels"] = int(self.edffile.read(4))
        m["channel_names"] = list()
        for i in range(m["num_channels"]):
            m["channel_names"].append(self.edffile.read(16).strip())
        m["electrode_type"] = list()
        for i in range(m["num_channels"]):
            m["electrode_type"].append(self.edffile.read(80).strip())
        m["phy_dims"] = list()
        for i in range(m["num_channels"]):
            m["phy_dims"].append(self.edffile.read(8).strip())
        m["phy_min"] = list()
        for i in range(m["num_channels"]):
            m["phy_min"].append(float(self.edffile.read(8).strip()))
        m["phy_max"] = list()
        for i in range(m["num_channels"]):
            m["phy_max"].append(float(self.edffile.read(8).strip()))
        m["dig_min"] = list()
        for i in range(m["num_channels"]):
            m["dig_min"].append(float(self.edffile.read(8).strip()))
        m["dig_max"] = list()
        for i in range(m["num_channels"]):
            m["dig_max"].append(float(self.edffile.read(8).strip()))
        m["prefilter"] = list()
        for i in range(m["num_channels"]):
            m["prefilter"].append(self.edffile.read(80).strip())
        m["single_record_num_samples"] = list()
        for i in range(m["num_channels"]):
            m["single_record_num_samples"].append(int(self.edffile.read(8).strip()))
        m["reserved"] = self.edffile.read(32*m["num_channels"])

        # check position in file!
        assert self.edffile.tell() == m["num_bytes_header"], "EDF Header corrupt!"

        self.edf_header_length = self.edffile.tell()

        return m

    def read_edf_data(self):
        """read one record inside the data section of the edf-file"""
        edfsignal = []
        edfmarkers = numpy.ones(max(self.num_samples))*(-1)

        # get markers from self.annotations
        if self.annotations is not None:
            current_annotations = numpy.where(
                numpy.array(self.annotations.keys()) <
                self.timepoint+self.delta)[0]

            for c in current_annotations:
                tmarker = self.annotations.keys()[c]-self.timepoint
                pmarker = int((tmarker/self.delta)*max(self.num_samples))
                edfmarkers[pmarker] = self.markerids[self.annotations[self.annotations.keys()[c]]]
                self.annotations.pop(self.annotations.keys()[c])

        self.timepoint += self.delta

        # in EDF+ the last channel has the annotations,
        # otherwise it is treated as regular signal channel
        if self.edf_plus:
            for i,n in enumerate(self.num_samples):
                data = self.edffile.read(n*2)
                if len(data) != n*2:
                    raise IOError
                channel = numpy.fromstring(data, dtype=numpy.int16).astype(numpy.float32)
                signal = (channel - self.dig_min[i]) * self.gains[i] + self.phy_min[i]

                # simple upsampling for integer factors
                # TODO: may use scipy.resample ..
                if signal.shape[0] != max(self.num_samples):
                    factor = max(self.num_samples)/signal.shape[0]
                    assert type(factor) == int, str("Signal cannot be upsampled by non-int factor %f!" % factor)
                    signal = signal.repeat(factor, axis=0)
                edfsignal.append(signal)

        else:
            for i,n in enumerate(self.num_samples):
                data = self.edffile.read(n*2)
                if len(data) != n*2:
                    raise IOError
                channel = numpy.fromstring(data, dtype=numpy.int16).astype(numpy.float32)
                signal = (channel - self.dig_min[i]) * self.gains[i] + self.phy_min[i]

                # simple upsampling for integer factors
                # TODO: may use scipy.resample ..
                if signal.shape[0] != max(self.num_samples):
                    factor = max(self.num_samples)/signal.shape[0]
                    assert type(factor) == int, str("Signal cannot be upsampled by non-int factor %f!" % factor)
                    signal = signal.repeat(factor, axis=0)
                edfsignal.append(signal)

        return edfsignal, edfmarkers

    def parse_annotations(self):
        """ Parses times and names of the annotations
           This is done beforehand - annotations are later
           added to the streamed data. """

        self.edffile.seek(self.edf_header_length, os.SEEK_SET)

        self.annotations = dict()

        data_bytes_to_skip = sum(self.num_samples)*2

        while True:
            self.edffile.read(data_bytes_to_skip)

            anno = self.edffile.read(self.num_samples_anno*2)
            if len(anno) != self.num_samples_anno*2:
                break
            anno = anno.strip()

            marker = anno.split(chr(20))
            if marker[2][1:].startswith(chr(0)):
                continue

            base = float(marker[0])
            offset = float(marker[2][1:])
            name = str(marker[3])
            self.annotations[base+offset] = name.strip()

    def generate_meta_data(self):
        """ Generate the necessary meta data for the windower """
        m = self.read_edf_header()

        # calculate gain for each channel
        self.gains = [(px-pn)/(dx-dn) for px,pn,dx,dn in zip(m["phy_max"], m["phy_min"], m["dig_max"], m["dig_min"])]
        self.dig_min = m["dig_min"]
        self.phy_min = m["phy_min"]

        self._channelNames = m["channel_names"]
        self.num_channels = m["num_channels"]
        self.num_samples = m["single_record_num_samples"]

        # separate data from annotation channel
        if m["edf_c_d"] in ["EDF+D", "EDF+C"]:
            self.edf_plus = True
            # the annotation channel is called "EDF Annotations" and is the last channel
            assert "EDF Annotations" == m["channel_names"][-1], "Cannot determine Annotations Channel!"
            if m["edf_c_d"] in ["EDF+D"]:
                warnings.warn(str("The file %s contains non-continuous data-segments.\n"
                        "This feature is not supported and may lead to unwanted results!") % self.edffile.name)
            self.num_samples_anno = self.num_samples.pop() # ignore sampling rate of the annotations channel
        else :
            self.edf_plus = False

        # calculate sampling interval for each channel
        self.frequency = [ns/m["single_record_duration"] for ns in self.num_samples]
        self._dSamplingInterval = max(self.frequency)

        self._stdblocksize = max(self.num_samples)
        self.delta = self.stdblocksize / max(self.frequency)

        # generate all marker names and ids
        self._markerids['null'] = 0
        # in edf+ case we can parse them from annotations
        if self.edf_plus :
            self.parse_annotations()
            for i,(t,name) in enumerate(self.annotations.iteritems()):
                self._markerids[name] = i+1
        else:
            warnings.warn("no marker channel is set - no markers will be streamed!")
            for s in range(1,256,1):
                self.markerNames[str('S%3d' % s)] = s
            for r in range(1,256,1):
                self.markerNames[str('R%3d' % r)] = r+256

        # generate reverse mapping
        for k,v in zip(self._markerids.iterkeys(), self._markerids.itervalues()):
            self._markerNames[v] = k

        # reset file position to begin of data section
        self.edffile.seek(self.edf_header_length, os.SEEK_SET)

    # Register callback function
    def regcallback(self, func):
        self.callbacks.append(func)

    # Forwards block of data until all data is send
    def read(self, nblocks=1, verbose=False):
        """read data and call registered callbacks """
        n = 0
        while nblocks == -1 or n < nblocks:
            try:
                samples, markers = self.read_edf_data()
            except IOError:
                break
            n += 1
            for c in self.callbacks:
                c(samples, markers)

        return n

class SETReader(AbstractStreamReader):
    """ Load eeglab .set format
    
    Read eeglab format when the data has not been segmented yet. It is further
    assumed that the data is stored binary in another file with extension .fdt.
    Further possibilities are .dat format or to store everything in the .set 
    file. Both is currently not supported.
    """
    
    def __init__(self, abs_setfile_path, blocksize=100, verbose=False):
        self.abs_setfile_path = abs_setfile_path
        self._stdblocksize = blocksize
        
        self.callbacks = list()
        self._dSamplingInterval = 0
        self._markerids = {"null": 0}
        self._channelNames = dict()
        self._markerNames = {0: "null"}
        
        self.read_set_file()
        self.fdt_handle = open(self.abs_data_path,'rb')   
        self.latency = 0
        self.current_marker_index = 0
        
    @property
    def dSamplingInterval(self):
        return self._dSamplingInterval

    @property
    def stdblocksize(self):
        return self._stdblocksize

    @property
    def markerids(self):
        return self._markerids

    @property
    def channelNames(self):
        return self._channelNames

    @property
    def markerNames(self):
        return self._markerNames    
    
    def read_set_file(self):
        setdata = loadmat(self.abs_setfile_path + '.set', appendmat=False)
        # check if stream data
        ntrials = setdata['EEG']['trials'][0][0][0][0]
        assert(ntrials == 1), "Data consists of more than one trial. This is not supported!"
        # check if data is stored in fdt format
        datafilename = setdata['EEG']['data'][0][0][0]
        assert(datafilename.split('.')[-1] == 'fdt'), "Data is not in fdt format!"
        
        # collect meta information
        self._dSamplingInterval = setdata['EEG']['srate'][0][0][0][0]
        self._channelNames = numpy.hstack(setdata['EEG']['chanlocs'][0][0][ \
                                       'labels'][0]).astype(numpy.str_).tolist()
        self.nChannels = setdata['EEG']['nbchan'][0][0][0][0]
        self.marker_data = numpy.hstack(setdata['EEG']['event'][0][0][ \
                                                  'type'][0]).astype(numpy.str_)
        for marker in numpy.unique(self.marker_data):
            marker_number = len(self._markerNames)
            self._markerNames[marker_number] = marker
            self._markerids[marker] = marker_number
        self.marker_times = numpy.hstack(setdata['EEG']['event'][0][0][ \
                                                        'latency'][0]).flatten()
        self.abs_data_path = os.path.join(os.path.dirname(self.abs_setfile_path),
                                         datafilename)
   
    def regcallback(self, func):
        self.callbacks.append(func)
        
    def read(self, nblocks=1, verbose=False):
        readblocks = 0
        while (readblocks < nblocks or nblocks == -1):
            ret, samples, markers = self.read_fdt_data()
            if ret:
                for f in self.callbacks:
                    f(samples, markers)
            else:
                break
            readblocks += 1
        return readblocks
    
    def read_fdt_data(self):
        if self.fdt_handle == None:
            return False, None, None
        num_samples = self.nChannels * self._stdblocksize
        markers = numpy.zeros(self._stdblocksize)
        markers.fill(-1)
        
        ###### READ DATA FROM FILE ######
        try:
            samples = numpy.fromfile(self.fdt_handle, dtype=numpy.float32,
                                     count=num_samples)
        except MemoryError:
            # assuming, that a MemoryError only occurs when file is finished
            self.fdt_handle.close()
            self.fdt_handle = None
            return False, None, None
        
        # True when EOF reached in last or current block
        if samples.size < num_samples:
            self.fdt_handle.close()
            self.fdt_handle = None
            if samples.size == 0:
                return False, None, None
            temp = samples
            samples = numpy.zeros(num_samples)
            numpy.put(samples, range(temp.size), temp)
        
        # need channel x time matrix
        samples = samples.reshape((self.stdblocksize, self.nChannels)).T

        ###### READ MARKERS FROM FILE ######
        for l in range(self.current_marker_index,len(self.marker_times)):
            if self.marker_times[l] > self.latency + self._stdblocksize:
                self.current_marker_index = l
                self.latency += self._stdblocksize
                break
            else:
                rel_marker_pos = (self.marker_times[l] - 1) % self._stdblocksize
                markers[rel_marker_pos] = self._markerids[self.marker_data[l]]

        return True, samples, markers
    
class EEGReader(AbstractStreamReader):

    """ Load raw EEG data in the .eeg brain products format

    This module does the Task of parsing
    .vhdr, .vmrk end .eeg/.dat files and then hand them
    over to the corresponding windower which
    iterates over the aggregated data.
    """

    def __init__(self, abs_eegfile_path, blocksize=100, verbose=False):
        self.abs_eegfile_path = abs_eegfile_path
        self._stdblocksize = blocksize

        self.eeg_handle = None
        self.mrk_handle = None
        self.eeg_dtype = numpy.int16

        self.callbacks = list()

        # variable names with capitalization correspond to
        # structures members defined in RecorderRDA.h
        self.nChannels, \
        self._dSamplingInterval, \
        self.resolutions, \
        self._channelNames, \
        self.channelids, \
        self._markerids, \
        self._markerNames, \
        self.nmarkertypes = self.bp_meta()

        if verbose:
            print "channelNames:", self.channelNames, "\n"
            print "channelids:", self.channelids, "\n"
            print "markerNames:", self.markerNames, "\n"
            print "markerids:", self.markerids, "\n"
            print "resolutions:", self.resolutions, "\n"

        # open the eeg-file
        if self.eeg_handle == None:
            try:
                self.eeg_handle = open(self.abs_eegfile_path + '.eeg', 'rb')
            except IOError:
                try:
                    self.eeg_handle = open(self.abs_eegfile_path + '.dat', 'rb')
                except IOError:
                    raise IOError, "EEG-file [%s.{dat,eeg}] could not be opened!" % os.path.realpath(self.abs_eegfile_path)

        self.callbacks = list()

        self.ndsamples = None           # last sample block read
        self.ndmarkers = None         # last marker block read


    @property
    def dSamplingInterval(self):
        return self._dSamplingInterval

    @property
    def stdblocksize(self):
        return self._stdblocksize

    @property
    def markerids(self):
        return self._markerids

    @property
    def channelNames(self):
        return self._channelNames

    @property
    def markerNames(self):
        return self._markerNames

    # This function gathers meta information from the .vhdr and .vmrk files.
    # Only the relevant information is then stored in variables, the windower
    # accesses during the initialisation phase.
    def bp_meta(self):
        nChannels = 0
        dSamplingInterval = 0
        resolutions = list()
        channelNames = list()
        channelids = dict()
        markerids = dict()
        markerNames = dict()
        nmarkertypes = 0

        prefix = ''

        # helper function to convert resolutions
        # 0 = 100 nV, 1 = 500 nV, 2 = 10 {mu}V, 3 = 152.6 {mu}V
        def res_conv(num, res):
            # convert num to nV
            if ord(res[0]) == 194:
                num = num*1000

            if num <= 100: return 0
            if num <= 500: return 1
            if num <= 10000: return 2
            return 3

        # Start with vhdr file
        file_path = self.abs_eegfile_path + '.vhdr' 
        hdr = open(file_path)
        for line in hdr:
            if line.startswith(";"): continue

            # Read the words between brackets like "[Common Infos]"
            if line.startswith('['):
                prefix = line.partition("[")[2].partition("]")[0].lower()
                continue

            if line.find("=") == -1: continue

            # Common Infos and Binary Infos
            if(prefix == 'common infos' or prefix == 'binary infos'):
                key, value = line.split('=')
                key = key.lower()
                value = value.lower()
                if(key == 'datafile'):
                    pass # something like filename.eeg
                elif(key == 'markerfile'):
                    mrk_file = value
                elif(key == 'dataformat'):
                    pass # usually BINARY
                elif(key == 'dataorientation'):
                    eeg_data_or = value
                elif(key == 'datatype'):
                    pass # something like TIMEDOMAIN
                elif(key == 'numberofchannels'):
                    nChannels = int(value)
                elif(key == 'datapoints'):
                    pass # the number of datapoints in the whole set
                elif(key == 'samplinginterval'):
                    dSamplingInterval = int(1000000/float(value))
                elif(key == 'binaryformat'):
                    if re.match("int_16", value, flags=re.IGNORECASE) == None:
                        self.eeg_dtype = numpy.float32
                    else:
                        self.eeg_dtype = numpy.int16
                elif(key == 'usebigendianorder'):
                    bin_byteorder = value

            # Channel Infos
            # ; Each entry: Ch<Channel number>=<Name>,<Reference channel name>,
            # ; <Resolution in "Unit">,<Unit>,
            elif(prefix == 'channel infos'):
                key, value = line.split('=')
                if re.match("^[a-z]{2}[0-9]{1,3}", key, flags=re.IGNORECASE) == None:
                    continue
                ch_id = int(re.findall(r'\d+', key)[0])
                ch_name = value.split(',')[0]
                ch_ref = value.split(',')[1]

                if len(re.findall(r'\d+', value.split(',')[2])) == 0:
                    ch_res_f = 0
                else:
                    ch_res_f = float(re.findall(r'\d+', value.split(',')[2])[0])

                ch_res_unit = value.split(',')[3]

                channelNames.append(ch_name)
                channelids[ch_name] = ch_id
                resolutions.append(res_conv(ch_res_f, ch_res_unit))

                # Everything thats left..
            else:
                #print "parsing finished!"
                break
        hdr.close()

        # Continue with marker file
        # Priority:
        #   1: Path from .vhdr
        #   2: Path constructed from eegfile path
        prefix = ''
        markerNames[0] = 'null'
        try:
            self.mrk_handle = open(os.path.basename(self.abs_eegfile_path) + mrk_file)
        except IOError:
            try:
                self.mrk_handle = open(self.abs_eegfile_path + '.vmrk')
            except IOError:
                raise IOError, str("Could not open [%s.vmrk]!" % os.path.realpath(self.abs_eegfile_path))

        # Parse file
        for line in self.mrk_handle:
            if line.startswith(";"): continue

            # Read the words between brackets like "[Common Infos]"
            if line.startswith('['):
                prefix = line.partition("[")[2].partition("]")[0].lower()
                continue

            if line.find("=") == -1: continue

            if prefix == "marker infos":
                mrk_name = line.split(',')[1]
                if mrk_name != "" and mrk_name not in markerNames.values():
                    markerNames[len(markerNames)] = mrk_name

        # rewinds the marker file
        self.mrk_handle.seek(0, os.SEEK_SET)

        # helper struct for finding markers
        self.mrk_info = dict()
        self.mrk_info['line'] = ""
        self.mrk_info['position'] = 0
        # advance to first marker line
        while(re.match("^Mk1=", self.mrk_info['line'], re.IGNORECASE) == None):
            try:
                self.mrk_info['line'] = self.mrk_handle.next()
            except StopIteration:
                self.mrk_handle.close()
                raise StopIteration, str("Reached EOF while searching for first Marker in [%s]" % os.path.realpath(self.mrk_handle.name))

        # TODO: Sort markerNames?
        def compare (x,y):
            return cmp(self.int(re.findall(r'\d+', x)[0]), int(re.findall(r'\d+', y)[0]))

        for key in markerNames:
            markerids[markerNames[key]] = key

        markertypes = len(markerids)


        return nChannels, \
               dSamplingInterval, \
               resolutions, \
               channelNames, \
               channelids, \
               markerids, \
               markerNames, \
               markertypes

    # This function reads the eeg-file and the marker-file for every
    # block of data which is processed.
    def bp_read(self, verbose=False):

        if self.eeg_handle == None:
            return False, None, None

        num_samples = self.nChannels*self.stdblocksize
        markers = numpy.zeros(self.stdblocksize)
        markers.fill(-1)
        samples = numpy.zeros(num_samples)
        
        ###### READ EEG-DATA FROM FILE ######
        try:
            samples = numpy.fromfile(self.eeg_handle, dtype=self.eeg_dtype, count=num_samples)
        except MemoryError:
            # assuming, that a MemoryError only occurs when file is finished
            self.eeg_handle.close()
            self.eeg_handle = None
            return False, None, None

        # True when EEG-File's EOF reached in last or current block
        if samples.size < num_samples:
            self.eeg_handle.close()
            self.eeg_handle = None
            if samples.size == 0:
                return False, None, None
            temp = samples
            samples = numpy.zeros(num_samples)
            numpy.put(samples, range(temp.size), temp)

        samples = samples.reshape((self.stdblocksize, self.nChannels))
        samples = scipy.transpose(samples)

        ###### READ MARKERS FROM FILE ######
        self.mrk_info['position'] += self.stdblocksize
        mk_posi = 0
        mk_desc = ""
        while True:
            mk = self.mrk_info['line'].split(',')
            if len(mk) < 2 or mk[1] == "":
                try:
                    self.mrk_info['line'] = self.mrk_handle.next()
                except:
                    self.mrk_handle.close()
                    #self._log("WARNING: EOF[%s]\n" % os.path.realpath(self.mrk_handle.name))
                    break
                continue
            mk_desc = mk[1]
            mk_posi = int(mk[2])
            if mk_posi > self.mrk_info['position']:
                break

            # special treatment for 'malformed' markerfiles
            mk_rel_position = (mk_posi-1) % self.stdblocksize
            if markers[mk_rel_position] != -1 :
                # store marker for next point
                mk[2] = str(mk_posi+1)
                self.mrk_info['line'] = ",".join(["%s" % (m) for m in mk])
                #self._log(str("WARNING: shifted position of marker \"%s\" from %d to %d!\n" % (mk_desc, mk_posi, mk_posi+1)))
                if mk_rel_position+1 > self.stdblocksize-1:
                    return True, samples, markers
                else:
                    continue

            else :
                markers[mk_rel_position] = self.markerids[mk_desc]
                self.mrk_info['line'] = ""

            # try to read next line from markerfile
            try:
                self.mrk_info['line'] = self.mrk_handle.next()
            except:
                self.mrk_handle.close()
                break

        return True, samples, markers

    # string representation with interesting information
    def __str__(self):
        return ("EEGReader Object (%d@%s)\n" + \
                "\tEEG File:\t %s\n" + \
                "\tMRK File:\t %s\n" + \
                "\tFile Format:\t %s\n" + \
                "\tBlocksize:\t %d\n" + \
                "\tnChannels:\t %d\n") % (os.getpid(), os.uname()[1], os.path.realpath(self.eeg_handle.name),
                                          os.path.realpath(self.mrk_handle.name), self.eeg_dtype,
                                          self.stdblocksize, self.nChannels)

    # Register callback function
    def regcallback(self, func):
        self.callbacks.append(func)

    # Reads data from .eeg/.dat file until EOF
    def read(self, nblocks=1, verbose=False):

        self.stop = False
        readblocks = 0
        while (readblocks < nblocks or nblocks == -1):
            ret, self.ndsamples, self.ndmarkers = self.bp_read()
            if ret:
                for f in self.callbacks:
                    f(self.ndsamples, self.ndmarkers)
            else:
                break
            readblocks += 1
        return readblocks
