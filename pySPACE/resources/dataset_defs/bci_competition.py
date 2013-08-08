""" Read data from the BCI competition

Storing is not needed, because the data will be transformed to another collection.

Currently, only the use of the "train" datasets of the BCI Competition III,
Data set II: P300 speller paradigm is supported. 
See http://www.bbci.de/competition/iii/ for further information.

    **Parameters**
    
      :dataset_md:
          A dictionary with all the collections's meta data.
          
          (*optional, default: None*)
"""

from pySPACE.resources.dataset_defs.base import BaseDataset

import numpy
import scipy.io
import os
from pySPACE.resources.data_types.time_series import TimeSeries

class BciCompetitionDataset(BaseDataset):
    """ Class for reading the Berlin BrainComputerInterface-competition data
    
    This module contains a class (*BciCompetitionDataset*) that encapsulates
    most relevant code to use the data from the BCI competition.
    Currently, only reading of the data is supported.
    """
    
    def __init__(self, dataset_md=None, **kwargs):
        super(BciCompetitionDataset, self).__init__(dataset_md=dataset_md)
        self.data_directory = dataset_md["dataset_directory"]
        self.file_path = dataset_md["mat_file"]
        self.dataset_md = dataset_md
    
    def store(self, result_dir):
        """ Not yet implemented!"""
        raise NotImplementedError("Storing of BciCompetitionDataset is currently not supported!")
    
    
    def get_data(self, run_nr, split_nr, train_test):
        """ Return the train or test data for the given split in the given run.
        
        **Parameters**
          
          :run_nr: The number of the run whose data should be loaded.
          
          :split_nr: The number of the split whose data should be loaded.
          
          :train_test: "train" if the training data should be loaded.
                       "test" if the test data should be loaded.
    
        """
        # Do lazy loading of the time series objects.
        filepath = self.data_directory + os.path.sep + self.file_path
        data = scipy.io.loadmat(filepath)
        signal = data['Signal']
        flashing = data['Flashing']
        stimulus_code = data['StimulusCode']
        stimulus_type = data['StimulusType']
        target_char = data['TargetChar']

        window = 240
        channels = 64
        epochs = signal.shape[0]
        data_collection = []

        responses = numpy.zeros((12, 15, window, channels))
        for epoch in range(epochs):
            counter = 0
            rowcolcnt=numpy.ones(12)
            for n in range(1, signal.shape[1]):
                if (flashing[epoch,n]==0 and flashing[epoch,n-1]==1):
                    rowcol=stimulus_code[epoch,n-1]
                    responses[rowcol-1,rowcolcnt[rowcol-1]-1,:,:]=signal[epoch,n-24:n+window-24,:]
                    rowcolcnt[rowcol-1]=rowcolcnt[rowcol-1]+1

            avgresp=numpy.mean(responses,1)

            targets = stimulus_code[epoch,:]*stimulus_type[epoch,:]
            target_rowcol = []
            for value in targets:
                if value not in target_rowcol:
                    target_rowcol.append(value)

            target_rowcol.sort()

            for i in range(avgresp.shape[0]):
                temp = avgresp[i,:,:]
                data = TimeSeries(input_array = temp,
                                  channel_names = range(64), 
                                  sampling_frequency = window)
                if i == target_rowcol[1]-1 or i == target_rowcol[2]-1:
                    data_collection.append((data,"Target"))
                else:
                    data_collection.append((data,"Standard"))

        return data_collection
