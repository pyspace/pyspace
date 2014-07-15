""" Base Module for datasets to specify the interface for these """

import yaml
import os
import logging
import logging.handlers
import warnings
import socket
import cPickle
from pySPACE.run.scripts import md_creator
#import bz2
from collections import defaultdict

class UnknownDatasetTypeException(Exception):
    """ Wrapper around error, when dataset type is not available """
    pass

class BaseDataset(object):
    """ Base class for datasets
    
    This class (BaseDataset)  acts as base class for all dataset classes
    and specifies the interface for these. Furthermore it provides a factory 
    method *load* for all types of datasets. It expects a path to the datasets
    storage directory.

    The following methods must be implemented:

        :__init__:  The constructor must take an argument *dataset_md* that
                  is a dictionary containing meta data for the dataset
                  to be loaded.
        :store:  A method that stores a dataset in a certain directory.
               *store* and *__init__* should be written so that *__init__* can
               correctly recreate every dataset stored with *store*
        :add_sample: (*optional*)  Adds a new sample to the dataset.
                    BaseDataset provides a default implementation.

    Datasets store the data in the attribute *self.data*.
    This data is stored as a dictionary that maps (run, split, train/test)
    tuples to the actual data obtained in this split in this run for
    training/testing.
    """
    
    def __init__(self, dataset_md = None):
        # The data structure containing the actual data
        # The data is stored as a dictionary that maps
        # (run, split, train/test) tuples to the actual 
        # data obtained in this split in this run for
        # training/testing.
        self.data = defaultdict(list)
        
        # A dictionary containing some meta data for the respective
        # dataset type.
        self.meta_data = {"train_test": False,  # defaults
                          "splits": 1,
                          "runs": 1} 
        if not dataset_md is None:
            self.update_meta_data(dataset_md)

    @classmethod
    def load(cls, dataset_dir):
        """ Loads the dataset stored in directory *rel_dataset_dir*
        
        This method loads the dataset stored in the directory
        *rel_dataset_dir* . Depending on the type stored in the datasets
        meta-data file, the method creates an instance of a specific
        dataset class.
        
        The method expects the following parameters:
          * *dataset_dir* : The (absolute) directory in which the dataset \
                               that will be loaded is located   
        """
        # Loading the dataset meta data
        meta_data = cls.load_meta_data(dataset_dir)
        # Set the directory where this dataset is located
        meta_data["dataset_directory"] = dataset_dir

        # Mapping for Backward Compatibility
        if meta_data["type"].lower() == "raw_eeg":
            meta_data["type"] = "STREAM"
            meta_data["storage_format"] = "bp_eeg"

        # construct dataset module and class name dependent on the type
        # for backward compatibility type is casted to lower-case
        data_mod_name = meta_data["type"].lower()
        data_class_name = ''.join([x.title()
                                   for x in meta_data["type"].split('_')])
        data_class_name += "Dataset"
        # dynamic class import: from data_mod_name import col_class_name
        try:
            dataset_module = __import__(
                'pySPACE.resources.dataset_defs.%s' % data_mod_name,
                fromlist=[data_class_name])
        except:
            msg = "Dataset type %s in %s is unknown" % \
                (meta_data["type"], meta_data["dataset_directory"])
            raise UnknownDatasetTypeException(msg)
        dataset_class = getattr(dataset_module, data_class_name)
        # delegate to subclass
        return dataset_class(dataset_md=meta_data,
                             dataset_dir=dataset_dir)

    @staticmethod
    def load_meta_data(dataset_dir, file_name="metadata.yaml"):
        """ Load the meta data of the dataset """
        try:
            file_path = os.sep.join([dataset_dir,file_name])
            meta_file = open(file_path,'r')
        except IOError:
            pass
        else:
            meta_data = yaml.load(meta_file)
            if "ignored_columns" in meta_data:
                meta_data["ignored_columns"] = \
                    md_creator.parse_list(meta_data["ignored_columns"])
            if meta_data.has_key("ignored_rows"):
                meta_data["ignored_rows"] = \
                    md_creator.parse_list(meta_data["ignored_rows"])
            meta_file.close()
            return meta_data
        # Error handling and backward compatibility
        try:
            file_path = os.sep.join([dataset_dir, "collection.yaml"])
            meta_file = open(file_path,'r')
            meta_data = yaml.load(meta_file)
            if meta_data.has_key("ignored_columns"):
                meta_data["ignored_columns"] = \
                    md_creator.parse_list(meta_data["ignored_columns"])
            if meta_data.has_key("ignored_rows"):
                meta_data["ignored_rows"] = \
                    md_creator.parse_list(meta_data["ignored_rows"])
            meta_file.close()
            warnings.warn(
                "'collection.yaml' needs to be renamed to 'metadata.yaml'!")
            return meta_data
        except IOError, e:
            warnings.warn("IOError occurred: %s." % e)
            # check if we have a feature vector dataset with missing metadata.yaml
            csv_file = None
            for dirpath, dirnames,files in os.walk(dataset_dir):
                for file in files:
                    if file.endswith(".csv") or file.endswith(".arff"):
                        csv_file = file
                        break
                if csv_file:
                    break
            if csv_file:
                warnings.warn(
                    "If you want to use csv-files, you have to " +
                    "generate a %s! The pySPACE documentation " % file_name +
                    "tells you what you have to specify. You can also use " +
                    ":script:`pySPACE.run.scripts.md_creator.py`. " +
                    "We will try this in the following...")
                print("Found '%s' at '%s'!" % (csv_file, dirpath))
                if not dirpath==dataset_dir:
                    print("Maybe you specified the wrong input_path?")
                md_file = dirpath+os.sep+file_name
                if not os.path.isfile(md_file):
                    md_creator.main(md_file)
                    collection_meta_file=open(md_file)
                    meta_data = yaml.load(collection_meta_file)
                    collection_meta_file.close()
                    return meta_data
            raise Exception("No pySPACE dataset '%s' found. " % dataset_dir +
                            "You have to specify a %s in each " % file_name +
                            "dataset directory. Have a look at the pySPACE "
                            "documentation. Continuing...")
    
    @staticmethod
    def store_meta_data(dataset_dir, meta_data, file_name="metadata.yaml"):
        """ Stores the meta data of a dataset """
        # Loading the dataset meta file
        try:
            collection_meta_file = open(os.sep.join([dataset_dir, file_name]),
                                        'w')
        except IOError: 
            raise Exception("No pySPACE dataset %s found. Continuing..."
                                % dataset_dir)
        yaml.dump(meta_data, collection_meta_file)
        collection_meta_file.close()

    def add_sample(self, sample, label, train, split=0, run=0):
        """ Add a sample to this dataset
        
        Adds the sample *sample* along with its class label *label*
        to this dataset.
        
        The method expects the following parameters:
          * *sample* : The respective data sample
          * *label* : The label of the data sample
          * *train* : If *train*, this sample has already been used for training
          * *split* : The number of the split this sample belongs to. \
                     Defaults to 0.   
          * *run*: The run number this sample belongs to Defaults to 0
        
        """
        if train == "test":
            train = False
        if train:
            self.meta_data["train_test"] = True
        if split + 1 > self.meta_data["splits"]: 
            self.meta_data["splits"] = split + 1

        key = (run, split, "train" if train else "test")

        if isinstance(self.data[key], basestring):
            self.data[key] = []
        self.data[key].append((sample, label))

    def update_meta_data(self, meta_data):
        """ Updates the internal meta_data dictionary with *meta_data* """
        self.meta_data.update(meta_data)
    
    def get_run_numbers(self):
        """ Return the number of the runs contained in this dataset """
        runs = set(run for run, split, train_test in self.data.keys())
        return list(runs)
    
    def get_split_numbers(self, current_run=0):
        """ Return the number of the splits
        
        Returns the number of splits contained in this dataset
        for the given run number *current_number* """
        splits = set(split for run, split, train_test in self.data.keys()
                     if run == current_run)
        return list(splits)
    
    def dump(self, result_path, name):
        """ Dumps this dataset into a file.
        
        Dumps (i.e. pickle) this dataset object into a bz2 compressed file.
        In contrast to *store* this method stores the whole dataset
        in a file. No meta data are stored in a YAML file etc.
        
        The method expects the following parameters:
         * *result_path* The path to the directory in which the pickle \
                         file will be written.
         * *name* The name of the pickle file
        """  
        result_file = open(os.path.join(result_path, 
                                        name + ".pickle"), "wb")
        #result_file.write(bz2.compress(cPickle.dumps(self, protocol=2)))
        result_file.write(cPickle.dumps(self, protocol=2))
        result_file.close()
    
    def store(self, result_dir, s_format=None):
        """ Stores this dataset in the directory *result_dir*.
        
        In contrast to *dump* this method stores the dataset
        not in a single file but as a whole directory structure with meta
        information etc. The data sets are stored separately for each run, 
        split, train/test combination.
        
        The method expects the following parameters:
          * *result_dir* The directory in which the dataset will be stored
          * *s_format* The format in which the actual data sets should be stored.
        
        .. note:: Needs to be overwritten by the subclasses!
        """
        raise NotImplementedError()

    def _log(self, message, level=logging.INFO):
        """ Logs  the given message  with the given logging level """
        root_logger = logging.getLogger("%s-%s.%s" % (socket.gethostname(),
                                        os.getpid(),
                                        self))
        if len(root_logger.handlers) == 0:
            root_logger.addHandler(logging.handlers.SocketHandler('localhost',
                    logging.handlers.DEFAULT_TCP_LOGGING_PORT))

        root_logger.log(level, message)

    def __repr__(self):
        """ Return a string representation of this class"""
        return self.__class__.__name__
