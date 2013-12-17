""" Load and store data sets containing :mod:`Feature Vectors<pySPACE.resources.data_types.feature_vector>`

"""

import os
import cPickle
import yaml
import pwd
import numpy
import warnings

import logging

from pySPACE.resources.dataset_defs.base import BaseDataset
from pySPACE.resources.data_types.feature_vector import FeatureVector


class FeatureVectorDataset(BaseDataset):
    """ Feature vector dataset class
    
    This class is most importantly
    for loading and storing
    :class:`~pySPACE.resources.data_types.feature_vector.FeatureVector`
    to the file system.
    You can load it using a :mod:`~pySPACE.missions.nodes.source.feature_vector_source` node.
    It can be saved, using a :mod:`~pySPACE.missions.nodes.sink.feature_vector_sink` node
    in a :class:`~pySPACE.missions.operations.node_chain.NodeChainOperation`
    
    The constructor expects the argument *dataset_md* that
    contains a dictionary with all the meta data.
    It is normally loaded from the metadata.yaml file.
    
    It is able to load csv-Files, arff-files and pickle files,
    where one file is always responsible for one training or test set.
    The name conventions are the same as described in
    :class:`~pySPACE.resources.dataset_defs.time_series.TimeSeriesDataset`.
    It is important that a metadata.yaml file exists, giving all
    the relevant information of the data set,
    especially the storage format, which can be
    *pickle*, *arff*, *csv* or *csvUnnamed*.
    The last format is only for loading data without heading,
    and with the labels being not in the last column.
    
    **pickle-files**
    
    See :class:`~pySPACE.resources.dataset_defs.time_series.TimeSeriesDataset`
    for name conventions (in the tutorial).
    
    
    **arff-files**
    
    http://weka.wikispaces.com/ARFF
    
    This format was introduced to connect pySPACE with weka.
    So when using weka, you need to choose this file format as the parameter
    storage format in the preprocessing operation's spec file.
    
    
    **CSV (comma separated values)**
    
    These are tables in a simple text format.
    Therefore each column is separated with a comma and each row with a new line.
    Normally the first line gives the feature names and one row is giving
    the class labels. Therefore several parameters need to be specified in
    the metadata.yaml file.
    
    If no collection meta data is available for the input data, 
    the 'metadata.yaml' file can be generated with
    :mod:`~pySPACE.run.scripts.md_creator`.
    Please consider also some important parameters, described in the 
    get_data function.
    
    Preferably the labels are in the last column. This corresponds to
    *label_column* being -1 in the metadata.yaml file.
    
    **Parameters**
    
        :dataset_md: dictionary containing meta data for the collection
                        to be loaded

        The following  3 Parameters contain standard information
        for a feature vector data set. 
        Normally they are not yet needed (used), because a dataset_md
        is given and real data is loaded,
        and so this information could be loaded from the data.
        Nevertheless these are important entries, which should be found
        in each dataset_md, giving information about the data set.

        :classes_names: list of the used class labels
        
        :feature_names: list of the feature names
        
        :num_features: number of the given features
    
    .. todo:: Better integration and documentation of the data_pattern variable, 
              e.g. when reading arff files.
    
    **Special CSV Parameters**
    
        :label_column:
            Column containing the labels 
            
            Normally this column looses its heading.
            when saving the csv file, the default, -1, is used.
            
            (*recommended, default: -1*)
        
        :ignored_columns:
            List of numbers containing the numbers of irrelevant columns,
            e.g., `[1,2,8,42]`

            After the data is loaded, this parameter becomes obsolete.
            
            .. todo:: Enable and document eval syntax
            
            (*optional, default: []*)
        
        :ignored_rows:
            Replace row in description of 'ignored_columns'
            
            (*optional, default: []*)
        
        :delimiter:
            Symbol which separates the csv entries
            
            Typically `,` is used or the tabulator `\t`.
            When storing, `,` is used.
            
            (*recommended, default: ','*)
    """
    def __init__(self, dataset_md=None, classes_names=[], feature_names=None,
                 num_features = None, **kwargs):
        """ Read out the data from the given collection
        
        .. todo:: test data pattern usage on old data
        
        .. note:: main loading concept copied from time series collection
                  check needed if code can be sent to upper class
        """
        super(FeatureVectorDataset, self).__init__(dataset_md=dataset_md)
        if not self.meta_data.has_key("feature_names"):
            self.update_meta_data({"feature_names":feature_names})
        if not self.meta_data.has_key("classes_names"):
            self.update_meta_data({"classes_names":classes_names})
        if not self.meta_data.has_key("num_features"):
            self.update_meta_data({"num_features":num_features})
        if not dataset_md is None: #data has to be loaded
            self._log("Loading feature vectors from input collection.")
            dataset_dir = self.meta_data["dataset_directory"]
            s_format = self.meta_data["storage_format"]
            if type(s_format) == list:
                s_format = s_format[0]
            # mainly code copy from time series data set defs
            if dataset_md.has_key("data_pattern") and not self.meta_data["train_test"] \
                   and self.meta_data["splits"] == 1 \
                   and self.meta_data["runs"] == 1 :
                # The collection consists only of a single set of data, for
                # one run, one splitting, and only test data
                data = dataset_md["data_pattern"].replace("_run", "_run0") \
                                                     .replace("_sp","_sp0") \
                                                     .replace("_tt","_test")
                # File that contains the time series objects
                fv_file = os.path.join(dataset_dir,data)
                
                # Actual data will be loaded lazily
                self.data[(0, 0, "test")] = fv_file
            elif dataset_md.has_key("data_pattern"):
                for run_nr in range(self.meta_data["runs"]):
                    for split_nr in range(self.meta_data["splits"]):
                        for train_test in ["train", "test"]:
                            # The collection consists only of a single set of data, for
                            # one run, one splitting, and only test data
                            data = dataset_md["data_pattern"].replace("_run", "_run%s" % run_nr) \
                                                                 .replace("_sp","_sp%s" % split_nr) \
                                                                 .replace("_tt","_%s" % train_test)
                            # File that contains the time series objects
                            fv_file = os.path.join(dataset_dir,data)
                            # Actual data will be loaded lazily
                            self.data[(run_nr, split_nr, train_test)] = fv_file
            elif dataset_md.has_key("file_name"):
                fv_file = os.path.join(dataset_dir,self.meta_data["file_name"])
                self.data[(0, 0, "test")] = fv_file
            else:
                pass
                ##TODO: What should we do? - Raise Error, because data is not defined?
                #raise NotImplementedError()
        else: #dataset_md == None # called when storing or initialize empty collection
            # We create a new, empty collection
            pass

    def add_sample(self, sample, label, train, split=0, run=0):
        """ Add a sample to this collection
        
        Adds the sample *sample* along with its class label *label*
        to this collection.
        
        **Parameters**
        
            :sample: The respective data sample
            :label:  The label of the data sample
            :train:  If *train*, this sample has already been used for training
            :split:  The number of the split this sample belongs to. \
                       
                     (*optional, default: 0*)
                       
            :run:    The run number this sample belongs to
          
                     (*optional, default: 0*)
        """
        if self.meta_data["num_features"] is None:
            self.update_meta_data({"num_features": sample.size})
        elif not sample.size == self.meta_data["num_features"]:
            self.update_meta_data({"num_features": sample.size})
            warnings.warn("Mismatching feature number: %i given but %i occured."
                          % (self.meta_data["num_features"], sample.size))
        try:
            # Remember all class labels since these will be stored in the arff file
            if label not in self.meta_data["classes_names"]:
                self.meta_data["classes_names"].append(label)
        except KeyError:
            self.update_meta_data({"classes_names": [label]})
        # Delegate to super class
        super(FeatureVectorDataset, self).add_sample(sample, label,
                                                     train, split, run)

    def dump(self, result_path, name):
        """ Dumps this collection into a file.
        
        Dumps (i.e. pickle) this collection object into a bz2 compressed file.
        In contrast to *store* this method stores the whole collection
        in a file. No meta data are stored in a YAML file etc.
        
        The method expects the following parameters:
         * *result_path* The path to the directory in which the pickle \
                         file will be written.
         * *name* The name of the pickle file
         
        """
        # Remove the feature names from the feature vectors since this leads to
        # unnecessary large sizes on the disk
        for values in self.data.itervalues():
            for (sample, label) in values:
                sample.feature_names = []
        
        # Delegate to super class
        super(FeatureVectorDataset, self).dump(result_path, name)

    def get_data(self, run_nr, split_nr, train_test): # feature_file, storage_format):
        """ Loads the data from the feature file of the current input collection 
        depending on the storage_format.
        Separates the actual vectors from the names and returns both as lists.
        
        The method expects the following
        
        **Parameters**
        
            :feature_file:      the file of feature vectors to be loaded
            :storage_format:    One of the first components in
                                ['arff', 'real'], ['csv', 'real'], 
                                ['csvUnnamed', 'real'] or .
                                Format in which the feature_file was saved.
                                Information need to be present in meta data.
        
        For arff and pickle files documentation see to the class description
        (docstring). Pickle format files do not need any special 
        loading because they
        already have the perfect format.
        
        **CSV**
        
        If no collection meta data is available for the input data, 
        the 'metadata.yaml' file can be generated with
        :mod:`pySPACE.run.node_chain_scripts.md_creator`.
        
        If you created the csv file with pySPACE, you automatically have the
        standard *csv* format with the feature names in the first row
        and the labels in the last column.
        
        If you have a csv tabular without headings,
        you have the *csvUnnamed* format,
        and in your 'label_column' column, specified in your spec file,
        the labels can be found.
        
        .. note:: main loading concept copied from time series collection
          check needed if code can be sent to upper class
        """
        ## init ##
        classes_names = self.meta_data["classes_names"]
        s_format = self.meta_data["storage_format"]
        if type(s_format) == list:
            s_format = s_format[0]

        # todo: automatical loading of csv?
        delimiter = self.meta_data.get("delimiter", ",")
        if not len(delimiter) == 1:
            self._log("Wrong delimiter ('%s') given. Using default ','." %
                      delimiter, level=logging.CRITICAL)
            delimiter = ","

        # Do lazy loading of the fv objects.
        if isinstance(self.data[(run_nr, split_nr, train_test)], basestring):
            self._log("Lazy loading of %s feature vectors from input "
                      "collection for run %s, split %s." % (train_test, run_nr, 
                                                            split_nr))
            if s_format == "pickle":
                # Load the data from a pickled file
                file = open(self.data[(run_nr, split_nr, train_test)], "r")
                self.data[(run_nr, split_nr, train_test)] = cPickle.load(file)
                file.close()
                sample = self.data[(run_nr, split_nr, train_test)][0][0]
                self.update_meta_data({"feature_names":sample.feature_names,
                                       "len_line":len(sample.feature_names)})
            elif s_format == "arff":
                names = []
                data = []
                # load file
                f = open(self.data[(run_nr, split_nr, train_test)])
                data_set = f.readlines()
                f.close()
                # Read the arff file completely ##
                for line in data_set:
                    if "@attribute class" in line \
                            or "@relation" in line \
                            or "@data" in line:
                        pass
                    elif "@attribute" in line:
                        name_line = line.split()
                        names.append(name_line[1])
                    else: 
                        data.append(line.split(delimiter))
                # the label is expected to be at the end 
                # of each line in the data.
                for line in data:
                    vector = line[0:-1]
                    label = line[-1].rstrip("\n\r ")  # --> label is string
                    if not label in classes_names:
                        classes_names.append(label)
                    sample = FeatureVector(numpy.atleast_2d([vector]).astype(
                        numpy.float64), feature_names=names)
                    self.add_sample(sample=sample, label=label, 
                                    train=train_test, split=split_nr, 
                                    run=run_nr)
                self.update_meta_data({"feature_names": sample.feature_names,
                       "len_line": len(sample.feature_names),
                       "classes_names": classes_names})
            elif "csv" in s_format: # csv or csv unnamed
                # load file
                f = open(self.data[(run_nr, split_nr, train_test)])
                data_set = f.readlines()
                f.close()
                # getting rid of all unwanted rows
                if "ignored_rows" in self.meta_data:
                    ignored_rows = self.meta_data["ignored_rows"]
                    if not type(ignored_rows) == list:
                        warnings.warn("Wrong format: Ignored rows included!")
                        ignored_rows = []
                    ignored_rows.sort()
                    remove_list = []
                    for i in ignored_rows:
                        remove_list.append(data_set[int(i)-1])
                    for j in remove_list:
                        data_set.remove(j)
                # get len_line and delete heading
                feature_names = self.meta_data["feature_names"]
                if s_format == "csv":
                    names = data_set[0].rstrip(",\n").split(delimiter)
                    data_set.pop(0)
                len_line = len(data_set[0].split(delimiter))

                # get and prepare label column numbers (len_line needed)
                try:
                    label_column = self.meta_data["label_column"]
                except KeyError:
                    label_column = -1
                # map column numbers to indices by subtracting -1
                if type(label_column) == int:
                    label_columns = [label_column - 1]
                elif type(label_column) == list:
                    label_columns = [int(l)-1 for l in label_column]
                for i in range(len(label_columns)):
                    # map to positive value and undo previous offset
                    if label_columns[i] < 0:
                        label_columns[i] = label_columns[i] + len_line+1
                    if label_columns[i] < 0:
                        label_columns[i] = -1 + len_line
                # very important sorting for index shifts
                label_columns.sort()
                # calculate unwanted columns
                # note: These indices begin with 1 .
                # They are internally shifted when used.
                if self.meta_data.has_key("ignored_columns"):
                    ignored_columns = self.meta_data["ignored_columns"]
                    if not type(ignored_columns) == list:
                        warnings.warn("Wrong format: Ignored columns included!")
                        ignored_columns = []
                    new_ignored_columns = []
                    for i in ignored_columns:
                        i =  int(i)
                        if i < 0:
                            i += len_line
                        for label_column in label_columns:
                            if i > label_column:
                                i -= 1
                        new_ignored_columns.append(i)
                else:
                    new_ignored_columns = []
                new_ignored_columns.sort()
                # get all relevant feature_names
                if feature_names is None:
                    if s_format == "csv":
                        # delete blanks and inverted commas 
                        for i in range(len(names)):
                            names[i] = names[i].strip(' "')
                            names[i] = names[i].strip(" '")
                        feature_names = names
                    else: #s_format=="csv_unnamed"
                        feature_names = ["feature_%s" % i for i in
                                         range(len_line)]
                    # switch label names to the end
                    i = 0 # reduce index, after previous labels were deleted
                    for label_column in label_columns:
                        try:
                            feature_names.append(feature_names[label_column-i])
                            del feature_names[label_column-i]
                            i += 1
                        except IndexError:
                            feature_names.append("")
                    # create new feature names
                    feature_names = [item for index, item in enumerate(
                        feature_names) if not index+1 in new_ignored_columns]
                    for _ in label_columns:
                        feature_names.pop(-1)

                # read the data line by line
                for line in data_set:
                    if not delimiter in line:
                        warnings.warn("Line without delimiter:\n%s" % str(line))
                        continue
                    line = line.split(delimiter)
                    line[-1] = line[-1].rstrip("\n\r")
                    label = []
                    i = 0
                    for label_column in label_columns:
                        label.append(line.pop(label_column-i))
                        i += 1
                    if type(label) == str:
                        label = label.strip(' "')
                        label = label.strip(" '")
                    if label not in classes_names:
                        classes_names.append(label)
                    # create new line without the ignored columns
                    vector = [item for index,item in enumerate(line) if not
                              index+1 in new_ignored_columns]
                    sample = FeatureVector(numpy.atleast_2d([vector]).astype(
                        numpy.float64), feature_names=feature_names)
                    if len(label) == 1:
                        label = label[-1]
                    self.add_sample(sample=sample, label=label, 
                                    train=train_test, split=split_nr, 
                                    run=run_nr)
                self.update_meta_data({"feature_names": sample.feature_names,
                       "num_features": len(sample.feature_names),
                       "classes_names": classes_names})

        return self.data[(run_nr, split_nr, train_test)]

    def store(self, result_dir, s_format = ["pickle", "real"]):
        """ Stores this collection in the directory *result_dir*.
        
        In contrast to *dump* this method stores the collection
        not in a single file but as a whole directory structure with meta
        information etc. The data sets are stored separately for each run, 
        split, train/test combination.
        
        The method expects the following parameters:
          * *result_dir* The directory in which the collection will be stored
          * *name* The prefix of the file names in which the individual \
                   data sets are stored. The actual file names are determined \
                   by appending suffixes that encode run, split, train/test \
                   information. Defaults to "features".
          * *format* A list with information about the format in which the 
                    actual data sets should be stored. The first entry specifies
                    the file format. If it is "arff" the second entry specifies the
                    attribute format. 
                    
                    Examples: ["arff", "real"], ["arff", "{0,1}"]
                    
                    .. todo:: Someone could implement the format ["fasta"] for sax features
                    
                    To store the data in comma separated values, use ["csv", "real"].
                    
                    (*optional, default: ["pickle", "real"]*)

        .. todo:: Adapt storing of csv file to external library instead of
                  doing it manually.

        """
        name = "features"
        # Update the meta data
        try:
            author = pwd.getpwuid(os.getuid())[4]
        except:
            author = "unknown"
            self._log("Author could not be resolved.",level=logging.WARNING)
        self.update_meta_data({"type": "feature_vector",
                               "storage_format": s_format,
                               "author": author,
                               "data_pattern": "data_run" + os.sep 
                                                 + name + "_sp_tt." + s_format[0]})
        
        if type(s_format) == list:
            s_type = s_format[1]
            s_format = s_format[0]
        else:
            s_type = "real"
            
        if not s_format in ["csv", "arff", "pickle"]:
            self._log("Storage format not supported! Using default.", 
                      level=logging.ERROR)
            s_format = "pickle"
        
        # Iterate through splits and runs in this dataset
        for key, feature_vectors in self.data.iteritems():
            # Construct result directory
            result_path = result_dir + os.sep + "data" \
                            + "_run%s" % key[0]
            if not os.path.exists(result_path):
                os.mkdir(result_path)
                
            key_str = "_sp%s_%s" % key[1:]
            # Store data depending on the desired format
            if s_format == "pickle":
                result_file = open(os.path.join(result_path, 
                                                name + key_str + ".pickle"),
                                   "w")
         
                cPickle.dump(feature_vectors, result_file, cPickle.HIGHEST_PROTOCOL)
            elif s_format == "arff": # Write as ARFF
                result_file = open(os.path.join(result_path, 
                                                name + key_str + ".arff"),"w")
                # Create the arff file header
                relation_name = result_dir.split(os.sep)[-1]
                result_file.write('@relation "%s"\n' % relation_name)
                # Write the type of all features
                for feature_name in self.meta_data["feature_names"]:
                    result_file.write("@attribute %s %s\n" % (feature_name,  s_type))
                classString = "" + ",".join(sorted(self.meta_data["classes_names"])) + ""

                result_file.write("@attribute class {%s}\n" % classString)
                
                result_file.write("@data\n")
                # Write all given training data into the ARFF file
                fv = feature_vectors[0][0]
                if numpy.issubdtype(fv.dtype, numpy.string_):
                    feature_format = "%s,"
                elif numpy.issubdtype(fv.dtype, numpy.floating):
                    feature_format = "%f,"
                elif numpy.issubdtype(fv.dtype, numpy.integer):
                    feature_format = "%d,"
                for features, class_name in feature_vectors:
                    for feature in features[0]:
                        result_file.write(feature_format % feature)
                    result_file.write("%s\n" % str(class_name))
            elif s_format == "csv": # Write as Comma Separated Value
                result_file = open(os.path.join(result_path, 
                                                name + key_str + ".csv"),"w")
                for feature_name in self.meta_data["feature_names"]:
                    result_file.write("%s," % (feature_name))
                result_file.write("\n")
                fv = feature_vectors[0][0]
                if numpy.issubdtype(fv.dtype, numpy.floating):
                    feature_format = "%f,"
                elif numpy.issubdtype(fv.dtype, numpy.integer):
                    feature_format = "%d,"
                else:
                    feature_format = "%s,"
                for features, class_name in feature_vectors:
                    f = features.view(numpy.ndarray)
                    for feature in f[0]:
                        result_file.write(feature_format % feature)
                    result_file.write("%s\n" % str(class_name))
            result_file.close()

        #Store meta data
        BaseDataset.store_meta_data(result_dir,self.meta_data)
