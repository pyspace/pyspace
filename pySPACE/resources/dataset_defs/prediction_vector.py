""" Load and store data sets containing :mod:`Prediction Vectors<pySPACE.resources.data_types.prediction_vector>` """

import os
import cPickle
import numpy
import logging
import warnings

from pySPACE.resources.dataset_defs.base import BaseDataset
from pySPACE.tools.filesystem import get_author
from pySPACE.resources.data_types.prediction_vector import PredictionVector

class PredictionVectorDataset(BaseDataset):
    """ Prediction Vector dataset class

    The class at hand contains the methods needed to work with the datasets
    consisting of :class:`~pySPACE.resources.data_types.prediction_vector.PredictionVectorDataset`

    The following data formats are currently supported:
        - `*.csv` - with or without a header column
        - `*.pickle`

    TODO: Add functionality for the `*.arff` format.

    .. note::
        The implementation of the current dataset is adapted from the
        :class:`~pySPACE.resources.dataset_defs.feature_vector.FeatureVectorDataset` and
        :class:`~pySPACE.resources.dataset_defs.time_series_vector.TimeSeriesDataset`.
        For a more thorough documentation, we refer the reader to the 2 datasets
        mentioned above.

    **Parameters**

        :dataset_md:

            Dictionary containing meta data for the collection to be loaded.
            Out of these parameters, the most important one is the number of
            predictors since it how the Prediction Vectors will be generated.

        :num_predictors:

            The number of predictors that each PredictionVector contains.
            This parameter is important for determining the dimensionality of
            the PredictionVector.



    **Special CSV Parameters**

        :delimiter:
            Needed only when dealing with `*.csv` files. The default is
            `,` is used or the tabulator `\t`. When storing, `,` is used.

            (*recommended, default: ','*)

        :label_column:
            Column containing the true label of the data point

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

    :Author: Andrei Ignat (andrei_cristian.ignat@dfki.de)
    :Created: 2014/10/15
    """
    def __init__(self, dataset_md=None, num_predictors=1,**kwargs):
        """ Read out the data from the given collection """
        super(PredictionVectorDataset, self).__init__(dataset_md=dataset_md)
        # we want to remember the number of predictions
        if not dataset_md is None: #data has to be loaded
            self._log("Loading prediction vectors from input collection.")
            dataset_dir = self.meta_data["dataset_directory"]
            s_format = self.meta_data["storage_format"]
            if type(s_format) == list:
                s_format = s_format[0]
            # code copied from TimeSeriesDataset defs
            if dataset_md.has_key("data_pattern") \
                    and not self.meta_data["train_test"] \
                    and self.meta_data["splits"] == 1 \
                    and self.meta_data["runs"] == 1:
                # The collection consists only of a single set of data, for
                # one run, one splitting, and only test data
                data = dataset_md["data_pattern"].replace("_run", "_run0")\
                    .replace("_sp", "_sp0")\
                    .replace("_tt", "_test")
                # File that contains the PredictionVector objects
                pv_file = os.path.join(dataset_dir,data)

                # Actual data will be loaded lazily
                self.data[(0, 0, "test")] = pv_file
            elif dataset_md.has_key("data_pattern"):
                for run_nr in range(self.meta_data["runs"]):
                    for split_nr in range(self.meta_data["splits"]):
                        for train_test in ["train", "test"]:
                            # The collection consists only of a single set of
                            # data, for one run, one splitting, and only test
                            # data
                            data = dataset_md["data_pattern"]\
                                .replace("_run", "_run%s" % run_nr)\
                                .replace("_sp","_sp%s" % split_nr)\
                                .replace("_tt","_%s" % train_test)
                            # File that contains the PredictionVector objects
                            pv_file = os.path.join(dataset_dir, data)
                            # Actual data will be loaded lazily
                            self.data[(run_nr, split_nr, train_test)] = pv_file
            elif dataset_md.has_key("file_name"):
                fv_file = os.path.join(dataset_dir,self.meta_data["file_name"])
                self.data[(0, 0, "test")] = fv_file
            else:
                pass
        else:
            pass

    def add_sample(self, sample, label, train, split=0, run=0):
        """ Add a prediction vector to this collection """
        # we count the total number of predictors in the dataset
        # and update it whenever a new sample is added

        if not self.meta_data.has_key("num_predictors"):
            self.update_meta_data({"num_predictors": numpy.size(sample.prediction)})
        elif self.meta_data["num_predictors"]!=numpy.size(sample.prediction):
            warnings.warn("Inconsistent number of predictors for sample."
                          "Expected %d and received %d predictors" %
                          (self.meta_data["num_predictors"],
                           numpy.size(sample.prediction)))

        try:
            # Remember all class labels since these will be stored
            if label not in self.meta_data["classes_names"]:
                self.meta_data["classes_names"].append(label)
        except KeyError:
            self.update_meta_data({"classes_names": [label]})

        super(PredictionVectorDataset, self).add_sample(sample, label,
                                                        train, split, run)

    def dump(self, result_path, name):
        """ Dumps this collection into a file """
        # Remove the predictor from the prediction vectors
        for values in self.data.itervalues():
            for (sample, label) in values:
                sample.predictor = None

        # Delegate to super class
        super(PredictionVectorDataset, self).dump(result_path, name)

    def get_data(self, run_nr, split_nr, train_test):
        """ Load the data from a prediction file """
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

        # Do lazy loading of the prediction vector objects.
        if isinstance(self.data[(run_nr, split_nr, train_test)], basestring):
            self._log("Lazy loading of %s prediction vectors from input "
                      "collection for run %s, split %s." % (train_test, run_nr,
                                                            split_nr))
            if s_format == "pickle":
                # Load the data from a pickled file
                file = open(self.data[(run_nr, split_nr, train_test)], "r")
                self.data[(run_nr, split_nr, train_test)] = cPickle.load(file)
                file.close()
                sample = self.data[(run_nr, split_nr, train_test)][0][0]
                self.update_meta_data({"num_predictors":numpy.size(sample.prediction)})

            elif "csv" in s_format:
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

                if s_format == "csv":
                    names = data_set[0].rstrip(",\n").split(delimiter)
                    data_set.pop(0)
                line = data_set[0].split(delimiter)
                line[-1] = line[-1].rstrip("\n\r")
                if line[-1] == '':
                        line.pop(-1)
                len_line = len(line)

                # get and prepare label column numbers (len_line needed)
                try:
                    true_label_column = self.meta_data["true_label_column"]
                except KeyError:
                    true_label_column = -1

                try:
                    num_predictors = self.meta_data["num_predictors"]
                except:
                    num_predictors = 1

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
                        if i > true_label_column:
                            i -= 1
                        new_ignored_columns.append(i)
                else:
                    new_ignored_columns = []
                new_ignored_columns.sort()

                # read the data line by line
                for line in data_set:
                    if not delimiter in line:
                        warnings.warn("Line without delimiter:\n%s" % str(line))
                        continue
                    line = line.split(delimiter)
                    line[-1] = line[-1].rstrip("\n\r")

                    true_label = line.pop(-1)

                    pred_labels = []
                    pred_scores = []

                    for i in range(num_predictors):
                        pred_scores.append(numpy.float64(line.pop()))
                        pred_labels.append(line.pop())

                    if true_label not in classes_names:
                        classes_names.append(true_label)

                    sample = PredictionVector(label=pred_labels,
                                              prediction=pred_scores)


                    self.add_sample(sample=sample, label=true_label,
                                    train=train_test, split=split_nr,
                                    run=run_nr)

                self.update_meta_data({"num_predictors": len(sample.prediction),
                       "classes_names": classes_names})

        return self.data[(run_nr, split_nr, train_test)]

    def store(self, result_dir, s_format=["pickle", "real"]):
        """ store the collection in *result_dir*"""

        name = "predictions"
        # Update the meta data
        author = get_author()
        self.update_meta_data({"type": "prediction_vector",
                               "storage_format": s_format,
                               "author": author,
                               "data_pattern": "data_run" + os.sep
                                                 + name + "_sp_tt." + s_format[0]})

        if not s_format in ["csv", "arff", "pickle"]:
            self._log("Storage format not supported! Using default.",
                      level=logging.ERROR)
            s_format = "pickle"

        for key, prediction_vectors in self.data.iteritems():
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
                cPickle.dump(prediction_vectors, result_file, cPickle.HIGHEST_PROTOCOL)

            elif s_format == "csv": # Write as Comma Separated Value
                result_file = open(os.path.join(result_path,
                                                name + key_str + ".csv"),"w")
                if self.meta_data["num_predictors"] == 1:
                    result_file.write("Predicted Label, Prediction Score, True Label \n")
                    for pv in prediction_vectors:
                        result_file.write("%s, %s, %s\n" % (pv[0].label[0], pv[0].prediction[0], pv[1]))
                else:
                    # we begin by dealing with the header of the csv file
                    base_header = "Predicted %(index)d Label, Prediction %(index)d Score, "
                    base_result = "%(label)s, %(score)s,"
                    header = ""
                    for i in range(self.meta_data["num_predictors"]):
                        header+= base_header % dict(index=i+1)
                    header += "True Label\n"
                    result_file.write(header)

                    # and now we can write each of the prediction vectors in turn

                    for pv in prediction_vectors:
                        result = ""
                        for i in range(self.meta_data["num_predictors"]):
                            result += base_result % dict(label=pv[0].label[i],
                                                         score=pv[0].prediction[i])

                        result += str(pv[1]) + "\n"
                        result_file.write(result)

        #Store meta data
        BaseDataset.store_meta_data(result_dir,self.meta_data)
