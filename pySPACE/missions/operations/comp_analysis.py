""" Creates various comparing plots on several levels for a :class:`~pySPACE.resources.dataset_defs.performance_result.PerformanceResultSummary`

This module contains implementations of an operation and
a process for analyzing data contained in a csv file (typically the result
of a Weka Classification Operation).

A *CompAnalysisProcess* consists of evaluating the effect of several parameter
on a set of metrics. For each numeric parameter, each pair of numeric parameters
and each nominal parameter, one plot is created for each metric.

Furthermore, for each value of each parameter, the rows of the data where
the specific parameter takes on the specific value are selected and the same
analysis is done for this subset recursively.

This is useful for large experiments where several parameters are differed.
For instance, if one wants to analyze how the performance is for certain
settings of certain parameters, on can get all plots in the respective
subdirectories. For instance, if one is interested only in the performance
of one classifier, on can go into the subdirectory of the respective classifier.

The "Comp Analysis Operation" is similar to the "Analysis Operation"; the only
difference is that plots corresponding to the same parameters but different
metrics are created in one single pdf file. This ensures better comparability
and lower computation time, as less files are created.

.. todo:: Unification with analysis operation to prevent from doubled code
.. todo:: correct parameter documentation
"""
import sys
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing
import pylab
import numpy
import os
import itertools
import matplotlib.font_manager
from collections import defaultdict
from pySPACE.tools.filesystem import create_directory
from pySPACE.resources.dataset_defs.base import BaseDataset

import pySPACE
from pySPACE.missions.operations.base import Operation, Process


class CompAnalysisOperation(Operation):
    """ Operation for analyzing and plotting data in compressed format
    
    A *CompAnalysisOperation* is similar to a *AnalysisOperation*:
    
    An *AnalysisOperation* loads the data from a csv-file (typically the result
    of a Weka Classification Operation) and evaluates the effect of various
    parameters on several metrics.
    """

    def __init__( self, processes, operation_spec, result_directory,
                   number_processes, create_process=None):
        super( CompAnalysisOperation, self ).__init__( processes,
                                             operation_spec, result_directory )

        self.operation_spec = operation_spec
        self.create_process = create_process
        self.number_processes = number_processes

    @classmethod
    def create(cls, operation_spec, result_directory, debug=False, input_paths=[]):
        """
        A factory method that creates an Analysis operation based on the
        information given in the operation specification operation_spec.
        If debug is TRUE the creation of the Analysis Processes will not
        be in a separated thread.
        """
        assert(operation_spec["type"] == "comp_analysis")
        input_path = operation_spec["input_path"]
        summary = BaseDataset.load(os.path.join(pySPACE.configuration.storage,
                                      input_path))
        data_dict = summary.data
        ## Done
        
        # Determine the parameters that should be analyzed
        parameters = operation_spec["parameters"]
        
        # Determine dependent parameters, which don't get extra resolution
        try:
            dep_par = operation_spec["dep_par"]
        except KeyError:
            dep_par=[]
        
        # Determine the metrics that should be plotted
        spec_metrics = operation_spec["metrics"]
        metrics=[]
        for metric in spec_metrics:
            if data_dict.has_key(metric):
                metrics.append(metric)
            else:
                import warnings
                warnings.warn('The metric "' + metric + '" is not contained in the results csv file.')
        if len(metrics)==0:
            warnings.warn('No metric available from spec file, default to first dict entry.')
            metrics.append(data_dict.keys()[0])
            
        # Determine how many processes will be created
        number_parameter_values = [len(set(data_dict[param])) for param in parameters]
        number_processes = cls._numberOfProcesses(0, number_parameter_values)+1
        
        logscale = False
        if operation_spec.has_key('logscale'):
            logscale = operation_spec['logscale']
        
        markertype='x'
        if operation_spec.has_key('markertype'):
            markertype = operation_spec['markertype']
        
        if debug == True:
            # To better debug creation of processes we don't limit the queue 
            # and create all processes before executing them
            processes = processing.Queue()
            cls._createProcesses(processes, result_directory, data_dict, parameters, 
                                   dep_par, metrics, logscale, markertype, True)
            return cls( processes, operation_spec, result_directory, number_processes)
        else:
            # Create all plot processes by calling a recursive helper method in 
            # another thread so that already created processes can be executed
            # although creation of processes is not finished yet. Therefore a queue 
            # is used which size is limited to guarantee that not to much objects 
            # are created (since this costs memory). However, the actual number 
            # of 100 is arbitrary and might be reviewed.
            processes = processing.Queue(100)
            create_process = processing.Process(target=cls._createProcesses,
                             args=( processes, result_directory, data_dict, 
                                    parameters, dep_par, metrics, logscale, markertype, True))
            create_process.start()
            # create and return the comp_analysis operation object
            return cls( processes, operation_spec, result_directory, 
                                   number_processes, create_process)

    @classmethod
    def _numberOfProcesses(cls, number_of_processes, number_of_parameter_values):
        """Recursive function to determine the number of processes that 
        will be created for the given *number_of_parameter_values*
        """
        if len(number_of_parameter_values) < 3:
            number_of_processes += sum(number_of_parameter_values)
            return number_of_processes
        else:
            for i in range(len(number_of_parameter_values)):
                number_of_processes += number_of_parameter_values[i] * \
                  cls._numberOfProcesses(0, 
                  [number_of_parameter_values[j] for j in \
                  range(len(number_of_parameter_values)) if j != i]) + \
                  number_of_parameter_values[i]
            return number_of_processes       
        
    @classmethod
    def _createProcesses(cls, processes, result_dir, data_dict, parameters,dep_par,
                         metrics, logscale, markertype, top_level):
        """Recursive function that is used to create the analysis processes
        
        Each process creates one plot for each numeric parameter, each pair of
        numeric parameters, and each nominal parameter based on the data
        contained in the *data_dict*. The results are stored in *result_dir*.
        The method calls itself recursively for each value of each parameter.
        """
        
        # Create the analysis process for the given parameters and the
        # given data and put it in the executing-queue
        process = CompAnalysisProcess(result_dir, data_dict, parameters, metrics, logscale, markertype)
        processes.put( process )
        # If we have less than two parameters it does not make sense to
        # split further
        if len(parameters) < 2 or len(parameters)==len(dep_par):
            # If we have only one parameter to visualize,
            # we don't need to create any further processes,
            # and we have to finish the creating process.
            return
        
        # For each parameter
        for proj_parameter in parameters:
            if proj_parameter in dep_par:
                continue
            # We split the data based on the values of this parameter
            remaining_parameters = [parameter for parameter in parameters
                                        if parameter != proj_parameter]
            # For each value the respective projection parameter can take on
            for value in set(data_dict[proj_parameter]):
                # Project the result dict onto the rows where the respective
                # parameter takes on the given value
                projected_dict = defaultdict(list)
                entries_added = False
                for i in range(len(data_dict[parameter])):
                    if data_dict[proj_parameter][i] == value:
                        entries_added = True
                        for column_key in data_dict.keys():
                            if column_key == proj_parameter: continue
                            projected_dict[column_key].append(data_dict[column_key][i])
                # If the projected_dict is empty we continue
                if not entries_added:
                    continue
                
                # Create result_dir and do the recursive call for the
                # projected data
                proj_result_dir = result_dir + os.sep + "%s#%s" % (proj_parameter,
                                                                   value)
                create_directory(proj_result_dir)
                cls._createProcesses(processes, proj_result_dir, projected_dict,
                                     remaining_parameters, dep_par, metrics, logscale, markertype, False)
        
        if top_level == True:
            # print "last process created"
            # give executing process the sign that creation is now finished
            processes.put(False)
    
    def consolidate(self):
        pass


class CompAnalysisProcess(Process):
    """ Process for analyzing and plotting data
    
    A *CompAnalysisProcess* is quite similar to a *AnalysisProcess*:
    
    An *CompAnalysisProcess* consists of evaluating the effect of several
    *parameters*  on a set of *metrics*. For each numeric parameter,
    each pair of numeric parameters and each nominal parameter,
    one plot is created for all metrics (instead of one plot for each metric as
    in *AnalysisProcess*).
    
    **Expected parameters**
      *result_dir* : The directory in which the actual results are stored
      *data_dict* : A dictionary containing all the data. The dictionary \
                    contains a mapping from an attribute  (e.g. accuracy) \
                    to a list of values taken by an attribute.  An entry is the\
                    entirety of all i-th values over all dict-values
      *parameters* : The parameters which have been varied during the \
                      experiment and whose effect on the *metrics* should be \
                      investigated. These must be keys of the *data_dict*.
      *metrics*: The metrics the should be evaluated. Must be keys of the \
                 *data_dict*.
      *logscale*: Boolean, numeric x-axis will be scaled log'ly if true.
      *markertype*: A string like '.' defining the marker type for certain \
                    plots. Default is 'x'.
    
    """
    
    def __init__(self, result_dir, data_dict, parameters, metrics, logscale, markertype):
        super(CompAnalysisProcess, self).__init__()
        
        self.result_dir = result_dir
        self.data_dict = data_dict
        self.parameters = parameters
        self.logscale = logscale
        self.markertype = markertype
        # Usually the value of a metric for a certain situation is just a scalar
        # value. However, for certain metrics the value can be a sequence
        # (typically the change of some measure over time). These cases must be
        # indicated externally by the the usage of ("metric_name", "sequence")
        # instead of just "mteric_name".
        self.metrics = [(metric, "scalar") if isinstance(metric, basestring)
                            else metric for metric in metrics]
    
    def __call__(self):
        """
        Executes this process on the respective modality
        """
        
        # Restore configuration
        pySPACE.configuration = self.configuration
        
        ############## Prepare benchmarking ##############
        super(CompAnalysisProcess, self).pre_benchmarking()
        
        # Split parameters into nominal and numeric parameters
        nominal_parameters = []
        numeric_parameters = []
        for parameter in self.parameters:
            try:
                # Try to create a float of the first value of the parameter
                float(self.data_dict[parameter][0])
                # No exception thus a numeric attribute
                numeric_parameters.append(parameter)
            except ValueError:
                # This is not a numeric parameter, treat it as nominal
                nominal_parameters.append(parameter)
            except KeyError:
                #"This exception should inform the user about wrong parameters in his YAML file.", said Jan-Hendrik-Metzen.
                import warnings
                warnings.warn('The parameter "' + parameter + '" is not contained in the performance results.')
            except IndexError:
                #This exception informs the user about wrong parameters in his YAML file.
                import warnings
                warnings.warn('The parameter "' + parameter + '" could not be found.')
            #TODO: Better exception treatment! The Program should ignore unknown
            # parameters and go on after giving information on the wrong parameter.
        
        #Create the figure and initialize ...
        fig1 = pylab.figure()
        ax = [] #... a list of matplotlib axes objects
        im = [] #... and images. This is needed for the color bars of
                # contour plots
        
        # For all performance measures
        # Pass figure and axes to functions, let them add their plot,
        # then return fig and axes.
        for metric in self.metrics:
            if metric[1] == 'scalar':
                fig1, ax, im = self._scalar_metric(metric[0], numeric_parameters,
                                    nominal_parameters, fig1, ax, im)
            
            else:
                im.append([]) # Don't need the image in this case
                fig1, ax = self._sequence_metric(metric[0], numeric_parameters,
                                      nominal_parameters, fig1, ax, **metric[2])

        
        nmetrics = self.metrics.__len__() # Check Nr of metrices
        naxes = ax.__len__()              # and axes
        # And compute how many subplot columns are needed
        plot_cols = numpy.int(numpy.ceil(numpy.float(naxes)
                                         / numpy.float(nmetrics)))
        
        # Now loop the axes
        for i in range(naxes):
            # And move every single one to it's desired location
            ax[i].change_geometry(nmetrics, plot_cols, i + 1)
            # in case of a contour plot also add a corresponding colorbar...
            if ax[i]._label[-1] == 'C':
                fig1.sca(ax[i])
                pylab.colorbar(im[i]) #... by using the correct image
        
        # enlarge canvas so that nothing overlaps
        fig1.set_size_inches((10 * plot_cols, 7 * nmetrics))
        # save...
        fig1.savefig("%s%splot.pdf" % (self.result_dir, os.sep), bbox_inches="tight")
        # and clean up.
        del fig1
        pylab.gca().clear()
        pylab.close("all")
        ############## Clean up after benchmarking ##############
        super(CompAnalysisProcess, self).post_benchmarking()
    
    def _scalar_metric(self, metric, numeric_parameters, nominal_parameters, fig1, ax, im):
        """ Creates the plots for a scalar metric """
        # For all numeric parameters
        for index, parameter1 in enumerate(numeric_parameters):
            # need not to save image in these cases, as no colorbar is needed
            im.append([])
            fig1, ax = self._plot_numeric(self.data_dict, self.result_dir,
                               fig1, ax,
                               x_key=parameter1,
                               y_key=metric,
                               one_figure=False,
                               show_errors=True)
            
            # For all combinations of two numeric parameters
            for parameter2 in numeric_parameters[index + 1:]:
                axis_keys = [parameter1, parameter2]
                fig1, ax, im = self._plot_numeric_vs_numeric(self.data_dict,
                                              self.result_dir, fig1, ax, im,
                                              axis_keys=axis_keys,
                                              value_key=metric)
            
            # For all combinations of a numeric and a nominal parameter
            for parameter2 in nominal_parameters:
                im.append([])
                axis_keys = [parameter1, parameter2]
                fig1, ax = self._plot_numeric_vs_nominal(self.data_dict,
                                              self.result_dir, fig1, ax,
                                              numeric_key=parameter1,
                                              nominal_key=parameter2,
                                              value_key=metric)
                # fig1.gca().set_xlim(0.0, 0.01)
                # fig1.gca().set_xscale('log')
        
        # For all nominal parameters:
        for index, parameter1 in enumerate(nominal_parameters):
            im.append([])
            fig1, ax = self._plot_nominal(self.data_dict, self.result_dir,
                               fig1, ax, x_key=parameter1,
                               y_key=metric)
        
        return fig1, ax, im
    
    def _sequence_metric(self, metric, numeric_parameters, nominal_parameters,
                         fig1, ax, mwa_window_length):
        """ Creates the plots for a sequence metric
        
        .. todo:: Do not distinguish nominal and numeric parameters for the moment
        """
        parameters = list(numeric_parameters)
        parameters.extend(nominal_parameters)
        
        metric_values = map(eval, self.data_dict[metric])
        # Sometimes, the number of values are not identical, so we cut all to
        # the same minimal length
        num_values = min(map(len, metric_values))
        metric_values = map(lambda l: l[0:num_values], metric_values)
        
        # Moving window average of the metric values
        mwa_metric_values = []
        for sequence in metric_values:
            mwa_metric_values.append([])
            for index in range(len(sequence)):
                # Chop window such that does not go beyond the range of
                # available values
                window_width = min(index, len(sequence) - index - 1,
                                   mwa_window_length / 2)
                subrange = (index - window_width, index + window_width)
                mwa_metric_values[-1].append(numpy.mean(sequence[subrange[0]:subrange[1]]))
        
        # For each parameter
        for parameter in parameters:
            # Split the data according to the values the parameter takes on
            curves = defaultdict(list)
            for row in range(len(self.data_dict[parameter])):
                curves[self.data_dict[parameter][row]].append(mwa_metric_values[row])
            
            # Plot the mean curve over all runs for this parameter setting
            ax.append(fig1.add_subplot(111, label="%d" % (ax.__len__() + 1)))
            fig1.sca(ax[-1])
            for parameter_value, curve in curves.iteritems():
                # Create a simple plot
                pylab.plot(range(len(metric_values[0])), numpy.mean(curve, 0),
                           label=parameter_value)
#                # Create an errorbar plot
#                pylab.errorbar(x = range(len(metric_values[0])),
#                               y = numpy.mean(curve, 0),
#                               yerr = numpy.std(curve, 0),
#                               elinewidth = 1, capsize = 5,
#                               label = parameter_value)
            
            lg = pylab.legend(loc=0,fancybox=True)
            lg.get_frame().set_facecolor('0.90')
            lg.get_frame().set_alpha(.3)
            
            pylab.xlabel("Step")
            pylab.ylabel(metric)
        
        return fig1, ax
    
    def _plot_numeric(self, data, result_dir, fig1, ax, x_key, y_key, conditions=[],
                          one_figure=False, show_errors=False):
        """ Creates a plot of the y_keys for the given numeric parameter x_key.
        
        A method that  allows to create a plot that visualizes the effect
        of differing one variable onto a second one (e.g. the effect of
        differing the number of features onto the accuracy).
        
        **Expected parameters**
          *data* : A dictionary, that contains a mapping from an attribute \
                  (e.g. accuracy) to a list of values taken by an attribute. \
                  An entry is the entirety of all i-th values over all dict-values
                  
          *result_dir* : The directory in which the plots will be saved.
          
          *x_key* : The key of the dictionary whose values should be used as \
                    values for the x-axis (the independent variables)
                    
          *y_key* : The key of the dictionary whose values should be used as\
                    values for the y-axis, i.e. the dependent variables\
                    
          *conditions* : A list of functions that need to be fulfilled in order to \
                         use one entry in the plot. Each function has to take two \
                         arguments: The data dictionary containing all entries and \
                         the index of the entry that should be checked. Each condition \
                         must return a boolean value.
                         
          *one_figure* : If true, all curves are plotted in the same figure.
                         Otherwise, for each value of curve_key, a new figure\
                         is generated (currently ignored)
                         
          *show_errors* : If true, error bars are plotted
        
        """
        
        ax.append(fig1.add_subplot(111, label="%d" % (ax.__len__() + 1)))
        fig1.sca(ax[-1])
        
        pylab.xlabel(x_key)
        
        curves = defaultdict(lambda : defaultdict(list))
        for i in range(len(data[x_key])):
            # Check is this particular entry should be used
            if not all(condition(data, i) for condition in conditions):
                continue
            
            # Get the value of the independent variable for this entry
            x_value = float(data[x_key][i])
            # Attach the corresponding value to the respective partition
            y_value = float(data[y_key][i])
            curves[y_key][x_value].append(y_value)
        
        for y_key, curve in curves.iteritems():
            curve_x = []
            curve_y = []
            for x_value, y_values in sorted(curve.iteritems()):
                curve_x.append(x_value)
                curve_y.append(y_values)
            
            # create the actual plot
            if show_errors:
                # Create an error bar plot
                if self.markertype == '':
                    linestyle = '-'
                else:
                    linestyle = '--'
                pylab.errorbar(x=curve_x,
                               y=map(numpy.mean, curve_y),
                               yerr=map(numpy.std, curve_y),
                               elinewidth=1,
                               capsize=5,
                               label=y_key,
                               fmt=linestyle,
                               marker=self.markertype)
            else:
                # Create a simple plot
                pylab.plot(curve_x,
                           map(numpy.mean, curve_y),
                           label=y_key)
        
        lg = pylab.legend(loc=0, fancybox=True)
        lg.get_frame().set_facecolor('0.90')
        lg.get_frame().set_alpha(.3)
        
        pylab.ylabel(y_key)
        # pylab.title('_plot_numeric')


        if self.logscale:
            fig1.gca().set_xscale('log')
            left_lim = min(curve_x)**(21.0/20.0) / max(curve_x)**(1.0/20.0)
            right_lim = max(curve_x)**(21.0/20.0) / min(curve_x)**(1.0/20.0)
            fig1.gca().set_xlim(left_lim, right_lim)
        else:
            border = (max(curve_x) - min(curve_x))/20
            fig1.gca().set_xlim(min(curve_x)-border, max(curve_x)+border)
            
        return fig1, ax

    
    def _plot_numeric_vs_numeric(self, data, result_dir, fig1, ax, im, axis_keys, value_key):
        """ Contour plot of the value_keys for the two numeric parameters axis_keys.
        
        A method that  allows to create a contour plot that visualizes the effect
        of differing two variables on a third one (e.g. the effect of differing
        the lower and upper cutoff frequency of a bandpass filter onto
        the accuracy).
        
        **Expected parameters**
          *data* : A dictionary that contains a mapping from an attribute \
                   (e.g. accuracy) to a list of values taken by an attribute. \
                   An entry is the entirety of all i-th values over all dict-values.

          *result_dir*: The directory in which the plots will be saved.

          *axis_keys*: The two keys of the dictionary that are assumed to have \
                       an effect on a third variable (the dependent variable)

          *value_key*: The dependent variables whose values determine the \
                       color of the contour plot
        
        """
        assert(len(axis_keys) == 2)
        
        # Determine a sorted list of the values taken on by the axis keys:
        x_values = set([float(value) for value in data[axis_keys[0]]])
        x_values = sorted(list(x_values))
        
        y_values = set([float(value) for value in data[axis_keys[1]]])
        y_values = sorted(list(y_values))

        # We cannot create a contour plot if one dimension is only 1d so we
        # return the unchanged figure
        if len(x_values) == 1 or len(y_values) == 1:
            return fig1, ax, im
        
        ax.append(fig1.add_subplot(111, label="%dC" % (ax.__len__() + 1)))
        fig1.sca(ax[-1])
        
        # Create a meshgrid of them
        X, Y = pylab.meshgrid(x_values, y_values)
        # Determine the average value taken on by the dependent variable
        # for each combination of the the two source variables
        Z = numpy.zeros((len(x_values), len(y_values)))
        counter = numpy.zeros((len(x_values), len(y_values)))
        for i in range(len(data[axis_keys[0]])):
            x_value = float(data[axis_keys[0]][i])
            y_value = float(data[axis_keys[1]][i])
            value = float(data[value_key][i])
            Z[x_values.index(x_value), y_values.index(y_value)] += value
            counter[x_values.index(x_value), y_values.index(y_value)] += 1
        Z = Z / counter
        
        # Create the plot for this specific dependent variable
        #pylab.figure()
        #cf = pylab.contourf(X, Y, Z.T, N=20)
        im.append(pylab.contourf(X, Y, Z.T, N=20))
        
        #pylab.colorbar(cf)
        #pylab.colorbar(cf)
        
        if self.logscale:
            fig1.gca().set_xscale('log')
            fig1.gca().set_yscale('log')
        
        pylab.xlim(min(x_values), max(x_values))
        pylab.ylim(min(y_values), max(y_values))
        
        pylab.xlabel(axis_keys[0])
        pylab.ylabel(axis_keys[1])

        pylab.title( value_key )
        
        if self.logscale:
            fig1.gca().set_xscale('log')
            fig1.gca().set_yscale('log')
            
        return fig1, ax, im
        
    
    def _plot_numeric_vs_nominal(self, data, result_dir, fig1, ax, numeric_key,
                                 nominal_key, value_key):
        """ Plot for comparison of several different values of a nominal parameter
        
        A method that  allows to create a plot that visualizes the effect of
        varying one numeric parameter onto the performance for several
        different values of a nominal parameter.
        
        **Expected parameters**
          * data* : A dictionary that contains a mapping from an attribute \
                    (e.g. accuracy) to a list of values taken by an attribute. \
                    An entry is the entirety of all i-th values over all dict-values.
                    
          *result_dir*: The directory in which the plots will be saved.
          
          *numeric_key*: The numeric parameter whose effect (together with the \
                         nominal parameter) onto the dependent variable should \
                         be investigated.
        
         *nominal_key*: The nominal parameter whose effect (together with the \
                        numeric parameter) onto the dependent variable should
                        be investigated.
        
          *value_key* : The dependent variables whose values determine the \
                        color of the contour plot
        
        """
        
        ax.append(fig1.add_subplot(111, label="%d" % (ax.__len__() + 1)))
        fig1.sca(ax[-1])
        
        # Determine a mapping from the value of the nominal value to a mapping
        # from the value of the numeric value to the achieved performance:
        # nominal -> (numeric -> performance)
        curves = defaultdict(lambda: defaultdict(list))
        for i in range(len(data[nominal_key])):
            curve_key = data[nominal_key][i]
            parameter_value = float(data[numeric_key][i])
            if value_key[0] is not "#":
                performance_value = float(data[value_key][i])
            else: # A weighted cost function
                weight1, value_key1, weight2, value_key2 = value_key[1:].split("#")
                performance_value = float(weight1) * float(data[value_key1][i]) \
                                        + float(weight2) * float(data[value_key2][i])
            
            curves[curve_key][parameter_value].append(performance_value)
        
        linecycler = itertools.cycle(['-']*7 + ['-.']*7 +
                                      [':']*7 + ['--']*7).next
        
        # Iterate over all values of the nominal parameter and create one curve
        # in the plot showing the mapping from numeric parameter to performance
        # for this particular value of the nominal parameter
        minmax_x = [pylab.inf,-pylab.inf]
        for curve_key, curve in curves.iteritems():
            x_values = []
            y_values = []
            for x_value, y_value in sorted(curve.iteritems()):
                x_values.append(x_value)
                # Plot the mean of all values of the performance for this
                # particular combination of nominal and numeric parameter
                y_values.append(pylab.mean(y_value))
            pylab.plot(x_values, y_values,marker=self.markertype,
                       label=curve_key, linestyle=linecycler())
            minmax_x = [min(minmax_x[0], min(x_values)),
                        max(minmax_x[1], max(x_values))]
        
        pylab.gca().set_xlabel(numeric_key.replace("_", " "))
        if value_key[0] is not "#":
            pylab.gca().set_ylabel(value_key.replace("_", " "))
        else:
            pylab.gca().set_ylabel("%s*%s+%s*%s" % tuple(value_key[1:].split("#")))
        
        # Shrink legend if too many curves
        if len(curves) > 6:
            prop = matplotlib.font_manager.FontProperties(size='small')
            lg = pylab.legend(prop=prop, loc=0, ncol=2,fancybox=True)
        else:
            lg = pylab.legend(loc=0,fancybox=True)
        lg.get_frame().set_facecolor('0.90')
        lg.get_frame().set_alpha(.3)
        
        if self.logscale:
            fig1.gca().set_xscale('log')
            try:
                left_lim = minmax_x[0]**(21.0/20.0) / minmax_x[1]**(1.0/20.0)
                right_lim = minmax_x[1]**(21.0/20.0) / minmax_x[0]**(1.0/20.0)
                fig1.gca().set_xlim(left_lim, right_lim)
            except:
                pass
        else:
            border = (minmax_x[1]-minmax_x[0])/20
            fig1.gca().set_xlim(minmax_x[0]-border,minmax_x[1]+border)
        
        return fig1, ax

    
    def _plot_nominal(self, data, result_dir, fig1, ax, x_key, y_key):
        """ Creates a boxplot of the y_keys for the given nominal parameter x_key.
        
        A method that  allows to create a plot that visualizes the effect
        of differing one nominal  variable onto a second one (e.g. the effect of
        differing the classifier onto the accuracy).
        
        **Expected parameters**
          *data*:  A dictionary, that contains a mapping from an attribute \
                   (e.g. accuracy) to a list of values taken by an attribute. \
                   An entry is the entirety of all i-th values over all dict-values
                   
          *result_dir*: The director in which the plots will be saved.
          
          *x_key*: The key of the dictionary whose values should be used as \
                    values for the x-axis (the independent variables)
                    
          *y_key*: The key of the dictionary whose values should be used as\
                    values for the y-axis, i.e. the dependent variable
        
        """
        
        ax.append(fig1.add_subplot(111, label="%d" % (ax.__len__() + 1)))
        fig1.sca(ax[-1])
        
        # Create the plot for this specific dependent variable
        values = defaultdict(list)
        for i in range(len(data[x_key])):
            parameter_value = data[x_key][i]
            if y_key[0] is not "#":
                performance_value = float(data[y_key][i])
            else: # A weighted cost function
                weight1, y_key1, weight2, y_key2 = y_key[1:].split("#")
                performance_value = float(weight1) * float(data[y_key1][i]) \
                                        + float(weight2) * float(data[y_key2][i])
            
            values[parameter_value].append(performance_value)
        
        values = sorted(values.items())
#        values = [("Standard_vs_Target", values["Standard_vs_Target"]),
#                  ("MissedTarget_vs_Target", values["MissedTarget_vs_Target"])]
        
        pylab.subplots_adjust(bottom=0.3,) # the bottom of the subplots of the figure
                             
        # pylab.boxplot(map(lambda x: x[1], values))
        
        b = pylab.boxplot( map( lambda x: x[1], values ) )
        medlines = b['medians']
        medians = range(len(medlines))
        for i in range(len(medians)):
            medians[i] = medlines[i].get_ydata()[0]
        # create array with median labels with 2 decimal places of precision
        upperLabels = [str(numpy.round(m, 2)) for m in medians]
        
        pylab.gca().set_xticklabels(map(lambda x: x[0], values))
        pylab.setp(pylab.gca().get_xticklabels(), rotation= -90)
        pylab.setp(pylab.gca().get_xticklabels(), size='x-small')
        pylab.gca().set_xlabel(x_key.replace("_", " "))
        
        # top = pylab.gca().get_ylim()[1]
        #         for i in range(len(medians)):
        #             pylab.gca().text(i+1,top-(top*0.05),upperLabels[i],
        #                 horizontalalignment='center', size='x-small')
        
        bottom = pylab.gca().get_ylim()[0]
        for i in range(len(medians)):
            pylab.gca().text(i+1,bottom+(bottom*0.05),upperLabels[i],
                  horizontalalignment='center', size='x-small')
        
        if y_key[0] is not "#":
            pylab.gca().set_ylabel(y_key.replace("_", " "))
        else:
            pylab.gca().set_ylabel("%s*%s+%s*%s" % tuple(y_key[1:].split("#")))

        return fig1, ax


