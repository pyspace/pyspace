""" Determine the optimal parameterization of a subflow

.. todo:: documentation: reference to subflow handler
"""

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.environments.chains.node_chain import NodeChain, NodeChainFactory, SubflowHandler
from pySPACE.tools.filesystem import create_directory

import copy
from numpy import ndarray, array, vstack, identity, ones, inf

import os
import cPickle
import logging

class ParameterOptimizationBase(BaseNode):
    """ Base class for parameter optimization nodes
    
    The overall goal is to determine the parameterization of a subflow that 
    maximizes a given metric.
    
    Nodes derived from this class can determine the optimal parameterization 
    (*variables*) of a subpart of a flow (*nodes*) fully autonomously. 
    For instance, for different numbers of features retained during feature 
    selection, different complexities/regularization coefficients might be 
    optimal for a classifier. In order to determine which feature-number is 
    optimal, one must choose classifier' complexity separately for each 
    feature number. 
    
    First of all the training data is split into training and validation data
    e.g. by cross validation (*validation_set*). Then for a chosen set of 
    parameters the data is processed as described in the specification
    (*nodes*). Then the classifier is evaluated (*evaluation*) using another 
    node, some combination of *metrics* or something else as for example a 
    derivative in future implementations. So the nodes specification should
    always include a classifier which is evaluated.
    
    This procedure is maybe repeated until a good set of parameters is found.
    The algorithm, which defines the way of choosing parameters should determine
    the node name (e.g. PatternSearchNode) and parameters specifically for the
    optimization procedure can be passed via the *optimization* spec. Also,
    general functions, especially the parts belonging to the evaluation of a 
    parameter are provided in this base class (e.g., function mapping parameter
    dictionaries to keys and a function to create a grid from given parameters)
    
    When a good parameter is found, the nodes are trained with this parameter
    on the whole data set.
    
    .. note::   In future, alternatives can be added, e.g. to combine
                the different flows of the cross validation with a simple
                ensemble classifier.
    
    .. note::   If you want to optimize parameters for each sub-split,
                this should not be done in this node.
    
    **Parameters**
    It is important to mention, that the definition of parameters of this node
    is structured into main parameters and sub parameters to describe the 
    different aspects of parameter optimization. So take care of indentations.
    
        :optimization:
            As mentioned above this parameter dictionary is used by the specific
            subclasses, i.e., by the specific optimization algorithms. Hence,
            see subclasses for documentation of possible parameters.
        
        :parallelization:
            This parameter dictionary is used for parallelization of subflow 
            execution. Possible parameters so far are *processing_modality*, 
            *pool_size* and *batch_size*. See :class:`~pySPACE.environments.chains.node_chain.SubflowHandler`
            for more information.
        
        :validation_set:
            :splits: 
                The number of splits used in an internal cross-validation loop.
                Note that more splits lead to better estimates of the 
                parametrization's performance but do also increase computation
                time considerably.
        
                (*recommended, default: 5*)
                
            :split_node:
                If no standard CV-splitter with "cv_splits" splits is used,
                an alternative node is specified by this node in
                YAML node syntax.
                
                (*optional, default: CV_Splitter*)
                
            :runs:
                Number of internal runs used for evaluation. Nodes as the 
                CV_Splitter behave different with every run. So we repeat the 
                calculation *runs* times and have a different randomizer each 
                time. The random seed used in each repetition is:
        
                    10 * external_run_number + internal_run_number
        
                The resulting performance measure is calculated with the 
                performances of different (internal) runs and splits. 
                (average - w * standard deviation)
        
                (*optional, default: 1*)
                
            :randomize:
                Changing cv-splitter with every parameter evaluation step.
        
                .. note:: Not yet implemented
        
        :evaluation:
            Specification of the sink node and the corresponding evaluation function
        
            :performance_sink_node:
                Specify a different sink node in YAML node syntax.
                Otherwise the default 'Classification_Performance_Sink'
                will be used with the following parameter *ir_class*.
                
                (*optional, default: Classification_Performance_Sink*)

            :ir_class:
                The class name (as string) for which IR statistics are computed. 

                (*recommended, default: 'Target'*)

            :metric:
                This is the metric that should be maximized. 
                Each :ref:`metric <metrics>` which is computed 
                by the *performance_sink_node* can be used, for instance 
                "Balanced_accuracy", "AUC", "F_measure"
                or even soft metrics or loss metrics.
                
                (*recommended, default: 'Balanced_accuracy'*)
        
            :std_weight:
                Cross validation gives several values for the estimated 
                performance. Therefore we use the difference expected value 
                minus std_weight times standard deviation.

                (*optional, default: 0*)
     
        :variables:
            List of the parameters, to be optimized and replaced in the 
            node_spec.
         
        :nodes:
            The original specification of the nodes that should be optimized.
            The value of the "nodes" parameter must be a standard NodeChain
            definition in YAML syntax (properly indented).
        
        :nominal_ranges:
            Similar to the ranges in the grid search, a grid can be specified
            mainly for nominal parameters.
            All the other parameters are then optimized dependent on these
            parameters. Afterwards the resulting performance value is compared
            to choose the best nominal parameter.
            When storing the results each nominal parameter is stored
            
            (*optional, default: None*)
            
        :debug:
            Switch on eventually existing debug messages
            
            (*optional, default: False*)

        :validation_parameter_settings:
            Dictionary, of parameter mappings to be replaced in the nodes
            in the validation phase of the parameter optimization.
            
            This works together with the *final_training_parameter_settings*.
            
            (*optional, default: dict()*) 
            
        :final_training_parameter_settings:
            Dictionary, of parameter mappings to be replaced in the nodes
            after the validation phase of the parameter optimization in the 
            final training phase.
            
            This works together with the *validation_parameter_settings*.
            
            A very important use case of these parameters is to switch of
            *retrain* mode in validation phase but nevertheless
            have it active in the final subflow or node.
            
            (*optional, default: dict()*)

    :Author: Mario Krell (mario.krell@dfki.de)
    :Created: 2011/08/03
    :LastChange: 2012/09/03 Anett Seeland - structural revision due to parallelization improvements
    """
    def __init__(self, flow_template, variables=[], metric='Balanced_accuracy', 
                 std_weight=0, runs=1, nominal_ranges=None, debug=False, 
                 validation_parameter_settings={}, 
                 final_training_parameter_settings={},
                 **kwargs):
        super(ParameterOptimizationBase, self).__init__(**kwargs)
        self.set_permanent_attributes(flow_template = flow_template,
                                      variables = variables,
                                      metric = metric,
                                      w = std_weight,
                                      runs = runs,
                                      nom_rng = nominal_ranges,
                                      debug = debug,
                                      flow = None,
                                      train_instances = None,
                                      performance_dict = {},
                                      search_history = [],
                                      validation_parameter_settings = validation_parameter_settings,
                                      final_training_parameter_settings = final_training_parameter_settings,
                                      classifier_information=None
                                      )

    def is_trainable(self):
        """ Return whether this node is trainable """
        return True
    
    def is_supervised(self):
        """ Return whether this node requires supervised training """
        return True

    @staticmethod
    def check_parameters(param_spec):
        """ Check input parameters of existence and appropriateness """
        assert("nodes" in param_spec and "optimization" in param_spec),\
                   "Parameter Optimization node requires specification of a " \
                   "list of nodes and optimization algorithm!"
        
        validation_set = param_spec.get("validation_set",{})
        validation_set["splits"] = validation_set.get("splits",5)
        validation_set["split_node"] = validation_set.get("split_node",
                           {'node': 'CV_Splitter', 
                            'parameters': {'splits': validation_set["splits"]}})
                
        evaluation = param_spec.get("evaluation",{})
        evaluation["ir_class"] = evaluation.get("ir_class", "Target")
        evaluation["performance_sink_node"] = evaluation.get("performance_sink_node",
                          { 'node': 'Classification_Performance_Sink', 
                            'parameters': {'ir_class': evaluation["ir_class"]}})
        
        # build flow template
        nodes_spec = param_spec.pop("nodes")
        flow_template = [{'node': 'External_Generator_Source_Node'},
                                                   validation_set["split_node"]]
        flow_template.extend(nodes_spec)
        flow_template.append(evaluation["performance_sink_node"])
        
        # Evaluate all remaining parameters
        BaseNode.eval_dict(param_spec)
        
        # params with defaults in __init__ have to be added to param_spec dict
        if validation_set.has_key("runs"): 
            param_spec["runs"] = validation_set["runs"] 
        if evaluation.has_key("metric"): 
            param_spec["metric"] = evaluation["metric"]
        if evaluation.has_key("std_weight"): 
            param_spec["std_weight"] = evaluation['std_weight']
        
        return param_spec, flow_template

    def _train(self, data, label):
        """ Train the flow on the given data vector *data* """
        # Remember the data, the actual training is done when all data is known
        if self.train_instances is None:
            self.train_instances = []
        self.train_instances.append((data, label))
        
    def _stop_training(self):
        """ Do the optimization step and define final parameter choice
        
        This is the main method of this node!
        
        .. todo:: Allow also parallelization over nominal_ranges! 
        """
        self._log("Starting optimization Process.")
        self.runs = [10 * self.run_number + run for run in range(self.runs)]
        original_flow_template = copy.copy(self.flow_template)
        # Fill in validation parameters in the template
        if not self.validation_parameter_settings=={}:
            self.flow_template = [NodeChainFactory.instantiate(template=node,
                             parametrization=self.validation_parameter_settings) for node in original_flow_template]
        if self.nom_rng is None:
            self.prepare_optimization()
            self.best_parametrization, self.best_performance = \
                                                 self.get_best_parametrization()
            self.performance_dict[self.p2key(self.best_parametrization)] = \
                              (self.best_performance, self.best_parametrization)
        else:
            nom_grid = self.search_grid(self.nom_rng)
            iterations = 0
            search_history = []
            # copy flow_template since we have to instantiate for every nom_par
            flow_template = copy.copy(self.flow_template)
            for nom_par in nom_grid:
                # for getting the best parameterization,
                # the class attribute flow_template must be overwritten
                self.flow_template = [NodeChainFactory.instantiate(template=node,
                             parametrization=nom_par) for node in flow_template]
                self.prepare_optimization()
                parametrization, performance = self.get_best_parametrization()
                self.performance_dict[self.p2key(nom_par)] = (performance, 
                                                                parametrization)
                iterations += self.iterations
                search_history.append((nom_par,self.search_history))
                # reinitialize optimization parameters
                self.re_init()
            # reconstructing the overwritten flow for further usage
            self.flow_template = flow_template
            self.iterations = iterations
            self.search_history = sorted(search_history, 
                                     key=lambda t: t[1][-1]["best_performance"])
            best_key = max(sorted(self.performance_dict.items()),
                                                          key=lambda t: t[1])[0]
            self.best_performance, self.best_parametrization = \
                                                 self.performance_dict[best_key]
            self.best_parametrization.update(dict(best_key))
        # when best parameter dict is calculated, this has to be logged
        # or saved and the chosen parameter is used for training on the
        # whole data set, independent of the chosen algorithm
        self._log("Using parameterization %s with optimal performance %s for " \
                  "metric %s." % (self.best_parametrization, 
                                            self.best_performance, self.metric))
        # Fill in the final parameters in the flow template
        if not self.final_training_parameter_settings=={}:
            self.flow_template = [NodeChainFactory.instantiate(template=node,
                             parametrization=self.final_training_parameter_settings) for node in original_flow_template]
        else:
            self.flow_template = original_flow_template
        best_flow_template = self.flow_template
        best_flow_template[1] = {'node': 'All_Train_Splitter'}
        #delete last node
        best_flow_template.pop(-1)
        self.flow = self.generate_subflow(best_flow_template, 
                                            self.best_parametrization, NodeChain)
        self.flow[-1].set_run_number(self.run_number)
        self.flow[0].set_generator(self.train_instances)
        self.flow.train()
        self._log("Training of optimal flow finished")

        # delete training instances that would be stored to disk if this node
        # is saved
        del self.train_instances

    def _execute(self, data):
        """ Execute the flow on the given data vector *data* 
        
        This method is used in offline mode and for delivering the training
        data for the next node. In the other case, *request_data_for_testing*
        is used.
        """
        if not self.classifier_information is None:
            # Delegate to internal flow object
            return self._get_flow().execute(data)
        else:
            result = self._get_flow().execute(data)
            # forward important node information via. classifier information
            try:
                self.classifier_information = \
                    result.predictor.classifier_information
            except:
                result.predictor.classifier_information = dict()
                self.classifier_information = \
                    result.predictor.classifier_information
            for key,value in self.best_parametrization.items():
                self.classifier_information[key] = value
            self.classifier_information["~~Pon_Iterations~~"] = self.iterations
            self.classifier_information["~~Pon_value~~"] = self.best_performance
            return result



    def _get_flow(self):
        """ Method introduced for consistency with flow_node 
        
        This node itself is no real flow_node,
        since the final flow is unknown during initialization,
        but specified during the optimization process.
        """
        return self.flow

    def _inc_train(self, data, class_label=None):
        """ Iterate through the nodes to train them

        The optimal parameter remains fixed and then the nodes in the optimal
        flow get the incremental training.

        Here it is important to know, that *first* the node is changed and then
        the changed data is forwarded to the *next* node.
        This is different to the normal offline retraining scheme.
        """
        for node in self._get_flow():
            if node.is_retrainable() and not node.buffering and \
                    hasattr(node, "_inc_train"):
                if not node.retraining_phase:
                    node.retraining_phase = True
                    node.start_retraining()
                node._inc_train(data,class_label)
            data = node._execute(data)

    def is_retrainable(self):
        """ Retraining if one node in subflow is retrainable """
        if self.is_retrainable:
            return True
        else:
            for node in self._get_flow():
                if node.is_retrainable():
                    return True
        return False

    def present_label(self, label):
        """ Forward the label to the subflow 
        
        *buffering* must be set to *True* only for the main node for
        incremental learning in application (live environment).
        The inner nodes must not have set this parameter.
        
        .. todo::
            Implement check on flow, if this the inner nodes do not buffer.
        """
        super(ParameterOptimizationBase, self).present_label(label)

    def store_state(self, result_dir, index=None):
        """ Store this node in the given directory *result_dir* """
        # ..todo ::  mapping of flow_id and parameterization?!
        if self.store:
            for node in self.flow:
                node.store_state(result_dir, index)
            class_dir = os.path.join(result_dir, self.__class__.__name__)
            create_directory(class_dir)
            # Store the search history
            name = "search_history_sp%d.pickle" % self.current_split
            result_file = open(os.path.join(class_dir, name), "wb")
            result_file.write(cPickle.dumps(self.search_history, 
                                            protocol=cPickle.HIGHEST_PROTOCOL))
            result_file.close()

    def get_sensor_ranking(self):
        """ Get the sensor ranking from the optimized trained flow """
        # The last node is the irrelevant 'sink node'. We need the previous one.
        return self.flow[-2].get_sensor_ranking()
    
    def re_init(self):
        """ Reset optimization params
        
        Subclasses can overwrite this method if necessary, e.g. in case some
        parameters have to be reinitialized if several optimizations are done
        """
        # handles nominal_ranges case
        pass
    
    def prepare_optimization(self):
        """ Initialize optimization procedure 
        
        Subclasses can overwrite this method if necessary.
        """
        pass
        
    def get_best_parametrization(self):
        """ Apply optimization algorithm 
        
        This method has to be implemented in the subclass. 
        """
        raise NotImplementedError("Method get_best_parametrization has not " \
                                  "been implemented in subclass %s" 
                                  % self.__class__.__name__)
        
    def get_best_dict_entry(self, performance_dict):
        """ Find the highest performance value in the dictionary
        """
        # get best performance value
        performance = max(performance_dict.items(), key=lambda t: t[1])[1]
        # get corresponding parameters
        # sorted is used here to have no randomness in the list
        best_parametrizations = [dict(par) for par,p in \
                           sorted(performance_dict.items()) if p == performance]
        return best_parametrizations[0], performance

    @staticmethod
    def search_grid(parameter_ranges):
        """ Combine each parameter in *parameter ranges* to a grid via cross product """
        # define cross product function
        crossproduct = lambda ss,row=[],level=0: len(ss)>1 \
            and reduce(lambda x,y:x+y,[crossproduct(ss[1:],row+[i],level+1)
                                         for i in ss[0]]) \
            or [row+[i] for i in ss[0]]

        # Generate grid of parameterization that should be analyzed
        if isinstance(parameter_ranges, basestring):
            parameter_ranges = eval(parameter_ranges)
        for key, value in parameter_ranges.iteritems():
            if isinstance(value, basestring) and value.startswith("eval("):
                parameter_ranges[key] = eval(value[5:-1])

        grid = map(lambda x: dict(zip(parameter_ranges.keys(), x)),
                   crossproduct(parameter_ranges.values()))
        return grid
    
    @staticmethod
    def p2key(parameters):
        """ Map parameter dictionary to hashable tuple (key for dictionary) """
        return tuple(sorted(parameters.items()))

class GridSearchNode(ParameterOptimizationBase, SubflowHandler):
    """ Grid search for optimizing the parameterization of a subflow
    
    For each parameter a list of parameters is specified (*ranges*).
    The crossproduct of all values in *ranges* is computed and a subflow is 
    evaluated for each of this parameterizations using cross-validation on the 
    training data and finally the best point in the search grid is chosen as 
    optimal point.
    
    **Parameters**
    This algorithms does not need the *variables* parameter, since it is also
    included in the ranges parameter.

        :ranges:
            A dictionary mapping parameters to the values they should be tested 
            for. If more than one parameter is given, the crossproduct of
            all parameter values is computed (i.e. each combination).
            For each resulting parameter combination, the flow specified
            in the YAML syntax is evaluated. The parameter names should be
            used somewhere in this YAML definition and should be unique
            since the instantiation is based on pure textual replacement.
            It is common to enforce this by starting and ending the parameter
            names by "~~". In the example below, the two parameters are 
            called "~~OUTLIERS~~" and "~~COMPLEXITY~~", each having 3 values.
            This results in 9 parameter combinations to be tested.
    
    **Exemplary Call**
    
    .. code-block:: yaml

        -
            node : Grid_Search
            parameters :
                optimization:
                    ranges : {~~OUTLIERS~~ : [0, 5, 10],
                              ~~COMPLEXITY~~: [0.01, 0.1, 1.0]}
                parallelization:
                    processing_modality : 'backend'
                    pool_size : 2
                validation_set :
                    split_node :
                        node : CV_Splitter
                        parameters :
                            splits : 10
                            stratified : True
                            time_dependent : True
                evaluation:
                    metric : "Balanced_accuracy"
                    std_weight: 1
                    performance_sink_node :
                        node : Sliding_Window_Performance_Sink
                        parameters :
                            ir_class : "Movement"
                            classes_names : ['NoMovement','Movement']
                            uncertain_area : 'eval([(-600,-350)])'
                            calc_soft_metrics : True
                            save_score_plot : True
                    
                variables: [~~OUTLIERS~~, ~~COMPLEXITY~~]
                
                nodes :
                    -
                        node : Feature_Normalization
                        parameters :
                            outlier_percentage : ~~OUTLIERS~~
                    -  
                        node: LibSVM_Classifier
                        parameters :
                            complexity : ~~COMPLEXITY~~
                            class_labels : ['NoMovement', 'Movement']
                            weight : [1.0, 2.0]
                            kernel_type : 'LINEAR'

    """
    def __init__(self, ranges, *args, **kwargs):
        ParameterOptimizationBase.__init__(self, *args, **kwargs)
        # extract parallelization dict for subflow handler
        SubflowHandler.__init__(self, **kwargs.get('parallelization',{}))
        self.set_permanent_attributes(grid = self.search_grid(ranges))

    @staticmethod
    def node_from_yaml(node_spec):
        """ Create the node based on the node_spec """
        node_spec = copy.deepcopy(node_spec)
        # call parent class method for most of the work
        node_spec["parameters"], flow_template = \
             ParameterOptimizationBase.check_parameters(node_spec["parameters"])
        # check grid search specific params
        optimization = node_spec["parameters"].pop("optimization")
        assert("ranges" in optimization), "Grid Search needs *ranges* parameter"
        BaseNode.eval_dict(optimization)
        node_obj = GridSearchNode(ranges=optimization["ranges"], 
                                  flow_template=flow_template, 
                                  **node_spec["parameters"])
        return node_obj

    def get_best_parametrization(self):
        """ Evaluate each flow-parameterization by running a cross validation
            on the training data for grid search
        """
        performance_dict = {}
        # create subflows 
        subflows = [self.generate_subflow(self.flow_template, grid_node) for \
                                                         grid_node in self.grid]
        # execute subflows
        result_collections = self.execute_subflows(self.train_instances, 
                                                   subflows, self.runs)
        for grid_node, result in zip(self.grid, result_collections):
            key = self.p2key(grid_node)
            performance = result.get_average_performance(self.metric) - \
                                self.w * result.get_performance_std(self.metric)
            performance_dict[key] = performance
        
        del subflows, result_collections
        # Determine the flow-parameterization that performed optimal with regard
        # to the specified metric on the grid
        best_parametrization, performance = self.get_best_dict_entry(performance_dict)
        self.iterations = len(self.grid)
        self.search_history=[{"best_parameter":best_parametrization,
                                    "best_performance":performance,
                                    "performance_dict":performance_dict,
                                    "iterations":self.iterations}]
        return best_parametrization, performance


class PatternSearchNode(ParameterOptimizationBase, SubflowHandler):
    """ Extension of the standard Pattern Search algorithm
    
    For main principle see: Numerical Optimization, Jorge Nocedal & Stephen J. Wright

    **Special Components**
    
        -   No double calculation of already visited points
        -   Step size increased, when better point is found, to speed up search
        -   Possible limit on iteration steps to be comparable to grid search
        -   cross validation cycle inside
    
    **Parameters**
    
    The following parameters have to be specified in the optimization spec.
    For the Algorithm the thereof variables parameter is important
    since it gives the order of parameters to simplify the specification of vectors,
    corresponding to point, directions or bounds.
    They can be specified as dictionaries, lists or tuple
    and they are transformed internally to array with the method *get_vector*.
    The transformation back to keys for filling them in in the node chains is done by *v2d*.
            
        :start:
            Starting point of the algorithm.
            For SVM optimization, the complexity has to be sufficiently small.
            
            (*recommended, default: ones(dimension)*)
            
        :directions:
            List of directions, being evaluated around current best point.
            
            (*optional, default: unit directions*)
            
        :start_step_size:
            First value to scale the direction vectors
            
            (*optional, default: 1.0*)
            
        :stop_step_size:
            Minimal value to scale the direction vectors
            
            If the step size gets lower, the algorithm stops.
            
            (*optional, default: 1e-10*)
            
        :scaling_factor:
            When evaluations does not deliver a better point,
            the current scaling of the directions is reduced by the scaling factor.
            Otherwise it is increased by the `up_scaling_factor`.
            
            (*optional, default: 0.5*)
        
        :up_scaling_factor:
            If the evaluation gives a better point, the step size
            is increased by this factor.
            In the default, there is no up scaling.
            
            (*optional, default: 1*)
        
        :max_iter:
            If the total number of evaluation of the directions
            exceeds this value, the algorithm also stops.
            
            (*optional, default: infinity*)
            
        :max_bound:
            Points exceeding this bounds are not evaluated.
            
            (*optional, default: inf*array(ones(dimension))*)
            
        :min_bound:
            Undershooting points are not evaluated.
            
            (*optional, default: -inf*array(ones(dimension))*)

    .. todo:: Evaluate if up_scaling makes sense and should be used here

    **Exemplary Call**
    
    .. code-block:: yaml

        -
            node : Pattern_Search
            parameters :
                parallelization :
                    processing_modality : 'local'
                    pool_size : 4
                optimization:
                    start_step_size : 0.002
                    start : [0.005,0.01]
                    directions : [[-1,-1],[1,1],[1,-1],[-1,1]]
                    stop_step_size : 0.00001
                    scaling_factor : 0.25
                    min_bound : [0,0]
                    max_bound : [10,10]
                    max_iter : 100
                validation_set :
                    split_node :
                        node : CV_Splitter
                        parameters :
                            splits : 5
                    runs : 2
                evaluation:
                    metric : "Balanced_accuracy"
                    std_weight: 1
                    ir_class : "Target" 
    
                variables: [~~W1~~, ~~W2~~]
                
                nodes :
                    -
                        node: LibSVM_Classifier
                        parameters :
                            complexity : 1
                            class_labels : ['Standard', 'Target']
                            weight : [~~W1~~, ~~W2~~]
                            kernel_type : 'LINEAR'
    """
    def __init__(self, start=[], directions=[], start_step_size=1.0, 
                 stop_step_size=1e-10, scaling_factor=0.5, up_scaling_factor=1, 
                 max_iter=inf, max_bound=[], min_bound=[],
                 **kwargs):
        ParameterOptimizationBase.__init__(self, **kwargs)
        # extract parallelization dict for subflow handler
        SubflowHandler.__init__(self, **kwargs.get('parallelization',{}))
                            
        dim = len(self.variables)
        if start != []:
            x_opt = self.get_vector(start)  
        else:
            x_opt = ones(dim)
            self._log("No starting vector given. Vector of all ones taken.",
                                                       level = logging.CRITICAL)
        
        if directions == []: 
            directions = list(vstack((identity(dim),-identity(dim)))) 
            self._log("No search directions given! Using unit directions.",
                                                        level = logging.WARNING)
        directions = [tuple(d) for d in directions]
        # delete duplicates
        directions = list(set(directions))
        
        max_bound = self.get_vector(max_bound) if max_bound != [] else inf*ones(dim)
        min_bound = self.get_vector(min_bound) if min_bound != [] else -inf*ones(dim)
        
        assert((x_opt < max_bound).all() and (x_opt > min_bound).all()), \
                                                  "Starting point is not valid!"
        # copy all params which will be changed during processing
        init_params = {"x_opt": x_opt, "directions": directions, 
                       "step_size": start_step_size}
        self.set_permanent_attributes(x_opt = x_opt,
                                      directions = directions,
                                      step_size = start_step_size,
                                      stop_step_size = stop_step_size,
                                      scaling_factor = scaling_factor,
                                      up_scaling_factor = up_scaling_factor,
                                      max_iter = max_iter,
                                      min_bound = min_bound, 
                                      max_bound = max_bound,
                                      init_params = init_params)

    @staticmethod
    def node_from_yaml(node_spec):
        """ Create the node based on the node_spec """
        node_spec = copy.deepcopy(node_spec)
        # call parent class method for most of the work
        node_spec["parameters"], flow_template = \
            ParameterOptimizationBase.check_parameters(node_spec["parameters"])
        if node_spec["parameters"].has_key("optimization"):
            BaseNode.eval_dict(node_spec["parameters"]["optimization"])
            # since pattern search specific params are all optional, add them to
            # **kwargs and let the __init__ do the default assignments
            for key, value in node_spec["parameters"].pop("optimization").iteritems():
                node_spec["parameters"][key] = value
        node_obj = PatternSearchNode(flow_template=flow_template, 
                                     **node_spec["parameters"])
        return node_obj

    def re_init(self):
        """ Reset search for optimum """
        for key, value in self.init_params.iteritems():
            setattr(self, key, value)

    def prepare_optimization(self):
        """ Calculate initial performance value """
        # we need a subflow
        subflow = self.generate_subflow(self.flow_template,self.v2d(self.x_opt))
        # parameter_key for later evaluation
        x_opt_key = self.p2key(self.v2d(self.x_opt))
        # run the flow
        result_collections = self.execute_subflows(self.train_instances, 
                                                  [subflow], self.runs)
        
        f_max = result_collections[0].get_average_performance(self.metric) \
               - self.w * result_collections[0].get_performance_std(self.metric)
        
        self.set_permanent_attributes(f_max = f_max,
                                      t_performance_dict = {x_opt_key: f_max})

    def get_vector(self,v_spec):
        """Transform list, tuple or dict to an array/vector"""
        if type(v_spec) == list:
            return array(v_spec)
        elif type(v_spec) == tuple:
            return array(v_spec)
        elif type(v_spec) == dict:
            assert(sorted(v_spec.keys())==sorted(self.variables)), \
                                          "Dictionary %s is no vector!" % v_spec
            new_v =[]
            for key in self.variables:
                new_v.append(v_spec[key])
            return array(new_v)
        elif type(v_spec) == ndarray:
            pass
        else:
            raise Exception("Could not convert %s to array." % str(v_spec))

    def v2d(self,v):
        """ Transform vector to dictionary using self.variables """
        if type(v)==dict:
            import warnings
            warnings.warn("Type conversion error in v2d. Got dictionary as input")
            return v
        else:
            return dict([(self.variables[i],v[i]) for i in range(len(self.variables))])

    def get_best_parametrization(self):
        """ Perform pattern search
        
        Evaluate set of directions around current best solution.
        If better solution is found start new evaluation of directions.
        Otherwise, reduce length of directions by scaling factor
        """
        self.iterations = 1
        self.search_history=[{"best_parameter":self.x_opt,
                                "best_performance":self.f_max,
                                "performance_dict":dict(),
                                "step_size":self.step_size,
                                "iterations":self.iterations}]
        while self.step_size >= self.stop_step_size and \
                           (self.iterations<self.max_iter or self.max_iter==-1):
            self.ps_step()
        return self.v2d(self.x_opt), self.f_max
        
    def ps_step(self):
        """ Single descent step of the pattern search algorithm """
        grid = [array(array(d)*self.step_size+self.x_opt,dtype='float64') \
                for d in self.directions \
                if (array(array(d)*self.step_size+self.x_opt,dtype='float64') \
                    < self.max_bound).all() \
                and (array(array(d)*self.step_size+self.x_opt,dtype='float64') \
                    > self.min_bound).all()]
        performance_dict = {}
        
        if self.debug:
            print "########################################################"
            print "number of total iterations:"
            print self.iterations
            print "current center:"
            print self.x_opt
            print "current function value:"
            print self.f_max
            print "current step size:"
            print self.step_size
            print "current search grid:"
            print grid
            
        chec_p = len(grid)
        red_grid =[]
        for grid_node in grid:
            grid_key = self.p2key(self.v2d(grid_node))
            if self.t_performance_dict.has_key(grid_key):
                performance_dict[grid_key] = self.t_performance_dict[grid_key]
            else:
                red_grid.append(self.v2d(grid_node))
                # each flow has to be evaluated, so we don't need more flows
                # than possible iterations
                self.iterations += 1
                if self.iterations == self.max_iter:
                    break
        del(grid)
        
        # create subflows 
        subflows = [self.generate_subflow(self.flow_template, grid_node) for \
                                                          grid_node in red_grid]
        # execute subflows
        result_collections = self.execute_subflows(self.train_instances, 
                                                   subflows, self.runs)
        for grid_node, result in zip(red_grid, result_collections):
            key = self.p2key(grid_node)
            performance = result.get_average_performance(self.metric) - \
                                self.w * result.get_performance_std(self.metric)
            self.t_performance_dict[key] = performance
            performance_dict[key] = performance

        del subflows, result_collections
        
        if self.debug:
            print "results of grid evaluation"
            print performance_dict
        if not(self.iterations==self.max_iter):
            assert(chec_p==len(performance_dict)), \
                                          "Entries missing in performance dict!"
        # Determine the flow-parameterization that performed optimal with regard
        # to the specified metric on the grid
        if not(len(performance_dict)==0):
            # max sorted is there to get the same parameter
            best_parametrization, performance = \
                                      self.get_best_dict_entry(performance_dict)
        else:
            performance = -inf
            best_parametrization = "None"
        # the following ordering of comparison is necessary,
        # a performance result may be nan.
        old_step_size = self.step_size
        if performance>self.f_max:
            self.x_opt = self.get_vector(best_parametrization)
            self.f_max = performance
            # increasing step size
            self.step_size=self.step_size*self.up_scaling_factor
        elif performance == self.f_max:
            self.x_opt = self.get_vector(best_parametrization)
        else:
            # new scaling factor if we don't get an improvement
            self.step_size = self.step_size*self.scaling_factor
        # Search documentation - TODO: maybe only log relevant steps?
        self.search_history.append({"best_parameter": self.x_opt,
                                    "best_performance": self.f_max,
                                    "performance_dict": performance_dict,
                                    "step_size": old_step_size,
                                    "iterations": self.iterations})


# Specify special node names
_NODE_MAPPING = {"Grid_Search": GridSearchNode,
                 "Pattern_Search": PatternSearchNode}
