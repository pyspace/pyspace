""" Methods for sensor selection optimization algorithms

.. note:: The words *sensor* and *channel* are used as synonyms.

.. todo:: Adapt to new subflow concept for speed up via parallelization.
"""
import os
import random
from operator import itemgetter
from copy import deepcopy

import numpy

try:
    if map(int, __import__("scipy").__version__.split('.')) < [0,8,0]:
        from scipy.linalg.decomp import qr
    else:
        from scipy.linalg import qr
except:
    pass

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.resources.data_types.time_series import TimeSeries
from pySPACE.tools.filesystem import create_directory

# sensor ranking imports
from pySPACE.missions.nodes.source.external_generator_source\
import ExternalGeneratorSourceNode
from pySPACE.environments.chains.node_chain import NodeChain

# parallelization in evaluation
import sys
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing

import logging

class SensorSelectionBase(BaseNode):
    """ Template for nodes that select sensors
    
    This node implements the basic framework for nodes that select sensors.
    The train method has to be overwritten as it is the place for the specific
    selection procedures and criteria.
    
    **Parameters**
          :num_selected_sensors: Determines how many sensors are kept.
          
                    (*optional, default: 2*)
          
          :store:   In contrary to the base node, the default of this node
                    is to store the chosen sensors and rankings.

                    If the store parameter is set to True, one file named
                    "sensor_selection.txt" will be saved.
                    This text file holds the list of
                    chosen sensors with no particular order. If the
                    SensorSelectionRankingNode is used, another file called
                    "ordered_list_of_picks.txt" will be saved.
                    This is an ordered list of the
                    picks that were made due to the ranking.
                    E.g., in a "remove_1" setting, the
                    first sensor in the list is the first that was removed.
                    In a "add_1" setting it is the first one that was added.
                    Additionally, one file called "sensor_ranking.txt"
                    will be created. This file is a merge of the aforementioned.
                    The first entries are the selected channels
                    that can't be ranked
                    in alphabetical order.
                    Then come the (de-)selected sensors in order of
                    descending relevance. 
                    
                    (*optional, default: True*)

    The following shows a complete example using the
    SensorSelectionRankingNode to illustrate, how nodes of this type
    can be used. In this case, the number of sensors is first reduced to 8
    removing 2 sensors at a time, than increased back to 16 adding 4 at a
    time. 

    **Exemplary Call**
    
    .. code-block:: yaml
    
        - 
            node : Time_Series_Source
        -
            node : CV_Splitter
        -
            node : FFT_Band_Pass_Filter
            parameters : 
                    pass_band : [0.0, 4.0]
                    keep_in_history : True
        -
            node : Sensor_Selection_Ranking
            parameters :
                ranking : Remove_One_Performance_Ranking
                num_selected_sensors : 8
                recast_method : remove_2
                ranking_spec :
                    pool_size : 2
                    std_weight : 1
                    flow :
                        -
                            node : CV_Splitter
                        -
                            node : Time_Domain_Features
                        -
                            node : 2SVM
                        -
                            node : Classification_Performance_Sink
        -
            node : Sensor_Selection_Ranking
            parameters :
                ranking : Add_One_Performance_Ranking
                num_selected_sensors : 16
                recast_method : add_4
                store : True
                ranking_spec :
                    std_weight : 1
                    pool_size : 2
                    flow :
                        -
                            node : CV_Splitter
                        -
                            node : Time_Domain_Features
                        -
                            node : 2SVM
                        -
                            node : Classification_Performance_Sink
        -
            node : Time_Domain_Features
        -
            node : 2SVM
        -
            node : Classification_Performance_Sink
        
    :Author: Mario Krell & David Feess 2011/09/23
    :Created: 2011/09/23
    """
    def __init__(self, num_selected_sensors = 2, store = True, **kwargs):
        super(SensorSelectionBase, self).__init__(store = store,**kwargs)
        # mapping of old parameter to new generalized one (electrodes->sensors)
        if 'num_selected_electrodes' in kwargs:
            num_selected_sensors = kwargs['num_selected_electrodes']
            self._log("Please use 'num_selected_sensors' instead of 'num_selected_electrodes' in sensor selection node!", level=logging.WARNING)
        self.set_permanent_attributes(num_selected_sensors=num_selected_sensors,
                                      channel_names = None)

    def is_trainable(self):
        """ Returns whether this node is trainable. """
        return True
    
    def is_supervised(self):
        """ Returns whether this node requires supervised training """
        return True
    
    def _train(self,data,label):
        """ This method has to be overwritten by the different sensor selection nodes """
        raise NotImplementedError("Your method should overwrite the train method!")
    
    def _execute(self, data):
        """ Project the data onto the selected channels. """
        if (getattr(self,'add_remove','None') == 'add'):
            # "add channels" case - base on historic data that contained all channels 
            initial_data = data.history[-1]
        else:
            # "remove channels" case - work on what's left
            initial_data = data
            
        projected_data = initial_data[:, self.selected_indices]
        new_data = TimeSeries(projected_data, self.selected_channels,
                              data.sampling_frequency, data.start_time,
                              data.end_time, data.name, data.marker_name)
        new_data.inherit_meta_from(data)
        return new_data
        
    def store_state(self, result_dir, index=None): 
        """ Store the names of the selected sensors into *result_dir* """
        if self.store:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            if not index == None:
                node_dir += "_%i" % int(index)
            create_directory(node_dir)
             
            # This node stores which sensors have been selected
            name = "%s_sp%s.txt" % ("sensor_selection", self.current_split)
            result_file = open(os.path.join(node_dir, name), "w")
            result_file.write(str(self.selected_channels))
            result_file.close()


class SensorSelectionRankingNode(SensorSelectionBase):
    """ Iteratively choose sensors depending on a ranking function
    
    This node collects the training data and generates a ranker. Then it
    evaluates different sub-/supersets of the current set of sensors using
    this ranker, and dismisses or adds sensors according to the ranking result. 
    
    The ranking function can (and often will) in fact consist of the evaluation
    of an entire classification flow. After that, e.g., achieved performance or
    the values of certain classifier parameters can be used as ranking.
    See the PerformanceRanker and CoefficientRanker classes for details.

    .. note::
        The code of this node is partly copied from parameter optimization node.
    
    **Parameters**
        :num_selected_sensors: Determines how many sensors are kept.
          
          :ranking: String specifying the desired method for the ranking of
              sensors. The string must be known to the create_ranker method.
              So far implemented:
              
              * "Remove_One_Performance_Ranking"
                  Based on the current set of sensors, the ranking of one
                  sensor is computed by removing it and evaluating the
                  performance based on the remaining sensors. The
                  classification node chain has to be specified. One would
                  typically use this together with a "remove_*" ranking_spec
                  (see below) to implement a "recursive backwards
                  elimination".
                  
              * "Add_One_Performance_Ranking"
                  This Ranker takes the current set as fixed and extends it by
                  previously dismissed sensors. This implementation gains
                  access to the previously dismissed channels through the
                  data.history. Thus, in order for this to work, make sure
                  that a previous node in the flow (that works on a larger set 
                  of sensors) has set "keep_in_history : True". See the
                  example flow in the documentation of SensorSelectionBase.
                  The current set of sensors plus one of the dismissed
                  sensors will be evaluated. This will be repeated for each
                  of the previously dismissed sensors. The ranking results
                  from the classification performance. One would typically use
                  this together with a "add_n" ranking_spec to re-add the n best
                  performing sensors.
                  
              * "Coefficient_Ranking" 
                  performs a classification flow (which has
                  to be specified in ranking_spec) using all currently active 
                  sensors. The actual ranking is then provided by the
                  classifier's get_sensor_ranking method.
          
          :ranking_spec: Arguments passed to the ranker upon creation. Often
              contains a classification flow of some sort.
              
          :recast_method: Determines how the set of sensors is altered based
              on the ranking. Most commonly, the worst sensor will be removed
              until the desired number of sensors is reached. Alternatively,
              n sensors at a time could be removed. When using
              "Add_One_Performance_Ranking" sensors can even be added. Syntax
              is {add/remove}_n, e.g., add_3, remove_4.
              
              NB: When performing performance ranking, remove_* should be used
                  only together with the Remove_One_Performance_Ranking, and
                  add_* should only be used with Add_One_Performance_Ranking.
              
              (*optional, default: remove_1*)

    
    **Exemplary Call**
    
    See the description of SensorSelectionBase for an example usage of this node.
    
    :Author: Mario Krell (mario.krell@dfki.de) & David Feess
    :Created: 2011/09/23
    """
    def __init__(self, ranking_spec,ranking, recast_method='remove_1', **kwargs):
        super(SensorSelectionRankingNode, self).__init__(**kwargs)
        self.set_permanent_attributes(ranking = ranking,
                                      ranking_spec = ranking_spec,
                                      recast_method = recast_method,
                                      channel_names=None,
                                      training_data=None,
                                      add_remove=None,
                                      picked_sensors=[])

    def create_ranker(self,ranking_name,ranking_spec):
        """ A ranking method should return a sorted list of tuples (sensor, score),
        Where the first element is the worst sensor with the lowest score.
        Thus, in cases where a high score denotes a bad sensor: swap sign!
        """
        if ranking_name == "Remove_One_Performance_Ranking":
            return RemoveOnePerformanceRanker(ranking_spec=ranking_spec)
        elif ranking_name == "Coefficient_Ranking":
            return CoefficientRanker(ranking_spec=ranking_spec,
                                     run_number=self.run_number)
        elif ranking_name == "Add_One_Performance_Ranking":
            return AddOnePerformanceRanker(ranking_spec=ranking_spec)
        else:
            self._log("Ranking algorithm '%s' is not available!" % ranking_name,
                      level=logging.CRITICAL)
            raise NotImplementedError(
                "Ranking algorithm '%s' is not available!" % ranking_name)

    def _train(self, data, label):
        """ Save the *data*
        
        The actual training is done after all data has been collected.
        """
        if self.training_data is None:
            self.training_data = []
            self.channel_names = data.channel_names
        self.training_data.append((data, label))
    
    def _stop_training(self, debug=False):
        """ Recast sensor set """
        if self.load_path is not None:
            self.replace_keywords_in_load_path()
            self.picked_sensors = \
                __import__("yaml").load(open(self.load_path).read())
        # Parse desired recast method 
        [self.add_remove, add_remove_n] = self.recast_method.split('_')
        add_remove_n = int(add_remove_n)
        
        self.ranker=self.create_ranker(ranking_name = self.ranking,ranking_spec=self.ranking_spec)

        if self.add_remove == 'remove':
            self.remove_sensors(add_remove_n)
        elif self.add_remove == 'add':
            self.add_sensors(add_remove_n)
        

    def remove_sensors(self, n):
        """Iteratively remove n sensors from the current (sub)set"""
        # Memorize which channels are left through their channel_names list index. 
        active_elements = range(len(self.channel_names))
        # If a list of already picked elements has been loaded, remove those:
        for prev_dismissed in self.picked_sensors:
            active_elements.remove(self.channel_names.index(prev_dismissed))
        # Remove elements one-by-one until we retain only the requested number
        # of elements.
        while len(active_elements) > self.num_selected_sensors:
            self._log("%s active sensors remaining." % len(active_elements))
            selected_channels = [self.channel_names[i] for i in active_elements]
            # Ranker receives the complete set of active sensors
            ranking=self.ranker.get_ranking(selected_channels=selected_channels,
                                            training_data=self.training_data)
            # remove the worst n from active elements
            for i in range(n):
                # bad sensors are in front in the ranking, because omitting
                # them  results in high performance
                dismissed_sensor = ranking[i][0]
                self.picked_sensors.append(dismissed_sensor)
                active_elements.remove(self.channel_names.index(dismissed_sensor))
                self._log("Dismissing sensor %s." % dismissed_sensor)

                # Save the picked sensors  in every round as failsafe
                node_dir = os.path.join(self.temp_dir, self.__class__.__name__)
                create_directory(node_dir)
                # if not index == None:
                #     node_dir += "_%i" % int(index)
                name = "%s_sp%s.txt" % ("ordered_list_of_picks", self.current_split)
                result_file = open(os.path.join(node_dir, name), "w")
                result_file.write(str(self.picked_sensors))
                result_file.close()
        self.selected_indices = active_elements

        self.selected_channels=[]
        for i in range(len(ranking)):
            sensor = ranking[-1-i][0]
            if self.channel_names.index(sensor) in self.selected_indices:
                self.selected_channels.append(sensor)

    def add_sensors(self, n):
        """Iteratively add n sensors to the current subset"""
        # first generate actual training data - get dataset from history, because
        # also channels that had already been deselected are required here.
        # This step assumes that the last history entry originates from a node
        # before a channel selection that decreased the number of channels
        complete_training_data = [(x[0].history[-1], x[1]) for x in self.training_data]
        old_channels = complete_training_data[0][0].channel_names
        # active elements to start with are the channels left in training_data
        # but we need their index with respect to the complete_training_data
        active_elements = [old_channels.index(x)
                               for x in self.training_data[0][0].channel_names]
        # Behavior if load_path is used is not yet implemented
        if self.picked_sensors is not []:
            self._log("Behavior if load_path is used is not yet implemented for add_sensors!",
                level=logging.CRITICAL)
        while len(active_elements) < self.num_selected_sensors:
            self._log("%s active sensors remaining." % len(active_elements))
            selected_channels = [old_channels[i] for i in active_elements]
            # Ranker receives the complete set of sensors
            ranking=self.ranker.get_ranking(selected_channels=selected_channels,
                                    training_data=complete_training_data)
            # add the best n to active elements
            for i in range(n):
                # good sensors are in front in the ranking
                chosen_sensor = ranking[i][0]
                self.picked_sensors.append(chosen_sensor)
                active_elements.append(old_channels.index(chosen_sensor))
                self._log("Readding sensor %s." % chosen_sensor)
        self.selected_indices = active_elements
        self.selected_channels = \
                [old_channels[index] for index in self.selected_indices]
                
    def store_state(self, result_dir, index=None): 
        """ Store the names of the selected sensors into *result_dir* """
        super(SensorSelectionRankingNode, self).store_state(result_dir, 
                                                                  index)
        if self.store:
            node_dir = os.path.join(result_dir, self.__class__.__name__)
            if not index == None:
                node_dir += "_%i" % int(index)
            # Further, we also store in which order sensors were
            # selected/deselected.
            # This list is in the order the sensors were picked.
            # -> remove: worst first, add: best first 
            name = "%s_sp%s.txt" % ("ordered_list_of_picks", self.current_split)
            result_file = open(os.path.join(node_dir, name), "w")
            result_file.write(str(self.picked_sensors))
            result_file.close()
            
            # Last but not least, we get a ranking by joining the 2 above lists
            best = [x for x in self.selected_channels if x not in self.picked_sensors]
            if len(best)==len(self.selected_channels):
                # remove case: order of picks = bad -> good
                best = best + self.picked_sensors[::-1]
            else:
                # add case: order of picks = good -> bad
                # the best sensors are sorted alphabetically, because there's
                # no information in the order. FIXTHIS
                best.sort()
                best = best + self.picked_sensors
            name = "%s_sp%s.txt" % ("sensor_ranking", self.current_split)
            result_file = open(os.path.join(node_dir, name), "w")
            result_file.write(str(best))
            result_file.close()


class SensorSelectionSSNRNode(SensorSelectionBase):
    """ Select sensors based on maximizing the SSNR
    
    This node searches for an optimal sensor configuration for a given number
    of sensors. It can use different meta-heuristics (like evolutionary 
    algorithms or recursive backward elimination) for this search. The
    objective function that shall be maximized can be configured
    and is based on the signal to signal-plus-noise ratio (SSNR).
    
    **Parameters**
          :erp_class_label: Label of the class for which an ERP should be evoked. 
               For instance "Target" for a P300 oddball paradigm.
    
          :num_selected_sensors: Determines how many sensors are kept.

          :retained_channels: The number of pseudo-channels that are kept after 
                xDAWN filtering when using virtual sensor space. 
                Even though this node only selects sensors 
                and does no spatial filtering, this information is relevant since
                the SSNR after xDAWN spatial filtering is used in objective 
                functions in virtual sensor space and the SSNR depends 
                on the number of pseudo-channels. If one does not use virtual
                sensor space, this information can be ignored.
                
                (*optional, default: num_selected_sensors*)
              
          :search_heuristic: The search heuristic that is used to search an
              optimal sensor configuration. Can be either "evolutionary_search"
              or "recursive_backward_elimination".
              
              (*optional, default: "evolutionary_algorithm"*)
              
          :objective_function: The objective function that is used to determine
               which sensor selection are well suited and which less suited.
               Available objective functions are "ssnr_vs" (the signal to 
               signal-plus-noise ratio in virtual sensor space), "ssnr_as" 
               (the signal to signal-plus-noise ratio in actual sensor space),
               "ssnr_vs_loeo" (the minimum signal to signal-plus-noise ratio 
               in virtual sensor space when one of selected sensors wouldn't
               be present)
                
                (*optional, default: "ssnr_vs"*)
                
          :population_size: The number of individuals of which one generation 
                of the EA consists of. Each individual corresponds to one 
                sensor configuration.
                
                (*optional, default: 20*)
                
          :num_survivors: The number of individuals which survive at the end of
                a generation of the EA. The ratio of num_survivors to population_size 
                determines the selection pressure.
                
                (*optional, default: *8*)
                
          :mutant_ratio: The ratio of the next generation that consist of
                survivors that a underwent a mutation.
                
                (*optional, default: *0.3*)
          
          :crossover_ratio: The ratio of the next generation that consist of
                offspring of two survivors that were crossovered.
                
                (*optional, default: *0.3*)

          :iterations: The number of sensor configurations that are evaluated 
              before the EA terminates. The larger this value, the better 
              performance (higher SSNR) can be expected but the computation time 
              increases, too.

              (*optional, default: *1000*)

    **Exemplary Call**
    
    .. code-block:: yaml
    
            -
                node : Sensor_Selection_SSNR
                parameters :
                     erp_class_label : "'Target'"
                     num_selected_sensors : 8
                     retained_channels : 4
                     search_heuristic : "'evolutionary_algorithm'"
                     iterations : 1000
                     mutant_ratio : 0.3
                     crossover_ratio : 0.3
                     diversity_support : 0.0
                     objective_function : "'ssnr_vs'"

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2011/08/22
    """
    
    def __init__(self, num_selected_sensors, erp_class_label="Target",
                 retained_channels=None, search_heuristic="evolutionary_algorithm",
                 objective_function="ssnr_vs", population_size=20, num_survivors=8, 
                 mutant_ratio=0.3, crossover_ratio=0.3, iterations=1000,
                 **kwargs):
        super(SensorSelectionSSNRNode, self).__init__(num_selected_sensors=num_selected_sensors,**kwargs)
        # Check parameters
        search_heuristics = ["evolutionary_algorithm", 
                             "recursive_backward_elimination"]

        assert search_heuristic in search_heuristics, \
            "Unknown search heuristic %s. Must be in %s." % (search_heuristic,
                                                             search_heuristics)

        objective_functions = ["ssnr_vs", "ssnr_as", "ssnr_vs_test"]

        assert objective_function in objective_functions, \
            "Unknown objective function %s. Must be in %s." % (objective_function,
                                                               objective_functions)

        from pySPACE.missions.nodes.spatial_filtering.xdawn import SSNR
        # Set permanent attributes
        self.set_permanent_attributes(# Label of the class for which an ERP should be evoked.
                                      erp_class_label = erp_class_label,
                                      # Object for handling SSNR related calculations
                                      ssnr = SSNR(erp_class_label, retained_channels),

                                      num_selected_sensors=num_selected_sensors,
                                      search_heuristic=search_heuristic,
                                      objective_function=objective_function,
                                      population_size=population_size,
                                      num_survivors=num_survivors,
                                      mutant_ratio=mutant_ratio,
                                      crossover_ratio=crossover_ratio,
                                      iterations=int(iterations))

    def _train(self, data, label):
        """ Train node on given example *data* for class *label*. """
        # If this is the first data sample we obtain
        if self.channel_names == None:
            self.channel_names = data.channel_names

        self.ssnr.add_example(data, label)
    
    def _stop_training(self, debug=False):        
        # Determine objective function
        if self.objective_function == "ssnr_vs":
            objective_function = lambda selection: self.ssnr.ssnr_vs(selection)
        elif self.objective_function == "ssnr_vs_test":
            objective_function = lambda selection: self.ssnr.ssnr_vs_test(selection)
        elif self.objective_function == "ssnr_as":
            objective_function = lambda selection: self.ssnr.ssnr_as(selection)

        # Determine search heuristic
        if self.search_heuristic == "evolutionary_algorithm":
            heuristic_search = \
                EvolutionaryAlgorithm(self.ssnr.X.shape[1], 
                                       self.num_selected_sensors,
                                       self.population_size, self.num_survivors,
                                       self.mutant_ratio, self.crossover_ratio)
        elif self.search_heuristic == "recursive_backward_elimination":
            heuristic_search = \
                RecursiveBackwardElimination(total_elements=self.ssnr.X.shape[1], 
                                               num_selected_sensors=self.num_selected_sensors)

        # Search for a set of sensors that yield a maximal SSNR using
        # heuristic search
        self.selected_indices = \
            heuristic_search.optimize(objective_function, self.iterations)

        self.selected_channels = \
                [self.channel_names[index] for index in self.selected_indices]


#==============================================================================#


def evaluate_sensor_selection(cns, flow, metric, w, sensor_identifier, 
                                 training_data, runs=1):
    """ Execute the evaluation flow """
    # Getting together the two evaluation functions without self variables
    node_sequence = [ExternalGeneratorSourceNode(),
                     BaseNode.node_from_yaml(cns)]
    
    # For all nodes of the flow
    for sub_node_spec in flow:
        # Use factory method to create node
        node_obj = BaseNode.node_from_yaml(sub_node_spec)

        # Append this node to the sequence of node
        node_sequence.append(node_obj)
    
    # Check if the nodes have to cache their outputs
    for index, node in enumerate(node_sequence):
        # If a node is trainable, it uses the outputs of its input node
        # at least twice, so we have to cache.
        if node.is_trainable():
            node_sequence[index - 1].set_permanent_attributes(caching=True)
        # Split node might also request the data from their input nodes
        # (once for each split), depending on their implementation. We
        # assume the worst case and activate caching
        if node.is_split_node():
            node_sequence[index - 1].set_permanent_attributes(caching=True)

    flow = NodeChain(node_sequence)
        
    for run in range(runs):
        flow[-1].set_run_number(run)
        # Set input data
        flow[0].set_generator(training_data)
        # For every split of the data
        while True: # As long as more splits are available
            # Compute the results of the flow for the current split
            # by calling the method on its last node
            flow[-1].process_current_split()

            # If no more splits are available
            if not flow[-1].use_next_split():
                break
        # reset flow, collection is kept for the different runs
        for node in flow:
            node.reset()
    # Determine performance of the flow and store it in dict
    result_collection = flow[-1].get_result_dataset()
    performance = \
        result_collection.get_average_performance(metric) \
        - w * result_collection.get_performance_std(metric)
    return (sensor_identifier, performance)


#==============================================================================#


class PerformanceRanker(object):
    """ Rank sensors by performance after evaluating classification flows
    
    This class provides the functionality to evaluate different classification
    flows. Every flow has an sensor_identifier string associated.
    Afterwards, the classification performances (or a derived value - see 
    std_weight parameter) are sorted and returned together with the associated
    identifier. 
    
    .. note:: Classification performances are multiplied with (-1).
              In this way, high performances appear first in the sorted results.
    
    The flows differ in the sensors/channels that are used by using
    multiple Channel Name Selection (CNS) nodes. The way how these CNS nodes are
    generated, however, is specific for every particular selection procedure
    (such as "remove one backwards elimination" vs. "add one forward assembly").
    The actual generation of the flows happens in generate_cns_nodes. This
    template class only has a dummy for that method - overwrite it in your
    ranker! See RemoveOnePerformanceRanker or AddOnePerformanceRanker for
    examples.
    
    **Parameters**
        :flow: The processing chain (YAML readable). Usually, the flow
            will at least consist of a CV-Splitter, a classifier, and a
            :class:`~pySPACE.missions.nodes.sink.classification_performance_sink.ClassificationPerformanceSink`.
            See the documentation of :class:`SensorSelectionBase` for an example.

        :metric: The :ref:`metric <metrics>`
            for the classification performance used for
            the calculation of the ranking,
            if a performance value is used.

            (*optional, default: Balanced_accuracy*)

        :std_weight:
            As a result of cross validation often more than one
            performance result (*p*) per sensor set is calculated.
            The score (*s*) of one particular constellation is thus computed
            by calculating

            .. math:: s = mean(p) - \\text{std\_weight} \\cdot \\text{std\_dev}(p)

            Hence, for std_weight = 0 the mean is used. With increasing
            std_weight large spreads get penalized more strongly.

            (*optional, default: 0*)

        :runs: May be specified to perform multiple runs (and thus different
            CV-Splits)

            (*optional, default: 1*)

        :pool_size: May be specified to achieve parallelization of the
            classification subflow as normally only the main flow is parallelled.

            .. note:: Currently a pool size larger than 1 will not work with the MulticoreBackend,
                      because multiprocessing can't be nested.
                      Use loadl backend instead or no pool size!

            .. todo:: Distribute subflows with the subflowhandler using backend specific parallelization.

            (*optional, default: 1*)
    
    :Author: Mario Krell (mario.krell@dfki.de) & David Feess
    :Created: 2011/09/23
    """
    def __init__(self,ranking_spec):
        self.flow = ranking_spec["flow"]
        self.metric = ranking_spec.get("metric","Balanced_accuracy")
        self.std_weight = ranking_spec.get("std_weight", 0)
        self.runs =ranking_spec.get("runs", 1)
        self.pool_size = ranking_spec.get("pool_size", 1)
        
    def get_ranking(self,selected_channels, training_data):
        """Compute the ranking of the selected channels."""
        # to get the ranking, classification flows have to be evaluated on 
        # different subsets of the channels. These subsets are generated by
        # different channel name selection nodes *cns_nodes*
        cns_nodes = self.generate_cns_nodes(selected_channels, training_data)
        ranking=[]
        # one core case
        if self.pool_size==1:
            for sensor_identifier,cns_node in cns_nodes:
                sensor,performance = \
                    evaluate_sensor_selection(cns=cns_node,
                                                 flow=self.flow,
                                                 metric=self.metric,
                                                 w=self.std_weight,
                                                 sensor_identifier=sensor_identifier,
                                                 training_data=training_data,
                                                 runs=self.runs)
                ranking.append((sensor,-performance))
        # multiple cores: parallel case
        else:
            pool = processing.Pool(processes=self.pool_size)
            # This won't work with mcore
            results = [pool.apply_async(func=evaluate_sensor_selection,
                kwds={"cns":cns_node,"flow":self.flow,
                      "metric":self.metric,"w":self.std_weight,
                      "sensor_identifier":sensor_identifier,
                      "training_data":training_data,"runs":self.runs})
                for sensor_identifier,cns_node in cns_nodes]
            pool.close()
            # self._log("Waiting for parallel processes to finish")
            # this is not a node! there's no self._log here!
            pool.join()
            for result in results:
                sensor,performance =result.get()
                ranking.append((sensor,-performance))
            del(pool)
        # sort by performance before return
        # NB: Performances have been multiplied by (-1), s.t. high performances
        # appear first in the sorted lists.
        return sorted(ranking,key=lambda t: t[1])

    def generate_cns_nodes(self, selected_channels, training_data):
        """ This method has to be overwritten by the different sensor selection nodes """
        raise NotImplementedError("Your method should overwrite the "
                                  "generate_cns_nodes method in your Ranker!")


class RemoveOnePerformanceRanker(PerformanceRanker):
    """ Rank sensors by evaluating if classification performance drops without them
    
    Consider a set of n sensors. This ranker will always remove one sensor
    creating n-1 sized subsets. Every size n-1 subset is evaluated.
    
    NB: high performance == Unimportant sensor == good sensor to remove 
    
    See the description of PerformanceRanker for the required parameters.
    
    :Author: Mario Krell (mario.krell@dfki.de) & David Feess
    :Created: 2011/09/23
    """
    def __init__(self,**kwargs):
        super(RemoveOnePerformanceRanker, self).__init__(**kwargs)

    def generate_cns_nodes(self, selected_channels, training_data):
        """ Generate Channel Name Selection Nodes that use the current channels minus 1
        .. todo:: training_data parameter is not necessary!
        """
        # generates the list with cns nodes, each of which has a different
        # sensor removed. This function is specifically what makes this the
        # "remove_one"-ranker
        cns_nodes=[]
        for element in selected_channels:
            # Remove element temporarily and create channel name selector node
            # that selects all the remaining
            channels=deepcopy(selected_channels)
            channels.remove(element)
            cns_nodes.append((element,{'node': 'Channel_Name_Selector', 
                                       'parameters': {'selected_channels': channels}}))
        return cns_nodes

        
class AddOnePerformanceRanker(PerformanceRanker):
    """ Rank sensors by evaluating performance increase on usage
    
    Consider a set N of sensors and a fixed subset K of the sensors in N.
    This ranker will always add one sensor of N\K to K 
    creating k+1 sized subsets. Every subset is than evaluated.
    The score of the added sensors is determined by the classification performance, s.t.
    high performance == good sensor to add
    
    See the description of PerformanceRanker for the required parameters.
    
    :Author: Mario Krell (mario.krell@dfki.de) & David Feess
    :Created: 2011/09/23
    """
    def __init__(self,**kwargs):
        super(AddOnePerformanceRanker, self).__init__(**kwargs)

    def generate_cns_nodes(self, selected_channels, training_data):
        """Generate Channel Name Selection Nodes that use the current channels plus 1"""
        # generates the list with cns nodes, each of which has a different
        # sensor added. This function is specifically what makes this the
        # "add_one"-ranker
        channels_to_pick_from = [x for x in training_data[0][0].channel_names 
                                            if x not in selected_channels]

        cns_nodes=[]
        for element in channels_to_pick_from:
            # Remove element temporarily and create channel name selector node
            # that selects all the remaining
            channels=deepcopy(selected_channels)
            channels.append(element)
            cns_nodes.append((element,{'node': 'Channel_Name_Selector', 
                                       'parameters': {'selected_channels': channels}}))
        return cns_nodes


class CoefficientRanker(object):
    """ Get a ranking from the second last processing node 
    
    This ranking is given by this node,
    by adding up channel weights of linear classifiers or spatial filters.
    The details remain to the used node (last one in the node chain before
    sink node) and its method *get_sensor_ranking*.
    
    **Parameters**

        :flow: The classification flow (YAML readable). Usually, the flow
            will at least consist of a CV-Splitter, a classifier , and a 
            Classification_Performance_Sink. See the documentation of 
            SensorSelectionBase for an example.

            (*optional, default: 1*)
    """
    def __init__(self,ranking_spec,run_number):
        self.flow = ranking_spec["flow"]
        self.run_number = run_number

    def get_ranking(self,selected_channels, training_data):
        cns_node = {'node': 'Channel_Name_Selector',
                    'parameters': {'selected_channels': selected_channels}}
        # code copy from evaluate_sensor_selection
        node_sequence = [ExternalGeneratorSourceNode(),
                     BaseNode.node_from_yaml(cns_node)]
        
        # For all nodes of the flow
        for sub_node_spec in self.flow:
            # Use factory method to create node
            node_obj = BaseNode.node_from_yaml(sub_node_spec)
    
            # Append this node to the sequence of node
            node_sequence.append(node_obj)
        
        # Check if the nodes have to cache their outputs
        for index, node in enumerate(node_sequence):
            # If a node is trainable, it uses the outputs of its input node
            # at least twice, so we have to cache.
            if node.is_trainable():
                node_sequence[index - 1].set_permanent_attributes(caching=True)
            # Split node might also request the data from their input nodes
            # (once for each split), depending on their implementation. We
            # assume the worst case and activate caching
            if node.is_split_node():
                node_sequence[index - 1].set_permanent_attributes(caching=True)
    
        flow = NodeChain(node_sequence)
        flow[-1].set_run_number(self.run_number)
        flow[0].set_generator(training_data)
        flow[-1].process_current_split()
        # Since the last node is the sink node the second last is expected
        # to give the ranking
        # It can be a linear classification node or a spatial filter  
        result = flow[-2].get_sensor_ranking()
        del(flow)
        return result


#==============================================================================# 


class EvolutionaryAlgorithm(object):
    """ Black-box optimization using an evolutionary algorithm
    
    This implementation is tailored for the specific case that one wants to
    select M out of N elements and is looking for the M elements that maximize
    an objective function. For simplicity, it is assumed, the one works on the
    indices, i.e. the N-elementary set is {0,1,...,N-1}. 
    
    One may either provide the objective function to the object and let it
    autonomously optimize this function or use its "ask and tell" interface
    and keep control over the optimization procedure.
    
    **Parameters**
          :total_elements: The number of total elements (i.e. N)
          
          :num_selected_elements: The number of elements to be selected (i.e. M)
    
          :population_size: The number of individuals of which one generation 
                of the EA consists of. 
                
          :num_survivors: The number of individuals which survive at the end of
                a generation. The ratio of num_survivors to population_size 
                determines the selection pressure.
                
          :mutant_ratio: The ratio of the next generation that consist of
                survivors that a underwent a mutation.
          
          :crossover_ratio: The ratio of the next generation that consist of
                offspring of two survivors that were crossovered.
    """
    
    def __init__(self, total_elements, num_selected_elements, 
                 population_size, num_survivors, mutant_ratio, crossover_ratio):
        assert mutant_ratio + crossover_ratio <= 1.0
        self.total_elements = total_elements
        self.num_selected_elements = num_selected_elements
        self.population_size = population_size
        self.num_survivors = num_survivors
        self.mutant_ratio = mutant_ratio
        self.crossover_ratio = crossover_ratio

        # Create population to be used in evolutionary algorithm
        self.population = [random.sample(range(self.total_elements), 
                                         self.num_selected_elements)
                                 for i in range(self.population_size)]
        self.currentIndivudualIndex = 0
            
        self.fitnesses = []
        
        self.max_fitness = -numpy.inf
        self.best_individual = None
        
    def optimize(self, objective_function, evaluations):
        """ Search for maximum of objective_function
        
        Search for maximum of the given *objective_function*. Restrict number of 
        evaluations of objective function to *evaluations*.
        """
        for i in range(evaluations):
            # Fetch next configuration from evolutionary algorithm
            selected_elements = self.get_current_elements()

            # Compute fitness for this configuration
            fitness = objective_function(selected_elements)
            
            # Tell EA the fitness of configuration
            self.tell_fitness(fitness)
            
        # Return best configuration found
        return self.get_best_elements()
        
        
    def get_best_elements(self):
        """ Return the individual with the maximal fitness. """
        return self.best_individual
    
    def get_current_elements(self):
        """ Return the currently active individual. """
        return self.population[self.currentIndivudualIndex]
    
    def tell_fitness(self, fitness):
        """ Add a fitness sample for the current individual. """
        self.fitnesses.append((fitness, self.population[self.currentIndivudualIndex]))
        
        # If we have found an individual that gives rise to
        # the maximally fitness found so far: 
        if fitness > self.max_fitness:
            # Remember this sensor configuration and its SSNR
            self.max_fitness = fitness
            self.best_individual = self.population[self.currentIndivudualIndex]
        
        if self.currentIndivudualIndex + 1 == len(self.population):
            # Evaluation of a generation is finished.       
            # Determine survivors            
            survivors = map(itemgetter(1), 
                            sorted(self.fitnesses, reverse=True)[:self.num_survivors])
            # Create next generation's population by randomly picking survivors
            # of the previous generation and optionally mutate them
            self.population = []
            for i in range(self.population_size):
                r = random.random()
                if r < self.mutant_ratio: # Mutation
                    self.population.append(self._mutate(random.choice(survivors)))
                elif r < self.mutant_ratio + self.crossover_ratio: # Crossover
                    parent1, parent2 = random.sample(survivors, 2)
                    self.population.append(self._crossover(parent1, parent2))
                else: # Cloning
                    self.population.append(random.choice(survivors))
                                
            self.currentIndivudualIndex = 0
            self.fitnesses = []
        else:
            self.currentIndivudualIndex += 1
                    
    def _mutate(self, individual):
        """ Mutate the given individual with the given probability. """
        individual = list(individual) 
        
        # Replace one randomly chosen currently activate element by
        # an inactive element
        inactive_elements = \
            [element for element in range(self.total_elements)
                                            if element not in individual]
        individual[random.choice(range(len(individual)))] = \
                                    random.choice(inactive_elements)
        
        return individual

    def _crossover(self, parent1, parent2):
        """ Create offspring by crossover of two parent individuals. """
        elements = \
            set([element for element in range(self.total_elements)
                            if element in parent1 or element in parent2])
        
        individual = random.sample(elements, len(parent1))
        
        return individual

class RecursiveBackwardElimination(object):
    """ Black-box optimization using recursive backward elimination
    
    This implementation is tailored for the specific case that one wants to
    select M out of N elements and is looking for the M elements that maximize
    an objective function. For simplicity, it is assumed, the one works on the
    indices, i.e. the N-elementary set is {0,1,...,N-1}. 
    
    One may either call *optimize* which returns a set of M sensors that are
    selected using recursive backward elimination or call *rank* which returns
    a ranking of all sensors. For *rank* the specific value of M is not
    relevant and may be omitted.
    
    **Parameters**
          :total_elements: The number of total elements (i.e. N)
          
          :num_selected_elements: The number of elements to be selected (i.e. M)
    """
    
    def __init__(self, total_elements, num_selected_elements=None):
        self.total_elements = total_elements
        self.num_selected_elements = num_selected_elements
        
    def optimize(self, objective_function, *args, **kwargs):
        """ Search for an optimal configuration consisting of M elements """
        active_elements = range(self.total_elements)
        # Remove elements one-by-one until we retain only the requested number
        # of elements.
        while len(active_elements) > self.num_selected_elements:
            # Compute the performance that is obtained when one of the remaining
            # elements is removed
            configuration_performance = []
            for element in active_elements:
                # Remove element temporarily and determine performance
                active=deepcopy(active_elements)
                active.remove(element)
                configuration_performance.append((objective_function(active),
                                                  random.random(), # Break ties randomly 
                                                  element))
            # Remove element which makes the objective function maximal when
            # removed permanently
            dismissed_sensor = max(configuration_performance)[2]
            active_elements.remove(dismissed_sensor)
            
        # Return the selected sensors
        return active_elements
        
    def rank(self, objective_function, *args, **kwargs):
        """ Rank the elements. """
        ranking = []
        active_elements = range(self.total_elements)
        # Remove elements one-by-one. Elements which are removed early 
        # come last in the ranking.
        while len(active_elements) > 1:
            # Compute the performance that is obtained when one of the remaining
            # elements is removed
            configuration_performance = []
            for element in active_elements:
                # Remove element temporarily and determine performance
                active=deepcopy(active_elements)
                active.remove(element)
                configuration_performance.append((objective_function(active),
                                                  random.random(), # Break ties randomly 
                                                  element))
            # Remove element which makes the objective function maximal when
            # removed permanently
            dismissed_sensor = max(configuration_performance)[2]
            active_elements.remove(dismissed_sensor)
            ranking.append(dismissed_sensor)
        
        # Append remaining (i.e. best sensor)
        ranking.append(active_elements[0])
        # Return the ranking
        return reversed(ranking)


_NODE_MAPPING = {"Sensor_Selection_SSNR" : SensorSelectionSSNRNode,
                "Sensor_Selection_Ranking" : SensorSelectionRankingNode,
                "Electrode_Selection_SSNR" : SensorSelectionSSNRNode,
                "Electrode_Selection_Ranking" : SensorSelectionRankingNode}
