""" Select only a part of the instances

.. todo: group instance selectors
"""

import random
import logging
from collections import defaultdict

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.tools.memoize_generator import MemoizeGenerator


class InstanceSelectionNode(BaseNode):
    """Retain only a certain percentage of the instances

    The node InstanceSelectionNode forwards only
    *train_percentage_selected* percent of the training instances passed to
    him to the successor node and only
    *test_percentage_selected* percent of the test instances. The forwarded 
    instances are selected randomly but so that the class ratio is kept.

    If *reduce_class* is used, only the chosen class is reduced, without
    keeping the class ratio. So the total mount of reduced data does not match
    the percentage values.
    
    **Parameters**
        :train_percentage_selected:
            The percentage of training instances which
            is forwarded to successor node.

            (*optional, default: 100*)

        :test_percentage_selected:
            The percentage of test instances which 
            is forwarded to successor node.

            (*optional, default: 100*)
            
        :reduce_class:
            If you want only to reduce one class, choose this parameter
            otherwise, both classes are reduced in a balanced fashion.

            (*optional, default: False*)
            
        :num_train_instances:
             Instead of specifying *train_percentage_selected*, this option 
             allows to specify the absolute number of training instances of 
             class *class_label* that should be in the training set. 
             All instances that occur until *num_train_instances* are found are
             used for training.
         
             (*optional, default: None*)
        
        :class_label:
             If *num_train_instances*-option is used, this string determines the
             class of which training examples are count.

            (*optional, default: 'Target'*)
            
        :random:
             If *False*, the order of the data is retained. I.e. the first X
             percent or number of train instances are used for training. If 
             *True*, the training data is sampled randomly without taking into 
             consideration the data's order.
         
             (*optional, default: True*)

    **Exemplary call**
    
    .. code-block:: yaml
    
        -
            node : InstanceSelection
            parameters : 
                train_percentage_selected : 80
                test_percentage_selected : 100
                reduce_class : Standard

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/03/31
    """
    def __init__(self, train_percentage_selected=100, 
                 test_percentage_selected=100, reduce_class=False,
                 num_train_instances=None, class_label='Target', random=True,
                 **kwargs):
        super(InstanceSelectionNode, self).__init__(**kwargs)
        
        self.set_permanent_attributes(
            train_percentage_selected=train_percentage_selected,
            test_percentage_selected=test_percentage_selected,
            reduce_class=reduce_class,
            num_train_instances=num_train_instances,
            class_label=class_label, random=random)

    def get_num_data(self, iterator):
        """ Return a list of instances that contain *num_train_instances* many 
            instances of class *class_label* and all other instances that occur
            up to this point
        """
        counter = 0
        retained_instances = []
        while counter < self.num_train_instances:
            try:
                instance, label = iterator.next()
            except: #TODO: give some warning to user
                break
            else:
                if label == self.class_label:
                    counter += 1
                retained_instances.append((instance,label))
        return retained_instances

    def request_data_for_training(self, use_test_data):
        """ Returns data for training of subsequent nodes
        
        .. todo:: to document
        
        .. note::
              This method works differently in InstanceSelectionNode
              than in other nodes: Only *percentage_selected* of the available
              data are returned.
        """
        
        assert(self.input_node is not None)
        if self.train_percentage_selected > 100:
            self._log("Train percentage of %f reduced to 100." %
                      self.train_percentage_selected,
                      level=logging.ERROR)
            self.train_percentage_selected = 100
        self._log("Data for training is requested.", level=logging.DEBUG)

        if self.train_percentage_selected == 100 and \
                self.num_train_instances is None:
            return super(InstanceSelectionNode, self).request_data_for_training(
                use_test_data)

        # If we haven't computed the data for training yet
        if self.data_for_training is None:
            self._log("Producing data for training.", level=logging.DEBUG)
            # Train this node
            self.train_sweep(use_test_data)
            
            if not self.num_train_instances is None and self.random == False:
                retained_instances = self.get_num_data(
                    self.input_node.request_data_for_training(use_test_data))
            else:
                # Store all data
                if self.num_train_instances is None:
                    all_instances = defaultdict(list)
                    for instance, label in self.input_node.request_data_for_training(
                        use_test_data):
                        all_instances[label].append(instance)
                else:
                    all_instances = list(
                        self.input_node.request_data_for_traning(use_test_data))

                if self.random:
                    r = random.Random(self.run_number)

                if not self.num_train_instances is None and self.random:
                    r.shuffle(all_instances)
                    retained_instances = self.get_num_data(
                        all_instances.__iter__())
                else:
                    retained_instances = []
                    self._log("Keeping only %s percent of training data" %
                          self.train_percentage_selected,
                          level=logging.DEBUG)
                    # Retain only *percentage_selected* percent of the data
                    for label, instances in all_instances.iteritems():
                        # enable random choice of samples
                        r.shuffle(instances)
                        if not self.reduce_class or \
                                self.train_percentage_selected == 100:
                            end_index = int(round(len(instances) *
                                                  self.train_percentage_selected / 100))
                        elif not (self.reduce_class == label):
                            end_index = len(instances)
                        else:  # self.reduce_class==label--> reduction needed
                            end_index = int(round(len(instances) *
                                                  self.train_percentage_selected / 100))
        
                        retained_instances.extend(zip(instances[0:end_index],
                                                      [label]*end_index))
            if self.random:
                # mix up samples between the different labels
                r.shuffle(retained_instances)
            # Compute a generator the yields the train data and
            # encapsulate it in an object that memoizes its outputs and
            # provides a "fresh" method that returns a new generator that will
            # yield the same sequence            
            train_data_generator = ((self.execute(data), label)
                                    for (data, label) in retained_instances)
                     
            self.data_for_training = MemoizeGenerator(train_data_generator,
                                                      caching=self.caching) 
        
        self._log("Data for training finished", level=logging.DEBUG)
        # Return a fresh copy of the generator  
        return self.data_for_training.fresh()
    
    def request_data_for_testing(self):
        """ Returns data for testing of subsequent nodes

        .. todo:: to document
        """
        assert(self.input_node is not None)
        if self.test_percentage_selected > 100:
            self._log("Test percentage of %f reduced to 100." %
                      self.test_percentage_selected,
                      level=logging.ERROR)
            self.test_percentage_selected = 100
        self._log("Data for testing is requested.", level=logging.DEBUG)

        if self.test_percentage_selected == 100:
            return super(InstanceSelectionNode, self).request_data_for_testing()

        # If we haven't computed the data for testing yet
        if self.data_for_testing is None:
            # Assert  that this node has already been trained
            assert(not self.is_trainable() or 
                   self.get_remaining_train_phase() == 0)
            
            # Divide available instances according to label
            all_instances = defaultdict(list)
            for instance, label in self.input_node.request_data_for_testing():
                all_instances[label].append(instance)
                
            self._log("Keeping only %s percent of test data" %
                      self.test_percentage_selected,
                      level=logging.DEBUG)
            r = random.Random(self.run_number)
            
            # Retain only *percentage_selected* percent of the data
            retained_instances = []
            for label, instances in all_instances.iteritems():
                # enable random choice of samples
                r.shuffle(instances)
                if not self.reduce_class or \
                        self.test_percentage_selected == 100:
                    end_index = int(round(len(instances) *
                                    self.test_percentage_selected / 100))
                elif not (self.reduce_class == label):
                    end_index = len(instances)
                else:  # self.reduce_class==label--> reduction needed
                    end_index = int(round(len(instances) *
                                    self.test_percentage_selected / 100))

                retained_instances.extend(zip(instances[0:end_index],
                                              [label]*end_index))
            # mix up samples between the different labels
            r.shuffle(retained_instances)
            # Compute a generator the yields the test data and
            # encapsulate it in an object that memoizes its outputs and
            # provides a "fresh" method that returns a new generator that'll
            # yield the same sequence
            self._log("Producing data for testing.", level=logging.DEBUG)
            test_data_generator = ((self.execute(data), label)
                                   for (data, label) in retained_instances)
                    
            self.data_for_testing = MemoizeGenerator(test_data_generator,
                                                     caching=self.caching)
        
        self._log("Data for testing finished", level=logging.DEBUG)
        # Return a fresh copy of the generator
        return self.data_for_testing.fresh()
    
    def _execute(self, time_series):
        return time_series  # We don't do anything with the kept instances


class ReduceOverrepresentedClassNode(BaseNode):
    """ Reject instances to balance categories for classification

    The node forwards only a reduced number 
    of the training and test instances of the bigger class
    to get a balanced ratio of the
    classes. The forwarded instances are selected randomly.
    All data of the underrepresented class is
    forwarded.
    
    **Parameters**

    **Exemplary call**

    .. code-block:: yaml
    
        -
            node : Reduce_Overrepresented_Class
            
    :Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2010/09/22

    """
    def __init__(self, **kwargs):
        super(ReduceOverrepresentedClassNode, self).__init__(**kwargs)

    def request_data_for_training(self, use_test_data):
        """ Returns data for training of subsequent nodes
        
        .. todo:: to document
        """
        assert(self.input_node is not None)
        
        self._log("Data for testing is requested.", level=logging.DEBUG)
        
        if self.data_for_training is None:
            self._log("Producing data for training.", level=logging.DEBUG)
            # Train this node
            self.train_sweep(use_test_data)
            
            # Divide available instances according to label
            all_instances = defaultdict(list)
            for instance, label in self.input_node.request_data_for_training(
                    use_test_data):
                all_instances[label].append(instance)
            
            retained_instances = self.balance_instances(all_instances)
            
            # Compute a generator the yields the test data and
            # encapsulate it in an object that memoizes its outputs and
            # provides a "fresh" method that returns a new generator that will
            # yield the same sequence
            self._log("Producing data for testing.", level=logging.DEBUG)
            train_data_generator = ((self.execute(data), label)
                                    for (data, label) in retained_instances)
                    
            self.data_for_training = MemoizeGenerator(train_data_generator,
                                                      caching=self.caching)
        
        self._log("Data for training finished", level=logging.DEBUG)
        # Return a fresh copy of the generator  
        return self.data_for_training.fresh()
    
    def request_data_for_testing(self):
        """ Returns data for testing of subsequent nodes

        .. todo:: to document
        """
        assert(self.input_node is not None)
        
        self._log("Data for testing is requested.", level=logging.DEBUG)
        
        # If we haven't computed the data for testing yet
        if self.data_for_testing is None:
            # Assert  that this node has already been trained
            assert(not self.is_trainable() or 
                   self.get_remaining_train_phase() == 0)
            
            # Divide available instances according to label
            all_instances = defaultdict(list)
            
            for instance, label in self.input_node.request_data_for_testing():
                all_instances[label].append(instance)
            
            retained_instances = self.balance_instances(all_instances)
            
            # Compute a generator the yields the test data and
            # encapsulate it in an object that memoizes its outputs and
            # provides a "fresh" method that returns a new generator that will
            # yield the same sequence
            self._log("Producing data for testing.", level=logging.DEBUG)
            test_data_generator = ((self.execute(data), label)
                                   for (data, label) in retained_instances)
                    
            self.data_for_testing = MemoizeGenerator(test_data_generator,
                                                     caching=self.caching)
        self._log("Data for testing finished", level=logging.DEBUG)
        # Return a fresh copy of the generator
        return self.data_for_testing.fresh()
    
    def _execute(self, time_series):
        return time_series # We don't do anything with the kept instances

    def balance_instances(self, all_instances):
        """Method that performs the rejections of the data in the oversized class"""
        retained_instances = []
            
        # it is supposed to have a binary classifier, e.g. to have exactly 2 classes
        #if not len(all_instances.keys())==2:
        #    raise ValueError("Too many classes: only binary classification supported")
            
        # count the number of instances per class 
        min_num_instances_per_class = float("+inf")
        for label, instances in all_instances.iteritems():
            min_num_instances_per_class = min(min_num_instances_per_class,
                                              len(instances))
        r = random.Random(self.run_number)
        # retain only the number of instances that corresponds 
        # to the size of smaller class 
        for label, instances in all_instances.iteritems():
            r.shuffle(instances)
            retained_instances.extend(
                zip(instances[0:min_num_instances_per_class],
                    [label]*min_num_instances_per_class))
        r.shuffle(retained_instances)
        return retained_instances


_NODE_MAPPING = {"RandomInstanceSelection": InstanceSelectionNode,
                 "Reduce_Overrepresented_Class": ReduceOverrepresentedClassNode}
