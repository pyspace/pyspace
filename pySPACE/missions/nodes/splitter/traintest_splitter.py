""" Split data into one training and one test data set with restriction like randomization or fixed percentages """
import random

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.tools.memoize_generator import MemoizeGenerator

class TrainTestSplitterNode(BaseNode):
    """ Split data into one training and one test data set with a fixed ratio
    
    The relative
    size of the two sets is controlled via the parameter train_ratio.

    .. warning:: the class ratio is not retained

    .. todo::
        introduce stratified parameter as in CV_Splitter
    
    **Parameters**
    
     :train_ratio:
         The ratio of the overall available data that is assigned to the 
         training set. The remaining data (1-train_ratio) is used for testing.
         
         (*optional, default: 0.5*)
         
     :num_train_instances:
         Instead of specifying a train_ratio, this option allows to specify the
         absolute number of training instances of class *class_label* that 
         should be in the training set. All instances that occur until 
         *num_train_instances* are found are used for training. The remaining
         data are used for testing.
         
         (*optional, default: None*)
    
     :class_label:
         If *num_train_instances*-option is used, this string determines the
         class of which training examples are count.
     
     :random:
         If *False*, the order of the data is retained. I.e. the train_ratio
         instances are used for training and the remaining as test data. If 
         *True*, the two sets are sampled randomly from the data without
         taking into consideration the data's order.
         
         (*optional, default: True*)
    
    **Exemplary Call**
    
    .. code-block:: yaml
    
        -
            node : TrainTestSplitter
            parameters :
                  train_ratio : 0.7
                  random : False
    
    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/03/08 (Documentation, old node)
    :LastChange: 2011/11/14 (Documentation) Anett Seeland
    """
    
    def __init__(self, train_ratio=0.5, random=True,
                 num_train_instances=None, class_label='Target', reverse=False,
                 **kwargs):
        super(TrainTestSplitterNode, self).__init__(**kwargs)
        assert(not(random and reverse)),"Reverse ordering makes no sense when randomization is active!"
        self.set_permanent_attributes(train_ratio=train_ratio,
                                      random=random,
                                      num_train_instances=num_train_instances,
                                      class_label=class_label,
                                      reverse=reverse,
                                      train_data=None,
                                      test_data=None)

    def is_split_node(self):
        """ Returns whether this is a split node. """
        return True

    def use_next_split(self):
        """ Use the next split of the data into training and test data.
        
        Returns True if more splits are available, otherwise False.
        
        This method is useful for benchmarking
        """
        return False
    
    def train_sweep(self, use_test_data):
        """ Performs the actual training of the node.
        
        .. note:: Split nodes cannot be trained
        """
        raise Exception("Split nodes cannot be trained")
    
    def request_data_for_training(self, use_test_data):
        """ Returns the data for training of subsequent nodes

        .. todo:: to document
        """
        # Create split lazily when required
        if self.train_data == None:
            self._create_split()

        # Create training data generator
        self.data_for_training = \
                MemoizeGenerator(instance for instance in self.train_data)
        
        return self.data_for_training.fresh()
    
    def request_data_for_testing(self):
        """ Returns the data for testing of subsequent nodes

        .. todo:: to document
        """
        # Create split lazily when required
        if self.test_data == None:
            self._create_split()
        
        # Create test data generator
        self.data_for_testing = \
                MemoizeGenerator(instance for instance in self.test_data)

        return self.data_for_testing.fresh()

    def _create_split(self):
        """ Create the split of the data into training and test data. """
        self._log("Splitting data into train and test data")
        train_data = list(self.input_node.request_data_for_training(use_test_data=False))

        # If there is already a  non-empty training set,
        # it means that we are not  the first split node in the node chain.
        if  len(train_data) > 0:
            raise Exception("No iterated splitting of data sets allowed\n "
                            "(Calling a splitter on a  data set that is already "
                            "split)")

        # Create generator instead of loading all data
        if self.num_train_instances and not (self.random):
            self.train_data = []
            input_generator=self.input_node.request_data_for_testing
            for i in range(self.num_train_instances):
                self.train_data.append(input_generator.next())
            self.test_data = input_generator
            return

        # Gather all test data
        test_data = list(self.input_node.request_data_for_testing())
        
        # Remember all the data and store it in memory
        # TODO: This might cause problems for large dataset
        data = train_data + test_data
        data_size = len(data)

        # Randomize order if randomization is not switched of
        if self.random:
            r = random.Random(self.run_number)
            r.shuffle(data)
        
        if self.num_train_instances!=None:
            if self.reverse:
                data = data[::-1]
            if len([i for i in range(len(data)) \
                  if data[i][1]==self.class_label])==self.num_train_instances:
                train_end = data_size
            else:
                counter = 0
                for (index, (window, label)) in enumerate(data):
                    # print "Label: ", label, "Zeitpunkt: ", window.start_time
                    if label == self.class_label:
                        counter += 1
                    if counter == self.num_train_instances:
                        train_end = index+1
                        break
                assert(self.num_train_instances==counter), \
                            "Too many instances to select."
        else:
            # Split data into train and test data according train_ratio
            train_end = int(round(data_size * self.train_ratio))
            
        self.train_data=data[0:train_end]
        self.test_data=data[train_end:]
