""" Filtering of windows based on the label
"""

import itertools
import logging

from pySPACE.missions.nodes.base_node import BaseNode
from pySPACE.tools.memoize_generator import MemoizeGeneratorNotRefreshableException


class FilterGenerator(object):
    """ Generator object that performs the actual filtering of the windows.
    """

    def __init__(self, generator, caching=False,
                 trigger_event = None,
                 activation_label = None,
                 positive_event = None,
                 negative_event = None,
                 deactivation_label = None):
        """ Stores the generator and creates an empty cache

        .. note::
                Since the output of the generator is ordered,
                the cache is an ordered sequence of variable length like a list
        """
        self.generator = generator
        self.caching = caching
        self.refreshable = True
        if self.caching:
            self.cache = []
        self.trigger_event = trigger_event
        self.activation_label = activation_label
        self.positive_event = positive_event
        self.negative_event = negative_event
        self.deactivation_label = deactivation_label
        self.target_shown = False

    def _fetch_from_generator(self):
        """
        Fetches one fresh value from the generator, store it in the cache and yield it
        """
        while True:
            x = self.generator.next()
            #decide if window should be filtered out = throw it away
            nextValue = self.filter_windows(x)
            if nextValue:
                if self.caching:
                    self.cache.append(nextValue)
                else:
                    self.refreshable = False
                yield nextValue
            else:
                continue


    def fresh(self):
        """ Return one generator that yields the same values
        like the internal one that was passed to __init__.

        .. note:: It does not recompute values that have already
            been requested before but just uses these from the internal cache.

        .. note:: Calling fresh invalidates all existing
                generators that have been created before using this method,
                i.e. there can only be one generator at a time
        """
        if self.caching:
            return itertools.chain(self.cache,
                                   self._fetch_from_generator())
        else:
            if not self.refreshable:
                raise MemoizeGeneratorNotRefreshableException( "This MemoizeGenerator does not cache elements from the generator and can thus not be reset")

            return self._fetch_from_generator()


    def filter_windows(self, value):
        """ Method that is called for the filtering process.
        """
        (data,label) = value


        result = []

        # check for trigger event
        if self.trigger_event is not None:
            if label in self.positive_event:
                self.target_shown = True
                self.last_target_data = (data, label)
            elif label in self.trigger_event:
                if self.target_shown:
                    result.append(self.last_target_data)
                    self.target_shown = False
                else:
                    pass
            elif label in self.negative_event:
                result.append((data, label))
            else:
                print "##### nothing forwarded ##### (label=%s, trigger_event=%s)" % (label, self.trigger_event)

        # check for activation_label
        elif self.activation_label is not None:
            if label in self.activation_label:
                print "Detection started"
                active = True
            elif label in self.deactivation_label:
                print "Detection stopped"
                active = False

            if label in self.positive_event and active:
                result.append((data, label))

        # classification for every other case
        else:
            result.append((data, label))

        if result != []:
            for (x,y) in result:
                return (x,y)
        else:
            return 0


class FilterWindowsNode(BaseNode):
    """Filter out the windows depending on their label.

    The node inspects the label of the window and decides,
    based on the specified criteria and past events, if the
    window should be forwarded or ignored.

    It is possible to use a simple switch on / switch off mechanism
    or more complex state-machine like filtering with triggers.

    **Parameters**

        :activation_label:
            Start streaming windows after a window with this label.

            (*optional, default: None*)

        :deactivation_label:
            Stop streaming windows after a window with this label.

            (*optional, default: None*)

        :positive_event:
            Positive event for the state machine.

            (*optional, default: None*)

        :negative_event:
            Negative event for the state machine.

            (*optional, default: None*)


    **Exemplary Call**

    .. code-block:: yaml

        -
            node : FilterWindows
            parameters :
                deactivation_label : Stop
                activation_label : Start


    :Authors: Hendrik Woehrle (hendrik.woehrle@dfki.de)
    :Created: 2013/04/04
    """
    def __init__(self, trigger_event = None,
                 activation_label = None,
                 deactivation_label = None,
                 positive_event = None,
                 negative_event = None,
                 use_test_data = True,
                 **kwargs):
        super(FilterWindowsNode, self).__init__(**kwargs)

        self.set_permanent_attributes(trigger_event = trigger_event,
                                      activation_label = activation_label,
                                      positive_event = positive_event,
                                      negative_event = negative_event,
                                      deactivation_label = deactivation_label,
                                      use_test_data = use_test_data)


    def request_data_for_training(self, use_test_data):
        """ Returns the filtered data for training of subsequent nodes """

        self._log("Data for training is requested.", level = logging.DEBUG)
        if self.data_for_training == None:
            self._log("Producing data for training.", level = logging.DEBUG)
            # Train this node
            self.train_sweep(use_test_data)
            train_data_generator = \
                     itertools.imap(lambda (data, label) : (data, label),
                                    self.input_node.request_data_for_training(use_test_data))

            self.data_for_training = FilterGenerator(train_data_generator,
                                                          caching=self.caching,
                                                          trigger_event = self.trigger_event,
                                                          activation_label = self.activation_label,
                                                          positive_event = self.positive_event,
                                                          negative_event = self.negative_event,
                                                          deactivation_label = self.deactivation_label)
            self._log("Data for training finished", level = logging.DEBUG)

        return self.data_for_training.fresh()


    def request_data_for_testing(self):
        """ Returns the data for testing of subsequent nodes """
        self._log("Data for testing is requested.", level = logging.DEBUG)
        self._log("Returning iterator over empty sequence.", level = logging.DEBUG)
        return (x for x in [].__iter__())


_NODE_MAPPING = {"Filter_Windows": FilterWindowsNode}
