""" Algorithms, using only one class for classification

Though this focuses on one class, the relation to the ``REST`` should still
be specified.
"""
from pySPACE.missions.nodes.classification.base import RegularizedClassifierBase
import logging

# import the external libraries
from pySPACE.missions.nodes.classification.svm_variants.external import LibSVMClassifierNode

try: # Libsvm
    import svmutil
except ImportError:
    pass


class OneClassClassifierBase(RegularizedClassifierBase):
    """ Base node to handle class labels during training to filter out irrelevant data

    :class_labels:
            List of the two or more classes,
            where first element is the relevant one
            and the second is the negative class.
    """
    def train(self, data, label):
        """ Special mapping for one-class classification

        Reduce training data to the one main class.
        """
        #one vs. REST case
        if "REST" in self.classes and not label in self.classes:
            label = "REST"
        # one vs. one case
        if not self.multinomial and len(self.classes)==2 and not label in self.classes:
            return

        if len(self.classes)==0:
            self.classes.append(label)
            self._log("No positive class label given in: %s. Taking now: %s."\
                      %(self.__class__.__name__,label),
                      level=logging.ERROR)
        if not label==self.classes[0]:
            if len(self.classes)==1:
                self.classes.append(label)
                self._log("No negative class label given in: %s. Taking now: %s."\
                          %(self.__class__.__name__,label),
                          level=logging.WARNING)
            return
        super(RegularizedClassifierBase, self).train(data,label)


class LibsvmOneClassNode(LibSVMClassifierNode, OneClassClassifierBase):
    """ Interface to one-class SVM in Libsvm package

    **Parameters**

        Parameters are as specified in
        :class:`~pySPACE.missions.nodes.classification.svm_variants.external.LibSVMClassifierNode`,
        except the ``svm_type``, which is set manually in this node to
        "one-class SVM".

        :class_labels:
            see: :class:`OneClassClassifierBase`


    **Exemplary Call**

    .. code-block:: yaml

        -
            node : LibsvmOneClass
            parameters :
                complexity : 1
                kernel_type : "LINEAR"
                class_labels : ['Target', 'Standard']
                weight : [1,3]
                debug : True
                store : True
                max_iterations : 100
    """
    def __init__(self,**kwargs):
        LibSVMClassifierNode.__init__(self,svm_type="one-class SVM", **kwargs)

    def train(self,data,label):
        OneClassClassifierBase.train(self, data, label)
