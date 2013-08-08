import abc


class Messenger(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        pass
        
    @abc.abstractmethod
    def register(self):
        return

    @abc.abstractmethod
    def end_transmission(self):
        return
        
    @abc.abstractmethod
    def send_message(self,message):
        return
    