# import pySPACE.tools.logging_stream_handler_colorer
import logging
from pySPACE.environments.live.communication import messenger

online_logger = logging.getLogger("pySPACELiveLogger")


import multiprocessing
import time


class AdrfMessenger(messenger.Messenger):
    
    def __init__(self):
        self.adrf_send_queue = multiprocessing.Queue()
        self.adrf_receive_queue = multiprocessing.Queue()
            
        self.adrf_interface = None
        
        self.adrf_interface_process = \
            multiprocessing.Process(target = self.adrf_process)
        self.adrf_interface_process.start()
        
        self.registered = False
        
    def adrf_process(self):
        # initialize and start the aBRI-Python-ADRF-Interface
        try:
            #TODO: Find a better way to load this!
            # sys.path.append('/Volumes/Daten_LRP/vi_bot_study/env/Applications/AbriPyADRF')
            # sys.path.append('/Users/ydong/repos/env/trunk/Applications/AbriPyADRF')
            import PythonAdrfInterface
        except:
            online_logger.error("'PythonAdrfInterface could not be imported!'")

        self.adrf_interface = PythonAdrfInterface.PythonAdrfInterface()
        online_logger.info("ADRF interface constructed")
        online_logger.info("starting ADRF interface")
        #self.adrf_interface.start()
        online_logger.info("ADRF interface started")
        command = None
        
        # start the event loop of the aBRI-Python-ADRF-Interface
        
        event = self.adrf_send_queue.get(block = True, timeout = None)
        while event != None:
            #online_logger.info("Got event " + event[0])
            if event[0] == 'Register' and not self.adrf_interface.isRegistered():
                #online_logger.info("Registering ADRF")
                while not self.adrf_interface.isRegistered():
                    self.adrf_interface.sendRegistration()
                    time.sleep(0.01)
            if event[0] == "UndoRegistration":
                #online_logger.info("Undo registration of ADRF")
                self.adrf_interface.undoRegistration()
            elif event[0] == 'CheckRegisterStatus':
                #online_logger.info("Checking register status of ADRF")
                self.adrf_receive_queue.put(self.adrf_interface.isRegistered())
            elif event[0] == 'GetCommand':
                command = self.adrf_interface.getCommand()
                self.adrf_receive_queue.put(command)
                command = None
            elif event[0] == 'GetConfig':
                config = self.adrf_interface.getConfig()
                self.adrf_receive_queue.put(config)
            elif event[0] == 'SetState':
                #online_logger.info("Setting state " + str(event[1]))
                config = self.adrf_interface.setState(event[1])
            elif event[0] == 'LRP':
                #online_logger.info(str(event[1]))
                self.adrf_interface.sendLRPProbability(event[1])
            elif event[0] == 'P300':
                self.adrf_interface.sendP300(event[1])
            elif event[0] == 'sendAbriFlow':
                self.adrf_interface.sendAbriFlow(event[1])
                   
            
            #online_logger.info('%s' % dataItem)
            while self.adrf_send_queue.empty():
                time.sleep(0.005)
            event = self.adrf_send_queue.get(block = True, timeout = None)

        # stop the event loop of the aBRI-Python-ADRF-Interface
        self.adrf_interface.stop()
        self.adrf_interface.wait()
        
    def adrf_receive_command(self):
        self.adrf_send_queue.put(['GetCommand'])
        command = self.adrf_receive_queue.get()
        return command

    def __del__(self):
        # input a 'None' to the ADRF Queue in order to stop the ADRF interface
        # process
        self.adrf_send_queue.put(None)
        # wait for the process to terminate
        while self.adrf_interface_process.is_alive():
            time.sleep(1)
            
    def register(self):
        self.adrf_send_queue.put(['Register'], block = True)
        
    def is_registered(self):
        self.adrf_send_queue.put(['CheckRegisterStatus'])
        register_status = self.adrf_receive_queue.get()
        return register_status
        
    def end_transmission(self):
        # input a 'None' to the ADRF queue in order to stop the ADRF interface
        # process
        self.adrf_send_queue.put(None)
        # wait for the process to terminate
        while self.adrf_interface_process.is_alive():
            time.sleep(1)
            
    def send_message(self,message):
        self.adrf_send_queue.put(message)
        
    def get_config(self):
        self.adrf_send_queue.put(['GetConfig'])
        config = self.adrf_receive_queue.get()
        return config

    def set_state(self, state):
        self.adrf_send_queue.put(['SetState',state])
        
    def undo_registration(self):
        self.adrf_send_queue.put(['UndoRegistration'])
    
    def sendAbriFlow(self, flow):
        self.adrf_send_queue.put(['sendAbriFlow', flow])        
        
        
