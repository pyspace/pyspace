import socket
import struct

import multiprocessing
import time
from pySPACE.environments.live.communication import messenger

class SocketMessenger(messenger.Messenger):
    #Constructor
    def __init__(self, port=54321, key="LRP"):
        #Used port
        self.port = port
        #Witch items should be tracked and sent
        self.key = key
        #Variables for the socket, the connection and the used address
        self.sock = 0
        self.conn = 0
        self.addr = 0
        
        self.tcp_send_queue = multiprocessing.Queue()
        self.tcp_process = \
            multiprocessing.Process(target = self.tcp_run)
        self.tcp_process.start()
        
    
    def __del__(self):
        self.tcp_send_queue.put(None)
        # wait for the process to terminate
        while self.tcp_process.is_alive():
            time.sleep(1)
        
    def register(self):
        pass
    
    def end_transmission(self):
        self.tcp_send_queue.put(None)
        # wait for the process to terminate
        while self.tcp_process.is_alive():
            time.sleep(1)

    #Main loop
    def tcp_run(self):
        #Init the socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #Dircet reuseability of the port
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        #Set host and port
        self.sock.bind(('', self.port))
        #Allow one connection and wait for client
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        print "socket_messenger.py: Connected with ",self.addr
        value = self.tcp_send_queue.get(block=True, timeout = None)
        while value != None:
            if value[0] in self.key:
                self.conn.send(struct.pack('f',float(value[1])))
            while self.tcp_send_queue.empty():
                time.sleep(0.001)
            value = self.tcp_send_queue.get(block=True, timeout = None)
            
        self.conn.close()
    
    #Sends value to the connected socket
    def send_message(self, message):
        self.tcp_send_queue.put(message)


    #Stops the thread
    def stop(self):
        self.tcp_send_queue.put(None)
        # wait for the process to terminate
        while self.tcp_process.is_alive():
            time.sleep(1)
            
