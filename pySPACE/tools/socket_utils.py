#!/usr/bin/env python
# encoding: utf-8
"""
Provide often needed functions when dealing with the Python socket module

** Functions **

    :receive: 
        Return a full message read from a socket connection
        
    :inform:
        Send a message to a socket connection
        
    :talk:
        Send a message to a socket connection and return the answer
        
:Author: Anett Seeland (anett.seeland@dfki.de)
:Created: 2012/08/28
"""

import socket
import warnings
import time
import traceback

def _reconnect(ip_port):
    """Return a new socket connection over TCP/IP """
    com = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    com.connect(ip_port)
    return com

def receive(conn, end_token="!END!"):
    """Return a full message read from a socket connection"""
    full_message = []
    msg = ''
    while True:
        msg = conn.recv(4096)
        if end_token in msg:
            full_message.append(msg[:msg.find(end_token)])
            break
        full_message.append(msg)
        if len(full_message)>1:
            #check if end_of_data was split
            last_pair=full_message[-2]+full_message[-1]
            if end_token in last_pair:
                full_message[-2]=last_pair[:last_pair.find(end_token)]
                full_message.pop()
                break
    return "".join(full_message)
    
def inform(to_send, conn=None, ip_port=None):
    """Send a message and return the connection
    
    If an connection is already established, the message is send to it.
    If the connection is broken and the message cannot be sent, 
    a new connection is created.
    
    **Parameters**
        :to_send: 
            A String representing the message to send.
            
        :conn:
            A client socket where the message is send to.
            
        :ip_port:
            A Tuple of IP and port for potentially reconnect.
            
    One of the parameters *conn* or *ip_port* has to be specified.
    """
    if conn != None:
        # let server some time to receive all data
        time.sleep(0.003)
        try:
            conn.sendall(to_send)
        except socket.error,e:
            warnings.warn(str( time.asctime( time.localtime(time.time()) ))+': recv '+ str(e))
            traceback.print_stack()
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except:
                pass
            conn.close()
            time.sleep(0.1)
            # try reconnection
            if ip_port != None:
                new_conn = _reconnect(ip_port)
                if "Connection timed out" in str(e):
                    return inform(to_send,new_conn,ip_port)
                else:
                    return inform(to_send,new_conn)
            else:
                raise RuntimeError("Communication failed!")
        else:
            return conn
    elif ip_port != None:
        conn = _reconnect(ip_port)
        return inform(to_send,conn,ip_port)
    else:
        raise Exception("Specify at least connection or ip/port")


def talk(to_send, conn=None, ip_port=None):
    """Send a message and return the answer and the connection
    
    If an connection is already established, the message is send to it
    and the answer is returned. If the connection is broken and the message
    cannot be sent, a new connection is created.
    
    **Parameters**
        :to_send: 
            A String representing the message to send.
            
        :conn:
            A client socket where the message is send to.
            
        :ip_port:
            A Tuple of IP and port for potentially reconnect.
            
    One of the parameters *conn* or *ip_port* has to be specified.
    """
    conn = inform(to_send, conn, ip_port)
    if conn != None:
        # let server some time to receive all data
        time.sleep(0.003)
        # receive answer
        try:
            answer = receive(conn)
        except socket.error,e:
            warnings.warn(str( time.asctime( time.localtime(time.time()) ))+': recv '+ str(e))
            traceback.print_stack()
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except:
                pass
            conn.close()
            time.sleep(0.1)
            # try reconnection
            if ip_port != None:
                new_conn = _reconnect(ip_port)
                if "Connection timed out" in str(e):
                    return talk(to_send,new_conn,ip_port)
                else:
                    return talk(to_send,new_conn)
            else:
                raise RuntimeError("Communication failed!")
        else:
            return (conn, answer)
    elif ip_port != None:
        # establish a connection
        conn = _reconnect(ip_port)
        return talk(to_send,conn,ip_port)
    else:
        raise Exception("Specify at least connection or ip/port")
