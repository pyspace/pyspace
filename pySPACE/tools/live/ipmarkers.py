import socket
import time
import struct
import threading
import select
import random
import Queue
import warnings

class MarkerSocket(threading.Thread):

    def __init__(self, ip="10.250.3.83", port=55555, name="test", **kwargs):
        super(MarkerSocket, self).__init__(**kwargs)
        self.name = name
        self.ip = ip
        self.port = port
        self.connected = False
        self.running = True

    def send(self, marker):
        if not self.connected:
            warnings.warn("%s not sent - socket is not connected!" % marker)
            return
        fmt = "bb6sQ"
        data = struct.pack(fmt, 2, struct.calcsize(fmt), marker[:6], long(time.time()*1000))
        try:
            self.s.send(data)
        except socket.error:
            warnings.warn("%s not sent - socket error: %s" % (marker, socket.errno))
            self.connected = False

    def run(self):
        fmt = "bb6sQQQQ"
        while self.running:
            while (not self.connected) and self.running:
                try:
                    self.s = socket.socket()
                    self.s.connect((self.ip,self.port))
                    self.s.send(self.name)
                except socket.error:
                    time.sleep(1)
                    continue
                self.connected = True
                break

            while self.connected and self.running:
                (r,w,e) = select.select([self.s], [], [], .01)
                if self.s in r:
                    _t2 = long(time.time()*1000)

                    beacon = ""
                    try:
                        beacon = str(self.s.recv(struct.calcsize(fmt)))
                    except socket.error:
                        warnings.warn("%s: error during recv!" % self.name)
                        self.connected = False

                    if len(beacon) == 0:
                        warnings.warn("%s: connection closed by remote!" % self.name)
                        self.connected = False
                        continue

                    (typ, size, progress, t1, t2, t3, t4) = struct.unpack(fmt, beacon)

                    t2 = _t2
                    t3 = long(time.time()*1000)
                    progress += "*"

                    beacon = struct.pack(fmt, typ, size, progress, t1, t2, t3, t4)
                    self.s.send(beacon)

    def stop(self):
        self.running = False

class MarkerServer(threading.Thread):

    def __init__(self, port=55555, sync_interval=10, **kwargs):
        super(MarkerServer, self).__init__(**kwargs)
        self.s = socket.socket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self.s.bind(("", port))
        self.s.listen(50)
        self.children = []
        self.sync_interval = sync_interval
        self.queue = Queue.Queue()
        self.running = True

    def stop(self):
        self.running = False

    def __repr__(self):
        s = super(MarkerServer, self).__repr__()
        return str("%s\n\tconnected to %d clients" % (s, len(self.children)))

    def run(self):
        while self.running:
            (r,w,e) = select.select([self.s], [], [], .25)
            if self.s in r:
                (client, address) = self.s.accept()
                print("connection requested %s" % (str(address)))
                c = MarkerAcquisitionThread(client, address,
                                            sync_interval=self.sync_interval,
                                            queue=self.queue)
                c.start()
                self.children.append(c)

            self.join_stopped_threads()

        self.s.close()

        for c in self.children:
            if c.isAlive():
                c.stop()
                c.join()
        self.children = []

    def join_stopped_threads(self):
        for c in self.children:
            if not c.isAlive():
                c.join()

    def read(self):
        if not self.queue.empty():
            return self.queue.get(block=False)
        return None, None

class MarkerAcquisitionThread(threading.Thread):

    def __init__(self, client, address, sync_interval=10, queue=None, **kwargs):
        super(MarkerAcquisitionThread, self).__init__(**kwargs)
        self.client = client
        self.address = address
        self.sync_interval = sync_interval
        self.delay_ms = 0
        self.name = self.client.recv(128)
        self.running = True
        self.queue = queue
        print("marker source %s@%s connected" % (self.name, self.address[0]))

    def stop(self):
        self.running = False

    def __repr__(self):
        s = super(MarkerAcquisitionThread, self).__repr__()
        return str("%s:%s" % (s, self.address))

    def run(self):
        sync_fmt = "bb6sQQQQ"
        mark_fmt = "bb6sQ"
        last_sync = 0.0
        while self.running:
            (r,w,e) = select.select([self.client], [], [], .01)
            if self.client in r:
                try:
                    msg = self.client.recv(struct.calcsize(sync_fmt))
                except socket.error as e:
                    print("client %s@%s: %s" % (self.name, self.address[0], e.strerror))
                    self.stop()
                    continue
                if len(msg) == 0:
                    print("client %s@%s: socket closed" % (self.name, self.address[0]))
                    self.stop()
                    continue
                elif len(msg) == struct.calcsize(mark_fmt):
                    self.show_marker(msg, mark_fmt)
                elif len(msg) == struct.calcsize(sync_fmt):
                    self.sync_end(msg, sync_fmt)
                else:
                    pass

            if int(time.time()-last_sync) > self.sync_interval:
                self.sync_start(sync_fmt)
                last_sync = time.time()

        self.client.close()

    def sync_start(self, fmt):
        t1 = long(time.time()*1000)
        beacon = struct.pack(fmt, 1, struct.calcsize(fmt), "*", t1, 0, 0, 0)
        self.client.send(beacon)

    def sync_end(self, beacon, fmt):
        (typ, size, progress, t1, t2, t3, _t4) = struct.unpack(fmt, beacon)
        t4 = long(time.time()*1000)
        self.delay_ms = ((int(t2)-int(t4))+(int(t3)-int(t1)))/2
        # print("SYNC done! delay: %f [ms]" % self.delay_ms)

    def show_marker(self, marker, fmt):
        (typ, size, mark, t1) = struct.unpack(fmt, marker)
        self.queue.put((str(mark).strip("\0"), int(t1-self.delay_ms)))


if __name__ == "__main__":

    markerserver = MarkerServer(port=55555, sync_interval=15)
    markerserver.start()

    sockets = []

    for i in range(25):
        c = MarkerSocket(ip="127.0.0.1", port=55555, name=str("client%d" % i))
        c.start()
        sockets.append(c)

    for s in sockets:
        mark = str("S%3d" % int(random.random()*255))
        print("sending marker %s with client %s" % (mark, s.name))
        s.send(mark)
        time.sleep(random.random()*1)
        while True:
            m = markerserver.read()
            if None in m:
                break
            print m

    print markerserver

    for c in sockets:
        c.stop()
        c.join()
    sockets = []

    for i in range(markerserver.sync_interval*1, 0, -1):
        print("waiting.. %d" % i)
        time.sleep(1)

    print markerserver

    markerserver.stop()
    markerserver.join()