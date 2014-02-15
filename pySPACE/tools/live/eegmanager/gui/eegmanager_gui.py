#!/usr/bin/env python
import os
import sys
import yaml
import pickle
import time
import subprocess
from threading import Thread
import random
import xmlrpclib
import getpass
import httplib

# Determine path of current file
path = os.path.abspath(__file__)

# Move up to parent directory
path, tail = os.path.split(os.path.split(path)[0])
eegmanager_path = os.path.join(path, "eegmanager")
if not os.path.exists(eegmanager_path):
    print "ERROR: EEGManager location not found!"
    print "       it should be in %s" % eegmanager_path
    sys.exit(0)

if os.name == 'nt':
    eegmanager_exe = os.path.join(eegmanager_path, "release", "eegmanager.exe")
else:
    eegmanager_exe = os.path.join(eegmanager_path, "eegmanager")
    
if not os.path.isfile(eegmanager_exe):
    print "ERROR: EEGManager executable not found!"
    print "       please compile.."
    print "       %s not found!" % eegmanager_exe
    sys.exit(0) 

try:
    from PyQt4.QtCore import * 
    from PyQt4.QtGui import *
except ImportError:
    print "ERROR: The EEGManager GUI requires the PyQt4 package."
    sys.exit(0)



class ConnectPopup(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        
        self.parent = parent
        
        self.layout = QVBoxLayout(self)
        
        self.first_line = QHBoxLayout()
        self.ip = QLineEdit(self)
        # self.ip.setInputMask("000.000.000.000;")
        #self.ip.setPlaceholderText("ip.adr.res.s")
        self.ip.textChanged.connect(self.validate_ip)
        self.ip_ok = QPushButton(self)
        self.ip_ok.setIcon(QIcon("icons/20120611_eeg_gui_unknown"))
        self.ip_ok.setMaximumSize(QSize(20, 20))
        self.ip_ok.setFlat(True)
        self.ip_ok.setEnabled(False)
        self.first_line.addWidget(self.ip_ok)
        self.first_line.addWidget(self.ip)
        
        self.second_line = QHBoxLayout()
        self.port = QLineEdit(self)
        # self.port.setInputMask("00000;")
        #self.port.setPlaceholderText("port")
        self.port.textChanged.connect(self.validate_port)
        self.port_ok = QPushButton(self)
        self.port_ok.setIcon(QIcon("icons/20120611_eeg_gui_unknown"))
        self.port_ok.setMaximumSize(QSize(20, 20))
        self.port_ok.setFlat(True)
        self.port_ok.setEnabled(False)
        self.second_line.addWidget(self.port_ok)
        self.second_line.addWidget(self.port)
        
        self.last_line = QHBoxLayout()
        self.ok = QPushButton("Connect", self)
        self.ok.setIcon(QIcon("icons/20120611_eeg_gui_unknown"))
        self.ok.clicked.connect(self.validate)
        self.cancel = QPushButton("Abort", self)
        self.cancel.clicked.connect(self.reject)
        self.last_line.addWidget(self.cancel)
        self.last_line.addWidget(self.ok)
        
        self.layout.addLayout(self.first_line)
        self.layout.addLayout(self.second_line)
        self.layout.addLayout(self.last_line)
        self.setLayout(self.layout)
        
        self.eeg = None
        
    def validate(self):
        if self.validate_ip(self.ip.text()) and self.validate_port(self.port.text()):
            self.eeg = EegmanagerWrapper(parent=self.parent, ip=self.ip.text(), port=self.port.text())
            if self.eeg.connect_to_server():
                self.accept()
            else:
                self.ok.setIcon(QIcon("icons/20120611_eeg_gui_fail"))
                del self.eeg
                self.eeg = None
    
    def validate_ip(self, ip):
        i = ip.split(".")
        if len(i) != 4:
            self.ip_ok.setIcon(QIcon("icons/20120611_eeg_gui_fail"))
            return False
        else:
            self.ip_ok.setIcon(QIcon("icons/20120611_eeg_gui_ok"))
            return True
        
    def validate_port(self, port):
        try:
            p = int(port)
            if 1024 < p < 99999:
                self.port_ok.setIcon(QIcon("icons/20120611_eeg_gui_ok"))
                return True
        except:
            pass
        self.port_ok.setIcon(QIcon("icons/20120611_eeg_gui_fail"))
        return False
        
    def getData(self):
        return self.ip.text(), self.port.text()
        


class EegmanagerWrapper(QObject):
    """ Class for wrapping the functions of the eegmanager process """
    def __init__(self, parent=None, ip=None, port=None):
        QObject.__init__(self)
        self.remote = None
        self.process = None
        self.modules = None
        self.shutDown = False
        self.ip = ip
        self.port = port
        self.usage_infos = None

        self.parent = parent
        
        self.running = True
        self.refresh_ui = Thread(target=self.get_state, kwargs={"parent" : self.parent})
        self.refresh_ui.start()
        
        
    def __del__(self):
        self.running = False
        time.sleep(.3)
        self.refresh_ui.join()
        if self.process != None:
            self.process.kill()
        print "deleted!"
        
    def connect_to_server(self):
        if self.ip == None or self.port == None:
            return False
        self.remote = xmlrpclib.ServerProxy(str("http://%s:%s" % (self.ip, self.port)))
        try:
            self.modules = self.remote.show_modules()
        except:
            return False
        return True

    def create_server(self):
        xml_range = range(16253, 17253)
        random.shuffle(xml_range)
        
        for xmlport in xml_range:
            try:
                self.process = subprocess.Popen([eegmanager_exe, str(xmlport)])
            except OSError:
                print "ERROR: Executable could not be launched as subprocess"
                exit(0)

            time.sleep(.3)
            self.process.poll()
            if self.process.returncode == None:
                break
        
        self.remote = xmlrpclib.ServerProxy("http://127.0.0.1:%d" % xmlport)
        self.ip = "127.0.0.1"
        self.port = xmlport
        self.modules = self.remote.show_modules()

    def active(self):
        if self.local():
            self.process.poll()
            if self.process.returncode != None:
                print "subprocess exited with %d" % self.process.returncode
                return False
        return True
        
    def local(self):
        return self.process != None

    def startup(self):
        if self.ip == None and self.port == None and self.process == None:
            self.create_server()
        self.emit(SIGNAL("updateModuleList(PyQt_PyObject)"), self.modules)
        
        self.usage_infos = dict()
        for m in self.modules:
            self.usage_infos[m] = self.remote.usage(m)
        self.emit(SIGNAL("updateModuleUsage(PyQt_PyObject)"), self.usage_infos)
        
    def apply_setup(self, setup):
        needs_update = False
        
        if not self.active():
            return False, 0, str("subprocess exited!")
            
        remote_state = self.remote.get_state()
        remote_setup = self.remote.get_setup()
        
        # check for equality of the two setups here
        if len(setup) != len(remote_setup):
            needs_update = True
        for t in zip(remote_setup, setup):
            if t[0] != t[1]:
                print "mismatch:", t[0], t[1]
                needs_update = True
                break

        if remote_state[0] == "IDLE" or needs_update:
            self.remote.stop()
            i = self.remote.apply_setup(setup)
            if 0 == i:
                pass
            else:
                output = self.stdout()
                self.emit(SIGNAL("stdoutUpdate(PyQt_PyObject)"), output)
                output = output.strip()
                self.remote.stop()
                return False, i, output[output.find("FATAL"):]                 
                
        return True, 0, str("")
        
    def start(self):
        if self.active():
            self.remote.start()
            
        
            
    def _idle(self):
        self.emit(SIGNAL("idle()"))
            
    def _configure(self):
        self.emit(SIGNAL("configure()"))
            
    def _run(self):
        self.emit(SIGNAL("run()"))
        
    def _toggle(self):
        self.emit(SIGNAL("toggle()"))
        
    def get_state(self, parent):
        last_state = "IDLE"
        while self.running:
            try:
                state = self.remote.get_state()
            except:
                state = ["UNKNOWN"]
                
            # setup = self.eeg.remote.get_setup()
            # self.tiles.reset()
            # for t in range(0, len(setup), 2):
            #     self.tiles.addTile(setup[t], setup[t+1])

            if state[0] == "IDLE":
                if last_state != "IDLE":
                    print "IDLE now"
                    last_state = "IDLE"
                    self._idle()
                time.sleep(2)
            elif state[0] == "CONFIGURED":
                if last_state != "CONFIGURED":
                    print "CONFIGURED now"
                    if last_state == "RUNNING":
                        self._toggle()
                    last_state = "CONFIGURED"
                    self.emit(SIGNAL("processState(PyQt_PyObject)"), state)
                self._configure()
                time.sleep(.3)
            elif state[0] == "RUNNING":
                if last_state != "RUNNING":
                    print "RUNNING now"
                    last_state = "RUNNING"
                    self._run()
                    self._toggle()
                self.emit(SIGNAL("processState(PyQt_PyObject)"), state)
                time.sleep(.3)
            else :
                time.sleep(2)
                
        self.emit(SIGNAL("processState(PyQt_PyObject)"), state)
            
    def stop(self):
        if self.active():
            self.remote.stop()
            # print self.remote.stdout()

    def shutdown(self):
        if not self.active():
            return
        if not self.local():
            return
        print "shutting down.. @ %s" % self.remote
        self.running = False
        time.sleep(.3)
        self.refresh_ui.join()
        if not self.shutDown:
            try:
                self.remote.stop()
                self.remote.shut_down()
            except:
                print "error shutting down process"

            self.shutDown = True
            
    def leave(self):
        self.running = False
        time.sleep(.3)
        self.refresh_ui.join()
            
    def kill(self):
        self.running = False
        time.sleep(.3)
        self.refresh_ui.join()
        try:
            self.remote.stop()
            self.remote.shut_down()
        except:
            print "error shutting down process"

        self.shutDown = True

    def usage(self, module):
        try:
            u = self.usage_infos[str(module)]
        except:
            print "unknown module [%s]" % module
            return ['[-e|--error] <error> --this --should --not --happen!', '', '', '']
        return u
        # if not self.active():
        #     return ['[-e|--error] <error> --this --should --not --happen!', '', '', '']
        # try:
        #     u = self.remote.usage(str("%s" % module))
        # except httplib.BadStatusLine:
        #     print "caught bad-status-line error!"
        #     u = -1
        # if type(u) == int:
        #     print "Error retrieving usage for module %s" % module
        #     return ['[-e|--error] <error> --this --should --not --happen!', '', '', '']
        # return u
        
    def stdout(self):
        return self.remote.stdout()



class LibraryWidget(QDockWidget):
    """ Widget which lists all available modules

    Modules can be dragged onto the tile widget for instantiation.
    """
    def __init__(self, title="empty", parent=None, flags=0, modules=["foo", "bar"]):
        super(LibraryWidget, self).__init__(parent)
        
        self.setAllowedAreas(Qt.LeftDockWidgetArea|Qt.RightDockWidgetArea)
        
        self.parent = parent
        self.modules = modules
        self.content  = QWidget()
        self.layout = QVBoxLayout(self.content)
        
        self.view = QListView(self.content)
        self.model = QStringListModel(self.modules, self)
        self.view.setModel(self.model)
        self.view.setDragEnabled(True)
        self.view.setObjectName("Library")
        
        self.layout.addWidget(self.view)
        self.setWidget(self.content)
        
        self.model.dataChanged.connect(self.view.repaint)
        
    def setModules(self, modules):
        self.model.setStringList(modules)
        # for m in modules:
        #     print m
        #     print self.parent.eeg.usage(m) 
        #     print
        self.emit(SIGNAL('dataChanged'))
        
    def setUsage(self, usage):
        pass
        # can be stored here, optionaly..
        #print usage
    
    def closeEvent(self, event):
        self.toggleEvent()
    
    def toggleEvent(self):
        if self.isVisible() :
            self.parent.actionLibrary.setChecked(False)
            self.hide()
        else:
            self.parent.actionLibrary.setChecked(True)
            self.show()


class ConsoleWidget(QDockWidget):
    """ Widget which displays the stdout of the underlying process """
    def __init__(self, title="empty", parent=None, flags=0):
        super(ConsoleWidget, self).__init__(parent)

        self.setAllowedAreas(Qt.BottomDockWidgetArea)

        self.parent = parent
        
        self.content  = QWidget()
        self.layout = QVBoxLayout(self.content)
        self.view = QTextBrowser(self.parent)
        self.refresh = QPushButton(QIcon(), "refresh", self.parent)
        self.refresh.clicked.connect(self.refreshStdout)
        self.layout.addWidget(self.view)
        self.layout.addWidget(self.refresh)
        self.setWidget(self.content)
        self.hide()

    def refreshStdout(self):
        text = self.parent.centralwidget.eeg.stdout()
        self.appendStdout(text)
    
    def appendStdout(self, text):
        self.view.append(str("%s"%text))
                
    def closeEvent(self, event):
        self.toggleEvent()
    
    def hide(self):
        self.parent.actionConsole.setChecked(False)
        super(ConsoleWidget, self).hide()
        
    def show(self):
        self.parent.actionConsole.setChecked(True)
        super(ConsoleWidget, self).show()

    def toggleEvent(self):
        if self.isVisible() :
            self.hide()
        else:
            self.show()


class CentralWidget(QWidget):
    """ The main Widget for displaying and controlling the execution of the eegmanager """
    def __init__(self, parent=None, flags=0, scene_path=None):
        super(CentralWidget, self).__init__(parent=parent)
        
        self.parent = parent
        
        # fill layout
        self.layout = QVBoxLayout(self)
        self.tiles = TileWidget(self)
        self.control = ControlWidget(self)
        self.dockWidget = LibraryWidget(parent=parent)
        self.parent.addDockWidget(Qt.DockWidgetArea(0x1), self.dockWidget)
        self.console = ConsoleWidget(parent=parent)
        self.parent.addDockWidget(Qt.DockWidgetArea(0x8), self.console)
        self.layout.addWidget(self.tiles)
        self.layout.addWidget(self.control)
        self.setLayout(self.layout)
        
        # the underlying control object for the eegmanager
        self.eeg = EegmanagerWrapper(parent=self)
        
        # connect signals and slots
        QObject.connect(self.eeg, SIGNAL("updateModuleList(PyQt_PyObject)"), self.dockWidget.setModules)
        QObject.connect(self.eeg, SIGNAL("updateModuleUsage(PyQt_PyObject)"), self.dockWidget.setUsage)
        QObject.connect(self.eeg, SIGNAL("processState(PyQt_PyObject)"), self.tiles.processState)
        QObject.connect(self.eeg, SIGNAL("stdoutUpdate(PyQt_PyObject)"), self.console.appendStdout)
        QObject.connect(self.eeg, SIGNAL("idle()"), self.idle_callback)
        QObject.connect(self.eeg, SIGNAL("configure()"), self.configure_callback)
        QObject.connect(self.eeg, SIGNAL("run()"), self.run_callback)
        QObject.connect(self.eeg, SIGNAL("toggle()"), self.toggle_callback)
        
        QObject.connect(self, SIGNAL("reviseSetup(PyQt_PyObject)"), self.tiles.setupError)
        self.control.reset.clicked.connect(self.reset)
        self.control.quit.clicked.connect(self.close)
        self.control.start.clicked.connect(self.start)
        self.control.restart.clicked.connect(self.restart)
        self.control.stop.clicked.connect(self.stop)
        
        self.eeg.startup()
        
        if scene_path != None:
            self.load_from_pickle(scene_path)
            self.dockWidget.hide()
            self.console.show()
    
    def open(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File', '.', "EEGManager-Files (*.egm)")
        if len(filename) > 0:
            self.load_from_pickle(filename)
            
    def update(self):
        self.emit(SIGNAL("titleChanged(PyQt_PyObject)"), str("EEGmanager %s:%s" % (self.eeg.ip, self.eeg.port)))

    def save(self):
        setup, valid = self.tiles.getSetup()
        if not valid:
            print "ERROR GENERATING SETUP FOR SAVING!"
            return
        filename = QFileDialog.getSaveFileName(self, 'Save File', '.', "EEGManager-Files (*.egm)")
        if len(filename) > 0:
            self.save_to_pickle(filename, setup)
            
    def reset(self):
        self.eeg.stop()
        self.tiles.reset()
            
    def configure_callback(self):
        setup_local, valid = self.tiles.getSetup()
        setup_remote = self.eeg.remote.get_setup()
        if len(setup_remote) != len(setup_local):
            self.tiles.reset()
            for t in range(0, len(setup_remote), 2):
                print "adding ", setup_remote[t], setup_remote[t+1]
                self.tiles.addTile(setup_remote[t], setup_remote[t+1])
        
    def idle_callback(self):
        setup_local, valid = self.tiles.getSetup()
        if len(setup_local) != 0:
            self.tiles.reset()
        
    def run_callback(self):
        setup = self.eeg.remote.get_setup()
        self.tiles.reset()
        for t in range(0, len(setup), 2):
            self.tiles.addTile(setup[t], setup[t+1])
            self.tiles.update(True, -1)
            
    def toggle_callback(self):
        self.control.toggle()
        self.tiles.toggle()
        
            
    def connect(self):
        print "connect.."
        c = ConnectPopup(parent=self)
        c.setModal(True)
        r = c.exec_()
        if r == 1:
            self.eeg.shutdown()
            QObject.disconnect(self.eeg, SIGNAL("updateModuleList(PyQt_PyObject)"), self.dockWidget.setModules)
            QObject.disconnect(self.eeg, SIGNAL("updateModuleUsage(PyQt_PyObject)"), self.dockWidget.setUsage)
            QObject.disconnect(self.eeg, SIGNAL("processState(PyQt_PyObject)"), self.tiles.processState)
            QObject.disconnect(self.eeg, SIGNAL("stdoutUpdate(PyQt_PyObject)"), self.console.appendStdout)
            QObject.disconnect(self.eeg, SIGNAL("idle()"), self.idle_callback)
            QObject.disconnect(self.eeg, SIGNAL("configure()"), self.configure_callback)
            QObject.disconnect(self.eeg, SIGNAL("run()"), self.run_callback)
            QObject.disconnect(self.eeg, SIGNAL("toggle()"), self.toggle_callback)
            
            del self.eeg
            self.eeg = c.eeg
            QObject.connect(self.eeg, SIGNAL("updateModuleList(PyQt_PyObject)"), self.dockWidget.setModules)
            QObject.connect(self.eeg, SIGNAL("updateModuleUsage(PyQt_PyObject)"), self.dockWidget.setUsage)
            QObject.connect(self.eeg, SIGNAL("processState(PyQt_PyObject)"), self.tiles.processState)
            QObject.connect(self.eeg, SIGNAL("stdoutUpdate(PyQt_PyObject)"), self.console.appendStdout)
            QObject.connect(self.eeg, SIGNAL("idle()"), self.idle_callback)
            QObject.connect(self.eeg, SIGNAL("configure()"), self.configure_callback)
            QObject.connect(self.eeg, SIGNAL("run()"), self.run_callback)
            QObject.connect(self.eeg, SIGNAL("toggle()"), self.toggle_callback)
            self.eeg.startup()
            self.update()
            if self.eeg.remote.get_state()[0] == 'RUNNING':
                setup = self.eeg.remote.get_setup()
                self.tiles.reset()
                for t in range(0, len(setup), 2):
                    self.tiles.addTile(setup[t], setup[t+1])
                self.control.toggle(runnig=True)
                self.tiles.toggle(runnig=True)

            elif self.eeg.remote.get_state()[0] == 'CONFIGURED':
                setup = self.eeg.remote.get_setup()
                self.tiles.reset()
                for t in range(0, len(setup), 2):
                    self.tiles.addTile(setup[t], setup[t+1])
                self.control.toggle(runnig=False)
                self.tiles.toggle(runnig=False)
                
        pass
        
    def start(self):
        # collect setup..
        setup, valid = self.tiles.getSetup()
        if not valid:
            return
        applied, index, msg = self.eeg.apply_setup(setup)
        if applied :
            self.eeg.start()
            self.tiles.toggle()
            # self.control.toggle()
        else :
            self.emit(SIGNAL("reviseSetup(PyQt_PyObject)"), [(index-1)/2, setup[index-1], msg])
            
        self.tiles.update(applied=applied, index=(index-1)/2)
        
    def restart(self):
        self.eeg.stop()
        self.start()
        pass

    def stop(self):
        self.eeg.stop()
        self.tiles.toggle()
        # self.control.toggle()
        
    def close(self):
        if self.control.killOnQuit.isChecked():
            # kill eegmanager process
            self.eeg.shutdown()
        self.eeg.leave()
        self.parent.close()
    
    def load_from_pickle(self, filename):
        f = open(filename, 'rb')
        load = pickle.load(f)
        f.close()
        if not load.has_key('setup'):
            print "ERROR: SAVEFILE HAS NO SETUP EMBEDDED!"
            return
        print load['setup']
        print "created by %s on %s" % (load['user'], time.ctime(load['created']))
        self.tiles.reset()
        for t in range(0, len(load['setup']), 2):
            self.tiles.addTile(load['setup'][t], load['setup'][t+1])
    
    def save_to_pickle(self, filename, setup):
        save = dict()
        save['setup'] = setup
        save['created'] = time.time()
        save['user'] = getpass.getuser()
        f = open(filename, 'wb')
        pickle.dump(save, f)
        f.close()


class TileWidget(QWidget):
    """ Widget for displaying and instantiating modules and displaying them as 'tiles' """
    def __init__(self, parent=None, flags=0):
        super(TileWidget, self).__init__(parent=parent)
        
        self.parent = parent
        self.setAcceptDrops(True)
        self.buffer_widgets = None
        
        self.tcolumn = 0
        self.bcolumn = 1

        self.layout = QGridLayout(self)
        self.placeholder = QLabel("Drop Modules here")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("font: 24pt \"Helvetica Neue\"; color: lightgray;")
        self.layout.addWidget(self.placeholder, 0, 0)
        self.setLayout(self.layout)
        
        sp=self.sizePolicy()
        sp.setVerticalStretch(10)
        self.setSizePolicy(sp)
        
    def toggle(self):
        self.setAcceptDrops(not self.acceptDrops())

    def reset(self):
        self.placeholder.setVisible(True)
        for i in range(self.layout.count()):
            item = self.layout.takeAt(0)
            widget = item.widget()
            widget.hide()
            self.layout.removeWidget(widget)
            self.layout.removeItem(item)
            widget.parent = None
            item.parent = None
            del widget
            del item
        self.tcolumn = 0
        self.bcolumn = 1
        
    def dragMoveEvent(self, e):
        #print "move: ", e.pos()
        pass
        
    def dragEnterEvent(self, e):
        #print "enter: ", e.pos()
        if e.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            e.accept()
        else:
            # print e.format()
            e.ignore() 

    def dropEvent(self, e):
        bytearray = e.mimeData().data("application/x-qabstractitemmodeldatalist");
        #print "drop: ", e.pos()
        data = []
        item = {}
        ds = QDataStream(bytearray)
        while not ds.atEnd():
            row = ds.readInt32()
            column = ds.readInt32()
            map_items = ds.readInt32()
            for i in range(map_items):
                key = ds.readInt32()
                value = QVariant()
                ds >> value
                item[Qt.ItemDataRole(key)] = value
            data.append(item)
        self.placeholder.setVisible(False)
        modulename = data[0][Qt.DisplayRole].toString()
        self.addTile(modulename)
        
    def addTile(self, modulename, moduleparams=''):
        u = self.parent.eeg.usage(modulename)
        w = ModuleWidget(self, 
                         title=modulename,
                         usage=u)
        if moduleparams != '':
            w.param.setParams(moduleparams)
        self.placeholder.setVisible(False)
        self.layout.addWidget(w, 1, self.tcolumn, 1, 2)
        self.setLayout(self.layout)
        if self.tcolumn > 0:
            self.addBuffer()
        self.tcolumn += 2
    
    def addBuffer(self):
        self.layout.addWidget(BufferWidget(parent=self), 0, self.bcolumn, 1, 2)
        self.setLayout(self.layout)
        self.bcolumn += 2
        
    def getModuleWidgets(self):
        w = list()
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, ModuleWidget):
                w.append(widget)
        return w
        
    def getBufferWidgets(self):
        w = list()
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, BufferWidget):
                w.append(widget)
        return w
                
    def getSetup(self):
        s = list()
        w = self.getModuleWidgets()
        valid = True
        for i, widget in enumerate(w):
            if i == 0:
                if widget.type != 'input':
                    widget.title.status_fail()
                    widget.title.setErrorText("no input node at begin of flow!\n")
                    valid = False
            else:
                if widget.input != w[i-1].output:
                    # print "incompatabile nodes in series!", i, widget.input, w[i-1].output
                    w[i-1].title.status_fail()
                    w[i-1].title.setErrorText("output-type does not match input-type of next node!\n")
                    valid = False
                    
            if i == len(w)-1 and widget.type != 'output':
                # print "no output node at end of flow!"
                widget.title.status_fail()
                widget.title.setErrorText("no output node at end of flow!\n")
                valid = False
                
            widget.title.status_warn()
                
            s.append(str(widget.title.text()))
            s.append(widget.param.text())
        return s, valid
        
    def update(self, applied, index):
        w = self.getModuleWidgets()
        if not applied:
            w[index].title.status_fail()
        else:
            w[index].title.status_ok()
                
    def setupError(self, error):
        w = self.getModuleWidgets()
        for i, widget in enumerate(w):
            if i == error[0] and error[1] == widget.title.text():
                widget.title.setErrorText(error[2])
                
    def processState(self, state):
        if self.buffer_widgets == None:
            self.buffer_widgets = self.getBufferWidgets()
        n = state[1:]
        if state[0] != 'RUNNING':
            n = [0]*len(self.buffer_widgets)*2
        for i in range(len(n)/2):
            self.buffer_widgets[i].setValues(percent_full=n[2*i+1], kb_per_sec=n[2*i])
        if state[0] != 'RUNNING':
            self.buffer_widgets = None


class ModuleWidget(QWidget):
    """ Widget for displaying the module name (title) and the corresponding parameters """
    def __init__(self, parent=None, flags=0, title="default", usage=['[-f|--foo] <foo> --bar', '', 'stream', 'input']):
        super(ModuleWidget, self).__init__(parent=parent)
        self.setStyleSheet("")
        
        self.name = title
        
        self.layout = QVBoxLayout(self)
        self.usage = usage

        self.input = usage[1]
        self.output = usage[2]
        self.type = usage[3]
        
        self.title = TitleWidget(title=title, parent=self)
        self.param = ParamWidget(usage=self.usage[0], parent=self)

        self.layout.addWidget(self.title)
        self.layout.addWidget(self.param)
        
        self.layout.insertStretch(-1, 10)
        self.setLayout(self.layout)
        
        self.setMaximumSize(QSize(250, 2048))


class TitleWidget(QWidget):
    """ Widget for displaying the Title and a status icon for one module. """
    def __init__(self, title="default", icon="icons/20120611_eeg_gui_unknown", parent=None):
        super(TitleWidget, self).__init__(parent=parent)
        
        self.layout = QHBoxLayout(self)
        
        self.icon = QPushButton(self)
        self.icon.setIcon(QIcon(icon))
        self.icon.setMaximumSize(QSize(20, 20))
        self.icon.setFlat(True)
        self.icon.setEnabled(False)
        
        self.menu = QMenu()
        self.icon.setMenu(self.menu)
        
        self.title = QLabel(title, self)
        self.title.setStyleSheet("font: 24pt \"Helvetica Neue\";")
        
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.icon)
        
        self.setLayout(self.layout)
        
    def status_ok(self):
        self.menu.clear()
        self.icon.setEnabled(False)
        self.setIcon("icons/20120611_eeg_gui_ok")
    
    def status_fail(self):
        self.icon.setEnabled(True)
        self.setIcon("icons/20120611_eeg_gui_fail")
        
    def status_warn(self):
        self.icon.setEnabled(False)
        self.setIcon("icons/20120611_eeg_gui_warn")
        
    def setIcon(self, icon):
        self.icon.setIcon(QIcon(icon))
        
    def text(self):
        return self.title.text()
        
    def setErrorText(self, text):
        self.menu.clear()
        for t in text.split('\n'):
            if t == "":
                continue
            self.menu.addAction(t.lstrip('\t'))


class ParamWidget(QWidget):
    """ Widget for displaying the parameters of one module """
    def __init__(self, usage="", parent=None):
        super(ParamWidget, self).__init__(parent=parent)
                
        self.usage = usage
        self.layout = QVBoxLayout(self)
        self.setUp()
        self.setLayout(self.layout)
        self.parent = parent
        
    def setParams(self, params):
        p = params.split(' ')
        j = 0
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            widget = item.widget()
            if widget.objectName() == p[j].lstrip('-'):
                if isinstance(widget, QCheckBox):
                    widget.setChecked(True)
                    if (j+1) >= len(p):
                        return
                    j += 1
                elif isinstance(widget, QLineEdit):  
                    widget.setText(p[j+1])
                    if (j+2) >= len(p):
                        return
                    j += 2
                else:
                    pass
                
    def setUp(self):
        u = self.usage.split(' ')
        open_button = False
        
        for i in range(len(u)):
            if u[i] == "--meta":
                open_button = True
            elif u[i].startswith("--"):
                name = u[i][2:]
                cb = QCheckBox(name)
                cb.setObjectName(name)
                self.layout.addWidget(cb)
            elif u[i].startswith("["):
                name = u[i].split("|")[1][2:-1]
                self.layout.addWidget(QLabel(name))
                le = QLineEdit()
                le.setObjectName(name)
                #le.setPlaceholderText(name)
                le.setAlignment(Qt.AlignRight)
                le.setObjectName(name)
                self.layout.addWidget(le)
            else:
                continue
                
        if open_button :
            op = QPushButton("open")
            op.setObjectName("open")
            op.clicked.connect(self.open)
            self.layout.insertWidget(21, op)
            
    def open(self):
        if self.parent.name == "FILEAcquisition" :
            filename = str(QFileDialog.getOpenFileName(self, 'Open File', '.', "EEGManager-Files (*.eeg)"))
            if len(filename) > 0:
                self.set_widget_with_name(name="filename", text=filename)
        elif self.parent.name == "FLOWWorkspace" :            
            filename = str(QFileDialog.getOpenFileName(self, 'Open File', '.', "EEGManager-Files (*.yaml)"))
            if len(filename) > 0:
                self.parse_yaml(filename)
        elif self.parent.name == "FILEOutput" :
            filename = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if len(filename) > 0:
                self.set_widget_with_name(name="directory", text=filename)
    
    def parse_yaml(self, filename):
        print "parsing %s" % filename
        content = yaml.load(open(filename, "r"))
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            widget = item.widget()
            if widget.objectName() in content.keys():
                c = content[str(widget.objectName())]
                if isinstance(c, list):
                    s = ""
                    for l in c:
                        s += str("%s=%d;" % (l["name"], l["phys"]))
                    print s
                    widget.setText(s)
                else:
                    widget.setText(str(c))
                    
    def set_widget_with_name(self, name=None, text=None):
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QLineEdit):
                if widget.objectName() == name:
                    widget.setText(text)
            
    def text(self):
        t = ""
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QCheckBox):
                if widget.isChecked():
                    # print "--%s " % widget.text()
                    t += str("--%s " % widget.text())
            elif isinstance(widget, QLineEdit):
                if len(str(widget.text())) > 0:
                    # print "--%s %s " % (widget.objectName(), widget.text())
                    t += str("--%s %s " % (widget.objectName(), widget.text()))
            else:
                pass
        return t[:-1]


class BufferWidget(QWidget):
    """ Widget for displaying a buffer meter

    This includes a percent-full-bar and a bandwidth number.
    """
    def __init__(self, parent=None):
        super(BufferWidget, self).__init__(parent=parent)
        
        self.layout = QVBoxLayout(self)
        self.buffer = QProgressBar(self)
        self.buffer.setMaximum(1000)
        self.buffer.setValue(0)
        self.bandwidth = QLabel("MB/s", self)
        self.bandwidth.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.bandwidth)
        self.layout.addWidget(self.buffer)
        self.setLayout(self.layout)
        
    def setValues(self, percent_full=42, kb_per_sec=23):
        self.buffer.setValue(int(float(percent_full)*10.))
        self.bandwidth.setText(str("%3.2f Mb/s" % (float(kb_per_sec)/1024.)))
        

class ControlWidget(QWidget):
    """ Widget for controlling the underlying process and the whole GUI window """
    def __init__(self, parent=None):
        super(ControlWidget, self).__init__(parent=parent)
        
        self.layout = QHBoxLayout(self)
        self.parent = parent
        
        self.start = QPushButton("start", self)
        self.stop = QPushButton("stop", self)
        self.restart = QPushButton("restart", self)
        self.stop.setEnabled(False)
        self.restart.setEnabled(False)
        self.quit = QPushButton("quit", self)
        self.reset = QPushButton("reset", self)
        self.killOnQuit = QCheckBox("kill on quit", self)
        self.killOnQuit.setChecked(True)
        
        self.layout.addWidget(self.quit)
        self.layout.addWidget(self.killOnQuit)
        self.layout.insertStretch(2, 10)
        self.layout.addWidget(self.reset)
        self.layout.addWidget(self.stop)
        self.layout.addWidget(self.restart)
        self.layout.addWidget(self.start)
        
        self.setLayout(self.layout)
        
    def toggle(self, running=None):
        if running == None:
            self.start.setEnabled(not self.start.isEnabled())
            self.reset.setEnabled(not self.reset.isEnabled())
            self.stop.setEnabled(not self.stop.isEnabled())
            self.restart.setEnabled(not self.restart.isEnabled())
            self.parent.parent.actionOpen.setEnabled(not self.parent.parent.actionOpen.isEnabled())
            self.parent.parent.actionSave.setEnabled(not self.parent.parent.actionSave.isEnabled())
        elif isinstance(running, bool):
            self.start.setEnabled(not running)
            self.reset.setEnabled(not running)
            self.stop.setEnabled(running)
            self.restart.setEnabled(running)
            self.parent.parent.actionOpen.setEnabled(not running)
            self.parent.parent.actionSave.setEnabled(not running)


#####
##### Main 
#####

class MainWindow(QMainWindow):
    """ The main window of the EEGManager gui """

    def __init__(self, parent=None, scene_path=None):
        super(MainWindow, self).__init__(parent)
        self.setObjectName('MainWindow')
        self.resize(600, 480)
        
        self.menubar = QMenuBar(self)
        
        self.actionLibrary = QAction("Library", self)
        self.actionLibrary.setCheckable(True)
        self.actionLibrary.setChecked(True)
        
        self.actionConsole = QAction("Console", self)
        self.actionConsole.setCheckable(True)
        self.actionConsole.setChecked(False)
        
        self.centralwidget = CentralWidget(self, scene_path=scene_path)
        QObject.connect(self.centralwidget, SIGNAL("titleChanged(PyQt_PyObject)"), self.setTitle)
        self.centralwidget.update()
        
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setTitle("File")

        self.actionOpen = QAction("Open", self)
        self.actionOpen.triggered.connect(self.centralwidget.open)
        self.actionOpen.setShortcuts(QKeySequence.Open);
        self.actionSave = QAction("Save", self)
        self.actionSave.triggered.connect(self.centralwidget.save)
        self.actionSave.setShortcuts(QKeySequence.Save);
        self.actionConnect = QAction("Connect ..", self)
        self.actionConnect.triggered.connect(self.centralwidget.connect)
        self.actionConnect.setShortcuts(QKeySequence.Copy);
        
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionConnect)

        self.menubar.addAction(self.menuFile.menuAction())
        
        self.menuWindow = QMenu(self.menubar)
        self.menuWindow.setTitle("Window")
        
        # self.actionLibrary = QAction("Library", self)
        self.actionLibrary.triggered.connect(self.centralwidget.dockWidget.toggleEvent)
        # self.actionLibrary.setCheckable(True)
        # self.actionLibrary.setChecked(True)

        # self.actionConsole = QAction("Console", self)
        self.actionConsole.triggered.connect(self.centralwidget.console.toggleEvent)
        # self.actionConsole.setCheckable(True)
        # self.actionConsole.setChecked(False)
        
        self.menuWindow.addAction(self.actionConsole)
        self.menuWindow.addAction(self.actionLibrary)
        self.menubar.addAction(self.menuWindow.menuAction()) 
        
    def setTitle(self, text):
        self.setWindowTitle(text)  
        
    def setupUi(self):  
        self.setCentralWidget(self.centralwidget)
        self.setMenuBar(self.menubar)

        
    def cleanup(self):
        self.centralwidget.close()
        
if __name__ == "__main__":
    import sys
    
    scene_path = None
    if len(sys.argv) > 1:
        scene_path = sys.argv[1]
        
    app = QApplication(sys.argv)
    gui = MainWindow(scene_path=scene_path)
    gui.setupUi()
    gui.show()
    
    e = app.exec_()
    
    gui.cleanup()
    sys.exit(e)
    