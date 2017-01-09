""" Rough basic interface to run pySPACE with node chains """
import logging

import os
import sys

# import for documentation generation
from threading import Thread
import time

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

import pySPACE

from pySPACE import create_operation_from_file, create_operation, create_backend

try:
    from PyQt4 import QtGui, QtCore
except:
    pass

if __name__ == "__main__":
    #### Find pySPACE package and import it ####
    # Determine path of current file
    path = os.path.abspath(__file__)
    
    # Move up to parent directory that contains the pySPACE tree
    suffix = []
    for i in range(4):
        path, tail = os.path.split(path)
        suffix.append(tail)
    parent_dir = path
    
    # Check proper directory structure
    if suffix != ['general_gui.py', 'gui','run', 'pySPACE']:
        raise RuntimeError, "Encountered incorrect directory structure. " \
            "general_gui.py needs to reside in $PARENT_DIR/pySPACE/run/gui"
    
    # Append pySPACE root directory to PYTHONPATH
    sys.path.append(parent_dir)
    #########################################

    try:
        from PyQt4 import QtGui, QtCore, QtWebKit
    except ImportError:
        import warnings
        warnings.warn("ERROR: This GUI requires the PyQt4 package. You may use "\
              "launch.py instead however.")
        sys.exit(0)

import pySPACE

class PySpaceGui(QtGui.QMainWindow):
    """The main window of the GUI"""
    
    def __init__(self, parent=None):
        super(PySpaceGui, self).__init__(parent)
        
        self.setWindowTitle('YAML based Signal Processing And Classification Environment written in Python (pySPACE)')
        self.resize(1024, 768)
        
    def set_up(self):
        """ Setting up of GUI components """
        # Lazy import of widgets

        # Putting everything together
        self.tabWidget = QtGui.QTabWidget(self)
        self.tabWidget.setTabsClosable(True)
        self.connect(self.tabWidget, QtCore.SIGNAL('tabCloseRequested (int)'), 
                     self._closeTab)
          
        # The tab for configuring and controlling operations
        self.ControlTab = ControlWidget(self)
        self.tabWidget.addTab(self.ControlTab, "Control")
                
        # The tabs that show the documentation
        # TODO: Currently, it is not possible to login via the viewer such that
        #       the desired page can not be displayed. Thus, we use the local
        #       documentation
        DocumentationTab = QtWebKit.QWebView(self)
        DocumentationTab.load(QtCore.QUrl("../../../docs/.build/html/index.html"))
        self.tabWidget.addTab(DocumentationTab, "pySPACE Documentation")
        self.connect(DocumentationTab.page().networkAccessManager(), 
                     QtCore.SIGNAL("sslErrors (QNetworkReply *, const QList<QSslError> &)"), 
                     self._sslErrorHandler)
        
        # Set initially selected tab
        self.tabWidget.setCurrentWidget(self.ControlTab)
        self.setCentralWidget(self.tabWidget)
        
        # Connect handler for signal that signals that an operation has been finished
        self.ControlTab.operationFinishedSignal.connect(self._operation_finished)
        
        # Implement proper shutdown
        self.connect(self, QtCore.SIGNAL('quit()'), self._close_event)
        
    def _operation_finished(self, result_file_name):
        # Called whenever an operation is finished
        from pySPACE.run.gui.performance_results_analysis import PerformanceResultsAnalysisWidget
        
        # Convert special string "None" to object None
        if result_file_name == "None": result_file_name = None
        
        # Open results analysis tab
        resultsAnalysisTab = \
                PerformanceResultsAnalysisWidget(results_file=result_file_name,
                                         parent=self)
        self.tabWidget.addTab(resultsAnalysisTab, "Result Analysis")
        self.tabWidget.setCurrentWidget(resultsAnalysisTab)

    def _close_event(self, event):
        super(PySpaceGui, self).closeEvent(event)
        
    def _sslErrorHandler(self, reply, errorList):
        # Ignore all SSL errors....
        reply.ignoreSslErrors()
        print "SSL error ignored: %s" % errorList[0].errorString()
        
    def _closeTab(self, tabIndex):
        # The control and documentation tabs must not be closed!
        if tabIndex <= 2: return 
        self.tabWidget.widget(tabIndex).close()
        self.tabWidget.removeTab(tabIndex)
            

if __name__ == '__main__':
    #Creating Qt application
    app = QtGui.QApplication(sys.argv) 
    gui = PySpaceGui()

    # Let user select a configuration file
    config_file = \
        str(QtGui.QFileDialog.getOpenFileName(gui, "Select a configuration file ",
                                              os.sep.join([parent_dir, "docs","examples", "conf"]),
                                              "configuration files (*.yaml)"))
    config_file = config_file.split(os.sep)[-1]
     
    # Load configuration file
    import pySPACE
    pySPACE.load_configuration(config_file)

    gui.set_up()
    gui.show()
 
    #Initing application
    sys.exit(app.exec_())

class ControlWidget(QtGui.QWidget):
    """ Widget for configuring a node chain operation and monitor its progress. """

    # Two signals that are used to provide the GUI with the status of the
    # running operation.
    operationStatusChangedSignal = \
                QtCore.pyqtSignal(int, name='operationStatusChanged')

    operationFinishedSignal = \
                QtCore.pyqtSignal(basestring, name='operationFinished')

    def __init__(self, results_file=None, parent=None):
        super(ControlWidget, self).__init__(parent)

        self.operation = None

        # hlayoutBackend
        hlayoutBackend = QtGui.QHBoxLayout()

        backendLabel = QtGui.QLabel("Backend")
        self.backendComboBox = QtGui.QComboBox(self)
        self.backendComboBox.addItems(["serial", "mcore", "mpi"])
        self.backendComboBox.setToolTip("Select the backend")

        hlayoutBackend.addWidget(backendLabel)
        hlayoutBackend.addWidget(self.backendComboBox)

        # hlayoutOperationType
        hlayoutOperationType = QtGui.QHBoxLayout()

        operationTypeLabel = QtGui.QLabel("Operation")
        self.operationTypeComboBox = QtGui.QComboBox(self)
        self.operationTypeComboBox.addItems(["Node Chain Operation"])
        self.operationTypeComboBox.setToolTip("Select the type of operation")
        opConfigButton = QtGui.QPushButton("&Create")
        self.connect(opConfigButton, QtCore.SIGNAL('clicked()'),
                     self._config_node_chain)
        opLoadButton = QtGui.QPushButton("&Load")
        self.connect(opLoadButton, QtCore.SIGNAL('clicked()'),
                     self._load_node_chain)

        hlayoutOperationType.addWidget(operationTypeLabel)
        hlayoutOperationType.addWidget(self.operationTypeComboBox)
        hlayoutOperationType.addWidget(opConfigButton)
        hlayoutOperationType.addWidget(opLoadButton)

        # start button
        startButton = QtGui.QPushButton("&Start Operation")
        self.connect(startButton, QtCore.SIGNAL('clicked()'),
                     self._start)

        # load operation button
        loadResultsButton = QtGui.QPushButton("&Load Operation Results")
        self.connect(loadResultsButton, QtCore.SIGNAL('clicked()'),
                     self._loadResults)

        # control bar
        hlayoutControl = QtGui.QHBoxLayout()
        hlayoutControl.addWidget(startButton)
        hlayoutControl.addWidget(loadResultsButton)


        # the log viewer
        logViewer = LogViewer(self)

        # operation progress
        hlayoutProgress = QtGui.QHBoxLayout()
        progressLabel = QtGui.QLabel("Operation progress")
        self.progressBar = QtGui.QProgressBar()
        self.etaLabel = \
            QtGui.QLabel("ETA: --:--:--")
        self.operationStatusChangedSignal.connect(self._update_progress_bar)
        hlayoutProgress.addWidget(progressLabel)
        hlayoutProgress.addWidget(self.progressBar)
        hlayoutProgress.addWidget(self.etaLabel)

        # Create the main layout of widget
        vlayout = QtGui.QVBoxLayout()
        vlayout.addLayout(hlayoutBackend)
        vlayout.addLayout(hlayoutOperationType)
        vlayout.addLayout(hlayoutControl)
        vlayout.addWidget(logViewer)
        vlayout.addLayout(hlayoutProgress)

        self.setLayout(vlayout)

    def _config_node_chain(self):
        # Launch window that allows to load/configure a node chain operation
        node_chain_config_frame = OperationConfigurationPopup(self)

    def _load_node_chain(self):
        # Let the user select an operation
        operation_file = \
            QtGui.QFileDialog.getOpenFileName(self, "Select an operation spec file",
                                              pySPACE.configuration.spec_dir + os.sep + "operations",
                                              "operation specification files (*.yaml)")
        if operation_file == None:
            return

        # Create operation
        self.operation = create_operation_from_file(str(operation_file))

    def _start(self):
        # Set number of processes in operation
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(self.operation.number_processes)
        self.progressBar.setValue(0)

        # Create backend
        self.backend = create_backend(str(self.backendComboBox.currentText()))

        # Run operation in a separate thread to keep GUI responsive
        self.operationThread = Thread(target=pySPACE.run_operation,
                                      args=(self.backend, self.operation))
        self.operationThread.start()

        # Update progress bar in a separate thread
        self.progressBarThread = Thread(target=self._monitor_progress, args=())
        self.progressBarThread.start()


    def _monitor_progress(self):
        while self.operationThread.isAlive():
            self.operationStatusChangedSignal.emit(self.backend.current_process)
            time.sleep(0.1)

        self.operationStatusChangedSignal.emit(self.backend.current_process)
        self.operationFinishedSignal.emit(self.operation.result_directory + os.sep + "results.csv")


    def _update_progress_bar(self, processesFinished):
        self.progressBar.setValue(processesFinished)

        if hasattr(self.backend, "progress_bar"):
            # Compute Estimated Time of Arrival
            pbar = self.backend.progress_bar
            if pbar.currval == 0:
                self.etaLabel.setText("ETA: --:--:--")
            elif pbar.finished:
                self.etaLabel.setText(time.strftime('%H:%M:%S', time.gmtime(pbar.seconds_elapsed)))
            else:
                elapsed = pbar.seconds_elapsed
                eta = elapsed * pbar.maxval / pbar.currval - elapsed
                self.etaLabel.setText("ETA: %s"
                                        % time.strftime('%H:%M:%S', time.gmtime(eta)))

    def _loadResults(self):
        # Load results of an already executed operation
        # This is done by pretending that an operation was finished but without
        # a results file. This causes the results analysis window to ask for the
        # location of the csv file.
        self.operationFinishedSignal.emit("None")


class OperationConfigurationPopup(QtGui.QMainWindow):
    """ Window for configuring a NodeChainOperation.

    Allows to select an input and a node chain. Additionally, the window
    allows to invoke the configuration GUI.
    """

    def __init__(self, main_window, *args, **kwargs):
        super(OperationConfigurationPopup, self).__init__(main_window)

        self.main_window = main_window

        self.selected_input = None

        self.resize(640,480)
        self.setWindowTitle('Operation Configuration')

        self.central_widget = QtGui.QWidget()

        # hlayoutData
        hlayoutData = QtGui.QHBoxLayout()

        self.DataLabel = QtGui.QLabel("Input: %s" % self.selected_input)
        DataButton = QtGui.QPushButton("&Select")
        self.connect(DataButton, QtCore.SIGNAL('clicked()'),
                     self._select_input)

        hlayoutData.addWidget(self.DataLabel)
        hlayoutData.addWidget(DataButton)

        # hlayoutNodeChain
        hlayoutNodeChain = QtGui.QHBoxLayout()

        NodeChainLabel = QtGui.QLabel("Node Chain")
        self.NodeChainComboBox = QtGui.QComboBox(self)
        self.NodeChainComboBox.addItems(os.listdir(pySPACE.configuration.spec_dir + os.sep + "node_chains"))
        self.NodeChainComboBox.setToolTip("Select the node chain to be used in the node chain operation")
        NodeChainCreateButton = QtGui.QPushButton("&Create")
        self.connect(NodeChainCreateButton, QtCore.SIGNAL('clicked()'),
                     self._configure_NodeChain)
        NodeChainRefreshButton = QtGui.QPushButton("&Refresh")
        self.connect(NodeChainRefreshButton, QtCore.SIGNAL('clicked()'),
                     self._refresh_node_chains)

        hlayoutNodeChain.addWidget(NodeChainLabel)
        hlayoutNodeChain.addWidget(self.NodeChainComboBox)
        hlayoutNodeChain.addWidget(NodeChainCreateButton)
        hlayoutNodeChain.addWidget(NodeChainRefreshButton)

        # Create button
        operationCreateButton = QtGui.QPushButton("&Create")
        self.connect(operationCreateButton, QtCore.SIGNAL('clicked()'),
                     self._create_operation)

        # Main layout
        vlayout = QtGui.QVBoxLayout()
        vlayout.addLayout(hlayoutData)
        vlayout.addLayout(hlayoutNodeChain)
        vlayout.addWidget(operationCreateButton)

        self.central_widget.setLayout(vlayout)
        self.setCentralWidget(self.central_widget)

        self.show()

    def _select_input(self):
        _input_path = \
            str(QtGui.QFileDialog.getExistingDirectory(self,
                                                       "Select your input",
                                                       pySPACE.configuration.storage))
        assert(_input_path.startswith(pySPACE.configuration.storage))

        self.selected_input = _input_path[len(pySPACE.configuration.storage)+1:]
        self.DataLabel.setText("Input: %s" % self.selected_input)

    def _configure_node_chain(self):
        from pySPACE.run.gui.node_chain_GUI import NodeChainConfigurationWidget

        NodeChainConfigurationPopup = QtGui.QMainWindow(self)
        NodeChainConfigurationWindow = \
                NodeChainConfigurationWidget(pySPACE.configuration.spec_dir + os.sep + "node_chains",
                                            NodeChainConfigurationPopup)

        NodeChainConfigurationPopup.setCentralWidget(NodeChainConfigurationWindow)
        NodeChainConfigurationPopup.setWindowTitle('NodeChain Configuration')
        NodeChainConfigurationPopup.show()

    def _refresh_node_chains(self):
        self.NodeChainComboBox.clear()
        self.NodeChainComboBox.addItems(os.listdir(pySPACE.configuration.spec_dir + os.sep + "node_chains"))

    def _create_operation(self):
        operation_spec = {"type": "node_chain",
                          "input": self.selected_input,
                          "templates" : [str(self.NodeChainComboBox.currentText())],
                          "store_node_chain" : False}
        self.main_window.operation= create_operation(operation_spec)

        self.close()


class QtStreamHandler(logging.Handler):
    """ Handle incoming text streaming input """

    def __init__(self, parent,  main):
        logging.Handler.__init__(self)
        self.parent = parent
        self.main = main

        self.textWidget = parent
        self.formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')

        self.buffer = ""
        self.lastUpdateTime = time.time()

    def emit(self, record):
        self.buffer += self.formatter.format(record) + "\n"

        if time.time() - self.lastUpdateTime > 0.1:
            self.textWidget.insertPlainText(self.buffer)
            self.textWidget.moveCursor(QtGui.QTextCursor.End)

            self.buffer = ""
            self.lastUpdateTime = time.time()


class LogViewer(QtGui.QWidget):
    """ Handle logging input """

    def __init__(self, parent=None):
        super(LogViewer, self).__init__(parent)

        # Log levels
        LEVELS = {'DEBUG': logging.DEBUG,
                  'INFO': logging.INFO,
                  'WARNING': logging.WARNING,
                  'ERROR': logging.ERROR,
                  'CRITICAL': logging.CRITICAL}

        # The actual logger object
        self.logger = logging.getLogger('')

        # Create log text field
        logTextField = QtGui.QTextEdit()
        self.logHandler = QtStreamHandler(logTextField,  self)
        self.logHandler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.logHandler)

        # Create combobox for selecting the log level
        logLevelLabel = QtGui.QLabel("Log level")
        logLevelComboBox = QtGui.QComboBox(self)
        logLevelComboBox.addItems(["DEBUG", "INFO", "WARNING", 'ERROR', 'CRITICAL'])

        def updateLogLevel(level):
            self.logHandler.setLevel(LEVELS[str(level)])

        self.connect(logLevelComboBox,
                     QtCore.SIGNAL('activated (const QString&)'),
                     updateLogLevel)

        # Create layout
        layout = QtGui.QVBoxLayout()
        hlayout = QtGui.QHBoxLayout()
        hlayout.addWidget(logLevelLabel)
        hlayout.addWidget(logLevelComboBox)
        layout.addLayout(hlayout)
        layout.addWidget(logTextField)
        self.setLayout(layout)