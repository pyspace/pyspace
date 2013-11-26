#!/usr/bin/env python
""" Simple Gui for loading and browsing :mod:`pySPACE.resources.dataset_defs.performance_result`

For more documentation have a look at :ref:`tutorial_performance_results_analysis`!

.. note:: Due to import problems, matplotlib import was shifted to main method.
"""
import sys
import os
import copy
from PyQt4 import QtGui
from PyQt4 import QtCore

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Qt4Agg")
    
    #### Find pySPACE package and import it ####
    # Determine path of current file
    path = os.path.realpath(__file__)
    # Move up to parent directory that contains the pySPACE tree
    suffix = []
    for i in range(4):
        path, tail = os.path.split(path)
        suffix.append(tail)
    parent_dir = path
    
    # Check proper directory structure
    if suffix != ['performance_results_analysis.py','gui','run', 'pySPACE']:
        raise RuntimeError, "Encountered incorrect directory structure. " \
            "This script needs to reside in $pySPACE_PARENT_DIR/pySPACE/run/gui"
    
    # Append pySPACE root directory to PYTHONPATH
    sys.path.append(parent_dir)
    #########################################

from pySPACE.resources.dataset_defs.performance_result import PerformanceResultSummary
from pySPACE.resources.dataset_defs.performance_result import ROCCurves


class ProjectionPopup(QtGui.QMainWindow):
    """ A pop up window that allows to specify the projection of the data.
    
    This pop up window allows to select a parameter and its desired value.
    The result collection currently loaded in *main_window* is then projected
    onto the subset where the chosen parameter has the chosen value.    
    """
    
    def __init__(self, main_window, *args, **kwargs):
        super(ProjectionPopup, self).__init__(main_window)
        
        self.main_window = main_window
        
        self.selected_variable = None
        self.selected_values = None
        
        self.resize(640,480)
        self.setWindowTitle('Project onto')
        
        self._create_central_widget()

    def _create_central_widget(self):
        """ Create central widget of the pop-up"""
        self.central_widget = QtGui.QWidget()
        
        # Create a list widget where the parameter can be selected
        self.variables_view = QtGui.QListWidget(self)
        self.variables_view.setSelectionMode(QtGui.QListView.SingleSelection)
        for variable_name in sorted(self.main_window.current_collection.get_gui_variables()):
            item = QtGui.QListWidgetItem(variable_name, self.variables_view)
        
        # Create a list widget where the parameter's value can be selected
        self.values_view = QtGui.QListWidget(self)
        # self.values_view.setSelectionMode(QtGui.QListView.MultiSelection)
        self.values_view.setSelectionMode(QtGui.QListView.ExtendedSelection)
        
        # Connect signals and slots (handler functions)
        self.connect(self.variables_view, 
                     QtCore.SIGNAL('currentItemChanged (QListWidgetItem *,QListWidgetItem *)'), 
                     self._variable_selected)
        self.connect(self.values_view, 
                     QtCore.SIGNAL('itemPressed (QListWidgetItem *)'), 
                     self._value_selected)
        self.connect(self.values_view, 
                     QtCore.SIGNAL('currentItemChanged (QListWidgetItem *,QListWidgetItem *)'), 
                     self._value_selected)
        
        # Add button that need to be pressed to perform projection
        self.project_button = QtGui.QPushButton("&Project")
        self.connect(self.project_button, QtCore.SIGNAL('clicked()'), 
                     self._project)
        
        # Add button that cancels projection when pressed
        self.cancel_button = QtGui.QPushButton("&Cancel")
        self.connect(self.cancel_button, QtCore.SIGNAL('clicked()'), 
                     self._cancel)
        
        # Add all to the central widgets layout
        hlayout = QtGui.QHBoxLayout()
        hlayout.addWidget(self.variables_view)
        hlayout.addWidget(self.values_view)
        vlayout = QtGui.QVBoxLayout()
        vlayout.addWidget(self.project_button)
        vlayout.addWidget(self.cancel_button)
        hlayout.addLayout(vlayout)
        
        self.central_widget.setLayout(hlayout)
        self.setCentralWidget(self.central_widget)
        
        self.show()
    
    def _variable_selected(self, selected, deselected):
        """ Handle the selection of a new parameter by the user"""
        self.selected_variable = str(selected.text())
        
        # Add the values of the chosen parameter to the value selection list
        self.values_view.clear()
        for value in sorted(self.main_window.current_collection.get_parameter_values(self.selected_variable)):                  
            item = QtGui.QListWidgetItem('%s' % value, self.values_view)
        
        self.show()
    
    def _value_selected(self, selected=None, deselected=None):
        """ Handle the selection of a new value by the user"""
        self.selected_values = self.values_view.selectedItems()
    
    def _project(self):
        """Called when user presses "project" button. Performs the actual projection."""
        # Check that parameter and value have actually been selected
        if self.selected_variable is None or self.selected_values is None:
            warning_box = QtGui.QMessageBox()
            warning_box.setText("A parameter and its value must be selected.")
            warning_box.exec_()
            return
        
        self.selected_values = [str(x.text()) for x in self.selected_values]
        self.main_window._project_onto(self.selected_variable,
                                       self.selected_values)
        self.close()

    def _cancel(self):
        """ Cancels projection """
        self.close()


class PerformanceResultsAnalysisWidget(QtGui.QWidget):
    """ The main window of the performance results analysis GUI
    
    **Parameters**
    
        :swap: Switch between zero and one to change different
               between two different printing cases.
               
               This is yet implemented in nominal vs. nominal plots 
               to switch in the presentation which parameter defines the main
               differentiation
               and
               in numeric vs. nominal to activate some dependent display 
               of functions with a hidden parameter, to show performance
               differences in time and `Balanced_accuracy`.
    """
    
    def __init__(self, results_file=None, parent=None):
        super(PerformanceResultsAnalysisWidget, self).__init__(parent)
                
        # Load a results csv file
        self._load_results_collection_from_file(results_file)
        # Try to load collection of ROC curves (may not always be available)
        self._load_roc_curves()
        
        # Create elements of this widget
        self._create_elements()
        
        # The currently selected projection parameters (required for ROC)
        self.projection_parameters = {}
        
        # To be able to swap the role of the parameters in nom_vs_nom plots
        self.swap = 1
        self.swap2 = 1
        self.save_path = "./"

    def _load_results_collection_from_file(self, file_name=None):
        """ Load results collection from file  """
        if file_name is None:
            # Let the user specify a file to be loaded
            self.file_name = \
                str(QtGui.QFileDialog.getOpenFileName(
                    parent=self, caption="Select a results file",
                    filter="results files (*.csv)"))
        else:
            self.file_name = file_name
        # Try to load specified file 
        dirname, filename = os.path.split(self.file_name)      
        self.result_collection = PerformanceResultSummary(dataset_dir=dirname,
                                                          csv_filename=filename)
        # Create working copy that can be modified
        self.current_collection = copy.deepcopy(self.result_collection)

    def _load_roc_curves(self):
        """ Load the collection of ROC curves. Max not always be available. """
        self.roc_curves = ROCCurves(base_path=os.path.dirname(self.file_name))

    def _create_elements(self):
        """ Create elements of this widget"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        
        # Create model and view of the variable selection
        # NOTE: One or two variables can be selected
        self.variables_model = QtGui.QStandardItemModel()
        self.variables_items = []
        for variable_name in sorted(self.current_collection.get_gui_variables()):
            item = QtGui.QStandardItem('%s' % variable_name)
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setData(QtCore.QVariant(QtCore.Qt.Unchecked),
                         QtCore.Qt.CheckStateRole)
            self.variables_model.appendRow(item)
            self.variables_items.append(item)
        
        self.variables_view = QtGui.QListView(self)
        self.variables_view.setModel(self.variables_model)
        
        # Create metrics selection list widget
        self.metrics_view = QtGui.QListWidget(self)
        self.metrics_view.setSelectionMode(QtGui.QListView.SingleSelection)
        self.metrics_items = []
        if not self.roc_curves.is_empty():  # If we can plot ROC curves
            self.metrics_items.append(QtGui.QListWidgetItem("ROC Curve",
                                                            self.metrics_view))
        for metric_name in sorted(self.current_collection.get_gui_metrics()):
            item = QtGui.QListWidgetItem('%s' % metric_name, self.metrics_view)
            self.metrics_items.append(item)
        self.metrics_items.append(QtGui.QListWidgetItem("Cost function",
                                                        self.metrics_view))
        
        # Add cost function box
        self.fpcost_label = QtGui.QLabel("False Positive Cost")
        self.fpcost_line_edit = QtGui.QLineEdit("1.0")
        self.fncost_label = QtGui.QLabel("False Negative Cost")
        self.fncost_line_edit = QtGui.QLineEdit("1.0") 
        
        # Create various buttons and connect them with the handler functions
        self.load_button = QtGui.QPushButton("&Load")
        self.connect(self.load_button, QtCore.SIGNAL('clicked()'), 
                     self._reload) #self._load_results_collection_from_file)
        
        self.draw_button = QtGui.QPushButton("&Draw or Toggle")
        self.connect(self.draw_button, QtCore.SIGNAL('clicked()'), 
                     self._draw_plot)
        
        self.hist_button = QtGui.QPushButton("&Histogram")
        self.connect(self.hist_button, QtCore.SIGNAL('clicked()'), 
                     self._draw_histogram)
        
        self.project_button = QtGui.QPushButton("&Project onto")
        self.connect(self.project_button, QtCore.SIGNAL('clicked()'), 
                     self._project_popup)
        
        self.save_button = QtGui.QPushButton("&Save")
        self.connect(self.save_button, QtCore.SIGNAL('clicked()'), self._save)
        
        self.reset_button = QtGui.QPushButton("&Reset")
        self.connect(self.reset_button, QtCore.SIGNAL('clicked()'), self._reset)
        
        # Create matplotlib canvas 
        self.fig = Figure((12.0, 8.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        
        # Create axes for plot
        self.axes = self.fig.add_subplot(111)
        
        # Text showing projection parameters
        self.project_params_label = QtGui.QLabel("No projections.")
        self.project_params_label.setWordWrap(1)
        self.project_params_label.setFixedWidth(self.canvas.width())
        
        # Create layout of widget
        vlayout1 = QtGui.QVBoxLayout()
        hlayout1 = QtGui.QHBoxLayout()
        vlayout2 = QtGui.QVBoxLayout()
        vlayout2.addWidget(self.variables_view)
        vlayout2.addWidget(self.metrics_view)
        vlayout2.addWidget(self.fpcost_label)
        vlayout2.addWidget(self.fpcost_line_edit)
        vlayout2.addWidget(self.fncost_label)
        vlayout2.addWidget(self.fncost_line_edit)
        hlayout1.addLayout(vlayout2)
        hlayout1.addWidget(self.canvas)
        vlayout1.addLayout(hlayout1)
        
        hlayout2 = QtGui.QHBoxLayout()
        hlayout2.addWidget(self.load_button)
        hlayout2.addWidget(self.draw_button)
        hlayout2.addWidget(self.hist_button)
        hlayout2.addWidget(self.project_button)
        hlayout2.addWidget(self.save_button)
        hlayout2.addWidget(self.reset_button)
        
        vlayout1.addWidget(self.project_params_label)
        vlayout1.addLayout(hlayout2)
        
        self.setLayout(vlayout1)

    def _project_onto(self, selected_variable, selected_values):
        """ Project onto the data where selected_variable has the values selected_values."""
        self.current_collection = \
            self.current_collection.project_onto(selected_variable,
                                                 selected_values)
        # Try if we can evaluate the value
        try:
            selected_values = eval(selected_values)
        except:
            pass
        # Remember projection for ROC curves    
        self.projection_parameters[selected_variable] = selected_values
        
        # Update projections label
        self.project_params_label.setText("Projections: " +
                                          str(self.projection_parameters))
        self.project_params_label.adjustSize()
        
        # Update selection box
        self._update_variable_selection()

    def _update_variable_selection(self):
        """ Updates the selection box for variables"""
        self.variables_model.clear()
        self.variables_items = []
        
        for variable_name in sorted(self.current_collection.get_gui_variables()):
            item = QtGui.QStandardItem('%s' % variable_name)
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setData(QtCore.QVariant(QtCore.Qt.Unchecked),
                         QtCore.Qt.CheckStateRole)
            self.variables_model.appendRow(item)
            self.variables_items.append(item)

    def _save(self):
        """Stores the current figure to a file"""
        dic = self.fig.canvas.get_supported_filetypes();
	if "pdf" in dic:
            extensionList = ["%s (*.pdf)" % dic["pdf"]]
        else:
            extensionList = []

        for ext,desc in dic.items():
            if (ext != "pdf"):
                extensionList.append("%s (*.%s)" % (desc,ext))
	selectionList = ";;".join(extensionList)

        file_name = \
            str(QtGui.QFileDialog.getSaveFileName(
                self, "Select a name for the graphic", self.save_path,
                selectionList))
        self.save_path = os.path.dirname(file_name)
        self.fig.savefig(file_name, dpi=400)

    def _reset(self):
        """ Reset working collection to originally loaded one"""
        self.current_collection = copy.deepcopy(self.result_collection)
        self.projection_parameters = {}
        self.project_params_label.setText("No projections.")
        self._update_variable_selection()

        self.metrics_items = []
        self.metrics_view.clear()
        if not self.roc_curves.is_empty():  # If we can plot ROC curves
            self.metrics_items.append(QtGui.QListWidgetItem("ROC Curve",
                                                            self.metrics_view))
        for metric_name in sorted(self.current_collection.get_gui_metrics()):
            item = QtGui.QListWidgetItem('%s' % metric_name, self.metrics_view)
            self.metrics_items.append(item)
        self.metrics_items.append(QtGui.QListWidgetItem("Cost function",
                                                        self.metrics_view))

    def _reload(self):
        """ Reinitialize and load new result file """
        self._load_results_collection_from_file()
        self._load_roc_curves()
        self._reset()

    def _draw_plot(self):
        """Draw a plot for the selected variable/metric combination. """
        # Determine selected metric
        selected_metric = None
        for item in self.metrics_items:
            if item.isSelected():
                selected_metric = str(item.text())
                break
            
        if selected_metric is None:
            warning_box = QtGui.QMessageBox()
            warning_box.setText("A metric must be selected.")
            warning_box.exec_()
            return
               
        # Determine selected variables
        selected_variables = self._get_selected_items(self.variables_items)
        if not 0 < len(selected_variables) <= 2 and selected_metric != "ROC Curve":
            warning_box = QtGui.QMessageBox()
            warning_box.setText("One or two variables must be selected.")
            warning_box.exec_()
            return
        if len(selected_variables) > 1 and selected_metric == "ROC Curve":
            warning_box = QtGui.QMessageBox()
            warning_box.setText("At most one variable can be selected for ROC curves.")
            warning_box.exec_()
            return

        # The "metric" ROC curve" needs a special treatment
        if selected_metric == "ROC Curve":
            selected_variable = None if selected_variables == [] else selected_variables[0]
            fpcost = eval(str(self.fpcost_line_edit.text()))
            fncost = eval(str(self.fncost_line_edit.text()))
            self.fig.clear()
            self.axes = self.fig.add_subplot(111) 
            self.roc_curves.plot(self.axes, selected_variable=selected_variable,
                                 projection_parameter=self.projection_parameters,
                                 fpcost=fpcost, fncost=fncost,
                                 collection=self.current_collection)
            self.canvas.draw()
            return
        elif selected_metric == "Cost function": # needs special treatment
            selected_metric = "#".join([str(self.fpcost_line_edit.text()),
                                       "False_positives",
                                       str(self.fncost_line_edit.text()),
                                       "False_negatives"])
        
        # Determine nominal and numeric parameters of the loaded table
        variables = self.current_collection.get_gui_variables()
        nominal_parameters = \
            list(self.current_collection.get_nominal_parameters(variables))
        numeric_parameters = \
            list(self.current_collection.get_numeric_parameters(variables))
        
        # Do the actual plotting
        self.fig.clear()
        self.axes = self.fig.add_subplot(111) 
        if len(selected_variables) == 1:
            if selected_variables[0] in nominal_parameters:
                self.current_collection.plot_nominal(self.axes,
                                                     selected_variables[0], 
                                                     selected_metric)
            else:
                self.current_collection.plot_numeric(self.axes,
                                                    selected_variables[0], 
                                                    selected_metric)
        else:
            # Canonical order: Numeric parameters have to be first
            if selected_variables[0] in nominal_parameters:
                selected_variables = [selected_variables[1], selected_variables[0]]
            # Plot for two nominal variables
            if selected_variables[0] in nominal_parameters:
                # For every click on Draw, swap the role of the parameters
                selected_variables=sorted(selected_variables)
                self.swap = 1 - self.swap
                if self.swap == 0:
                    selected_variables[0], selected_variables[1] = \
                    selected_variables[1], selected_variables[0]
                self.current_collection.plot_nominal_vs_nominal(self.axes,
                                                                selected_variables[0],
                                                                selected_variables[1],
                                                                selected_metric)
            elif selected_variables[1] in nominal_parameters:
                self.swap = 1 - self.swap
                if self.swap:
                    self.swap2 = 1 - self.swap2
                self.current_collection.plot_numeric_vs_nominal(self.axes,
                                                                selected_variables[0],
                                                                selected_variables[1], 
                                                                selected_metric,
                                                                self.swap,
                                                                self.swap2)
            else:
                self.swap = 1 - self.swap
                self.current_collection.plot_numeric_vs_numeric(self.axes,
                                                                selected_variables, 
                                                                selected_metric,
                                                                self.swap)
        self.canvas.draw()

    def _draw_histogram(self):
        """ Draw a histogram of the current collection for the specified metric """
        # Determine selected variables
        selected_variables = self._get_selected_items(self.variables_items)
        
        # Determine selected metric
        selected_metric = None
        for item in self.metrics_items:
            if item.isSelected():
                selected_metric = str(item.text())
                break

        if selected_metric is None:
            warning_box = QtGui.QMessageBox()
            warning_box.setText("A metric must be selected.")
            warning_box.exec_()
            return

        # Do the actual plotting
        self.fig.clear()
        self.axes = self.fig.add_subplot(111) 

        self.current_collection.plot_histogram(self.axes, selected_metric,
                                               selected_variables, [])
        self.canvas.draw()
        
    def _project_popup(self):
        """ Create 'project onto' pop up window """
        popup_frame = ProjectionPopup(self)
        
    def _get_selected_items(self, items):
        """ Determine selected items from a list of items """
        selected_items = []
        for item in items:
            if item.checkState() != 0:
                selected_items.append(str(item.text()))
        return selected_items


class PerformanceResultsAnalysisMainWindow(QtGui.QMainWindow):
    """The main window for analysis"""
    
    def __init__(self, results_file=None, parent=None):
        super(PerformanceResultsAnalysisMainWindow, self).__init__(parent)
        
        self.resize(1024, 768)
        self.setWindowTitle('results analysis ')
        
        self.central_widget = PerformanceResultsAnalysisWidget(
            results_file=results_file, parent=self)

        self.setCentralWidget(self.central_widget)

if __name__ == '__main__':
    #Creating Qt application
    app = QtGui.QApplication(sys.argv)
    
    if len(sys.argv) > 1:
        performance_results_analysis = PerformanceResultsAnalysisMainWindow(sys.argv[1])
    else:
        performance_results_analysis = PerformanceResultsAnalysisMainWindow()
    performance_results_analysis.show()
    
    #Initializing application
    sys.exit(app.exec_())

