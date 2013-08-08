""" Graphical user interface to create a complete processing flow

This version uses PyQt4 and YAML, and was developed with Python 2.6.5 

When started the GUI is checking for all existing nodes. It lists them and the corresponding parameters.
Comments can also be inserted. They will appear at the beginning of the output file, which has the YAML format.

:Author: Sirko Straube (sirko.straube@dfki.de)
:Created: 2010/06

.. todo::
    There are still some things on the ToDo-list that I did not manage yet:
    
        - improve the help system/documentation (probably with search function)
        - to define "user defined" nodes (this is often used within the framework)
        - display default values for parameters 
        - connect enter key with setParam

    Feel free to edit!
"""

import sys, yaml, copy
from PyQt4 import QtCore, QtGui
import os

file_path = os.path.dirname(os.path.abspath(__file__))
pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
if not pyspace_path in sys.path:
    sys.path.append(pyspace_path)

import pySPACE
#automatically generate all nodes...
import pySPACE.missions.nodes

class ConfDialog(object):
    """ Central class of the gui 
    
    Here all nodes are listed, the GUI is established and all necessary
    functions (within or behind the GUI) are defined. 
    Therefore an instance of this class is a complete, fully functioning GUI.
    
    The object is created in the class ConfiguratorDialog (see below).
    """
    def __init__(self, flow_directory):
        self.flow_directory = flow_directory
        raw_nodelist= pySPACE.missions.nodes.DEFAULT_NODE_MAPPING #create dict of nodes
        self.nodelist=[]
        self.types=['all']
        
        for node in raw_nodelist: #desired name is the name of each entry
            self.nodelist.append(node)
            
        self.nodelist.sort()
        
        for line in self.nodelist: #getting all possible node types for type box
            current_type, valid=self.get_node_type(line)
            
            if not current_type in self.types and valid:
                self.types.append(current_type)
        
        self.types.sort()        
            
    def setupUi(self, Dialog):
        """ This function does all the graphical stuff 
        
        The output is mainly modified from raw code created with QTDesigner.
        Here just the layout of the GUI is defined. 
        The whole arrangement is mainly controlled by horizontal and vertical 
        layouts objects and a grid layout for the whole window.
        """
        #setting up Dialog
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(780, 485)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMouseTracking(False)
        Dialog.setAcceptDrops(False)
        #Dialog.setSizeGripEnabled(True)
        
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Configurator", None, QtGui.QApplication.UnicodeUTF8))

        #left side of dialog containing node selection, corresponding parameters and values
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        
        self.hLayoutNodes = QtGui.QHBoxLayout()
        self.hLayoutNodes.setObjectName("hLayoutNodes")
        self.SelectedNodeTXT = QtGui.QLabel(Dialog)
        self.SelectedNodeTXT.setObjectName("SelectedNodeTXT")
        self.hLayoutNodes.addWidget(self.SelectedNodeTXT)
        self.SelectedNodeTXT.setText(QtGui.QApplication.translate("Dialog", "Selected Node", None, QtGui.QApplication.UnicodeUTF8))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.hLayoutNodes.addItem(spacerItem)
        self.TypeNodeBox = QtGui.QComboBox(Dialog)
        self.TypeNodeBox.setObjectName("TypeNodeBox")
        self.hLayoutNodes.addWidget(self.TypeNodeBox)
        self.verticalLayout.addLayout(self.hLayoutNodes)
        
        self.SelectNodeBox = QtGui.QComboBox(Dialog)
        self.SelectNodeBox.setObjectName("SelectNodeBox")
        self.verticalLayout.addWidget(self.SelectNodeBox)
        
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        
        self.PriorityLabel = QtGui.QLabel(Dialog)
        self.PriorityLabel.setObjectName("PriorityLabel")
        self.PriorityLabel.setText(QtGui.QApplication.translate("Dialog", "Priority", None, QtGui.QApplication.UnicodeUTF8))
        self.horizontalLayout_3.addWidget(self.PriorityLabel)       
        self.PriorityLineEdit = QtGui.QLineEdit(Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PriorityLineEdit.sizePolicy().hasHeightForWidth())
        self.PriorityLineEdit.setSizePolicy(sizePolicy)
        self.PriorityLineEdit.setMaximumSize(QtCore.QSize(30, 22))
        self.PriorityLineEdit.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)
        self.PriorityLineEdit.setObjectName("PriorityLineEdit")
        self.horizontalLayout_3.addWidget(self.PriorityLineEdit)
        self.NodeActive = QtGui.QCheckBox(Dialog)
        self.NodeActive.setObjectName("NodeActive")
        self.NodeActive.setChecked(True)
        self.NodeActive.setText(QtGui.QApplication.translate("Dialog", "is active", None, QtGui.QApplication.UnicodeUTF8))
        self.horizontalLayout_3.addWidget(self.NodeActive)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.HelpButton = QtGui.QPushButton(Dialog)
        self.HelpButton.setObjectName("HelpButton")
        self.HelpButton.setText(QtGui.QApplication.translate("Dialog", " ? ", None, QtGui.QApplication.UnicodeUTF8))
        self.horizontalLayout_3.addWidget(self.HelpButton)
        
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.InsertNodeButton = QtGui.QPushButton(Dialog)
        self.InsertNodeButton.setObjectName("InsertNodeButton")
        self.InsertNodeButton.setText(QtGui.QApplication.translate("Dialog", "Insert Node", None, QtGui.QApplication.UnicodeUTF8))
        self.horizontalLayout_7.addWidget(self.InsertNodeButton)
        self.DeleteNodeButton = QtGui.QPushButton(Dialog)
        self.DeleteNodeButton.setObjectName("DeleteNodeButton")
        self.DeleteNodeButton.setText(QtGui.QApplication.translate("Dialog", "Delete Node", None, QtGui.QApplication.UnicodeUTF8))
        self.horizontalLayout_7.addWidget(self.DeleteNodeButton)
        spacerItem4 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem4)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        
        self.line = QtGui.QFrame(Dialog)
        self.line.setLineWidth(5)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        
        self.AvailParamsTXT = QtGui.QLabel(Dialog)
        self.AvailParamsTXT.setObjectName("AvailParamsTXT")
        self.verticalLayout.addWidget(self.AvailParamsTXT)
        self.AvailParamsTXT.setText(QtGui.QApplication.translate("Dialog", "Available Parameters", None, QtGui.QApplication.UnicodeUTF8))
        
        self.ParamBox = QtGui.QComboBox(Dialog)
        self.ParamBox.setEditable(False)
        self.ParamBox.setObjectName("ParamBox")
        self.verticalLayout.addWidget(self.ParamBox)
        
        self.ParamValueTXT = QtGui.QLabel(Dialog)
        self.ParamValueTXT.setObjectName("ParamValueTXT")
        self.verticalLayout.addWidget(self.ParamValueTXT)
        self.ParamValueTXT.setText(QtGui.QApplication.translate("Dialog", "Value", None, QtGui.QApplication.UnicodeUTF8))
        
        self.ParamValue = QtGui.QLineEdit(Dialog)
        self.ParamValue.setObjectName("ParamValue")
        self.verticalLayout.addWidget(self.ParamValue)
        
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.SetParamButton = QtGui.QPushButton(Dialog)
        self.SetParamButton.setObjectName("SetParamButton")
        self.SetParamButton.setText(QtGui.QApplication.translate("Dialog", "Set", None, QtGui.QApplication.UnicodeUTF8))
        self.horizontalLayout_8.addWidget(self.SetParamButton)
        self.DefaultParamButton = QtGui.QPushButton(Dialog)
        self.DefaultParamButton.setObjectName("DefaultParamButton")
        self.DefaultParamButton.setText(QtGui.QApplication.translate("Dialog", "Default", None, QtGui.QApplication.UnicodeUTF8))
        self.horizontalLayout_8.addWidget(self.DefaultParamButton)
        spacerItem5 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem5)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
             
        self.line2 = QtGui.QFrame(Dialog)
        self.line2.setLineWidth(5)
        self.line2.setFrameShape(QtGui.QFrame.HLine)
        self.line2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line2.setObjectName("line2")
        self.verticalLayout.addWidget(self.line2)
                
        self.Notes = QtGui.QTextEdit(Dialog)
        self.Notes.setObjectName("Notes")
        self.Notes.setEnabled(True)
        self.Notes.setReadOnly(True)        
        self.verticalLayout.addWidget(self.Notes)
        self.Notes.setText("Notes: ")
        #some style changes
        self.Notes.setFrameStyle(QtGui.QFrame.NoFrame)
        notes_palette=self.Notes.palette()
        notes_palette.setColor(notes_palette.currentColorGroup(),notes_palette.Base, Dialog.palette().background().color())
        self.Notes.setPalette(notes_palette)

        spacerItem = QtGui.QSpacerItem(18, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        #right side of dialog containing name of configuration file and parameter list
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        
        self.FileTXT = QtGui.QLabel(Dialog)
        self.FileTXT.setObjectName("FileTXT")
        self.verticalLayout_2.addWidget(self.FileTXT)
        self.FileTXT.setText(QtGui.QApplication.translate("Dialog", "Selected Configuration File", None, QtGui.QApplication.UnicodeUTF8))
        
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.filename="untitled.yaml"
        self.FileEdit = QtGui.QLineEdit(Dialog)
        self.FileEdit.setText(self.filename)
        self.FileEdit.setObjectName("FileEdit")
        self.horizontalLayout_2.addWidget(self.FileEdit)
        self.BrowseButton = QtGui.QPushButton(Dialog)
        self.BrowseButton.setObjectName("BrowseButton")
        self.horizontalLayout_2.addWidget(self.BrowseButton)
        self.BrowseButton.setText(QtGui.QApplication.translate("Dialog", "Browse", None, QtGui.QApplication.UnicodeUTF8))
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        
        self.hLayoutSpecs = QtGui.QHBoxLayout()
        self.hLayoutSpecs.setObjectName("hLayoutSpecs")
        self.SpecsLabel = QtGui.QLabel(Dialog)
        self.SpecsLabel.setObjectName("SpecsLabel")
        self.SpecsLabel.setText(QtGui.QApplication.translate("Dialog", "Specifications", None, QtGui.QApplication.UnicodeUTF8))
        self.hLayoutSpecs.addWidget(self.SpecsLabel)
        spacerItemSpecs = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)  
        self.hLayoutSpecs.addItem(spacerItemSpecs)
        self.EditCommentButton = QtGui.QPushButton(Dialog)
        self.EditCommentButton.setObjectName("EditCommentButton")
        self.EditCommentButton.setText(QtGui.QApplication.translate("Dialog", "Edit Comment", None, QtGui.QApplication.UnicodeUTF8))
        self.hLayoutSpecs.addWidget(self.EditCommentButton)
        self.comment=''
        self.verticalLayout_2.addLayout(self.hLayoutSpecs)        
        self.SpecsBox = QtGui.QTextEdit(Dialog)
        self.SpecsBox.setEnabled(True)
        self.SpecsBox.setReadOnly(True)
        self.SpecsBox.setObjectName("SpecsBox")
        self.verticalLayout_2.addWidget(self.SpecsBox)
        self.specs=[] #no specifications at start
        
        #...and the Save, Append and Close Buttons
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.LoadButton = QtGui.QPushButton(Dialog)
        self.LoadButton.setObjectName("LoadButton")
        self.LoadButton.setText(QtGui.QApplication.translate("Dialog", "Load", None, QtGui.QApplication.UnicodeUTF8))
        self.horizontalLayout.addWidget(self.LoadButton)
        self.SaveButton = QtGui.QPushButton(Dialog)
        self.SaveButton.setObjectName("SaveButton")
        self.SaveButton.setText(QtGui.QApplication.translate("Dialog", "Save", None, QtGui.QApplication.UnicodeUTF8))
        self.horizontalLayout.addWidget(self.SaveButton)
        self.CloseButton = QtGui.QPushButton(Dialog)
        self.CloseButton.setObjectName("CloseButton")
        self.CloseButton.setText(QtGui.QApplication.translate("Dialog", "Close", None, QtGui.QApplication.UnicodeUTF8))
        self.horizontalLayout.addWidget(self.CloseButton)


        #setting Order of Tabulator Key
        Dialog.setTabOrder(self.FileEdit, self.BrowseButton)
        Dialog.setTabOrder(self.BrowseButton, self.SelectNodeBox)
        Dialog.setTabOrder(self.SelectNodeBox, self.ParamBox)
        Dialog.setTabOrder(self.ParamBox, self.ParamValue)
        Dialog.setTabOrder(self.ParamValue, self.SaveButton)
        Dialog.setTabOrder(self.SaveButton, self.InsertNodeButton)
        Dialog.setTabOrder(self.InsertNodeButton, self.CloseButton)

        #Finally, some further layout operations
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        spacerItem1 = QtGui.QSpacerItem(60, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 1, 1, 1)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 2, 1)
        
        #two more dialogues, probably needed
        self.init_CommentUi()
        self.init_HelpUi()

        #formatting Dialog, setting connections, and linking Node and Parameter Box
        self.TypeNodeBox.addItems(self.types)
        self.TypeNodeBox.setCurrentIndex(self.TypeNodeBox.findText('all',QtCore.Qt.MatchExactly))
        
        self.formatNodeBox()
        self.formatParamBox()

        self.resetNoteConnections()
        self.makeConnections()
    
    ##################################################################
    ##The following functions deal with the Comment and Help dialog

    def init_CommentUi(self):
        """setting up layout and connections of the comment dialog"""
        self.Comment_Dial=QtGui.QDialog()
        self.Comment_Dial.setObjectName("Comment")
        self.Comment_Dial.resize(362, 297)
        self.Comment_Dial.gridLayout = QtGui.QGridLayout(self.Comment_Dial)
        self.Comment_Dial.gridLayout.setObjectName("gridLayout")
        self.Comment_Dial.textBrowser = QtGui.QTextEdit(self.Comment_Dial)
        self.Comment_Dial.textBrowser.setObjectName("textBrowser")
        self.Comment_Dial.gridLayout.addWidget(self.Comment_Dial.textBrowser, 0, 0, 1, 1)
        self.Comment_Dial.horizontalLayout = QtGui.QHBoxLayout()
        self.Comment_Dial.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.Comment_Dial.horizontalLayout.addItem(spacerItem)
        self.Comment_Dial.OKButton = QtGui.QPushButton(self.Comment_Dial)
        self.Comment_Dial.OKButton.setObjectName("OKButton")
        self.Comment_Dial.OKButton.setText(QtGui.QApplication.translate("Comment", "OK", None, QtGui.QApplication.UnicodeUTF8))
        self.Comment_Dial.horizontalLayout.addWidget(self.Comment_Dial.OKButton)
        self.Comment_Dial.AbortButton = QtGui.QPushButton(self.Comment_Dial)
        self.Comment_Dial.AbortButton.setObjectName("AbortButton")
        self.Comment_Dial.AbortButton.setText(QtGui.QApplication.translate("Comment", "Abort", None, QtGui.QApplication.UnicodeUTF8))
        self.Comment_Dial.horizontalLayout.addWidget(self.Comment_Dial.AbortButton)
        self.Comment_Dial.gridLayout.addLayout(self.Comment_Dial.horizontalLayout, 1, 0, 1, 1)
        self.Comment_Dial.setWindowTitle(QtGui.QApplication.translate("Comment", "Comment", None, QtGui.QApplication.UnicodeUTF8))
        
        #Connections
        QtCore.QObject.connect(self.Comment_Dial.OKButton, QtCore.SIGNAL("clicked()"), self.update_comment)
        QtCore.QObject.connect(self.Comment_Dial.OKButton, QtCore.SIGNAL("clicked()"), self.Comment_Dial, QtCore.SLOT("close()"))
        QtCore.QObject.connect(self.Comment_Dial.AbortButton, QtCore.SIGNAL("clicked()"), self.Comment_Dial, QtCore.SLOT("close()"))
        
    def show_CommentUi(self):
        """shows and updates comment ui"""
        if self.comment:
            self.Comment_Dial.textBrowser.setText(self.comment)
        self.Comment_Dial.show()
        
    def update_comment(self):
        """writes content of commentUI-window into comment variable
        and modifies appropriately with # symbol (if necessary)"""
        comment=self.Comment_Dial.textBrowser.toPlainText()
        
        if not comment.startsWith('#') and not comment.isspace():
            comment.insert(0,'# ')
        
        pos=1
        
        while comment.indexOf('\n', pos)>=0:
            nextline=comment.indexOf('\n', pos)+1
            if nextline<len(comment) and not comment[nextline] == '#':
                comment.insert(nextline,'#')
            pos=nextline
        
        self.comment=comment
        
    def init_HelpUi(self):
        """setting up layout and connections of the help dialog"""
        self.Help_Dial=QtGui.QDialog()
        self.Help_Dial.setObjectName("Help")
        self.Help_Dial.resize(700, 400)
        self.Help_Dial.gridLayout = QtGui.QGridLayout(self.Help_Dial)
        self.Help_Dial.gridLayout.setObjectName("gridLayout")
        self.Help_Dial.textBrowser = QtGui.QTextEdit(self.Help_Dial)
        self.Help_Dial.textBrowser.setObjectName("textBrowser")
        self.Help_Dial.textBrowser.setEnabled(True)
        self.Help_Dial.textBrowser.setReadOnly(True)
        self.Help_Dial.gridLayout.addWidget(self.Help_Dial.textBrowser, 0, 0, 1, 1)
        self.Help_Dial.horizontalLayout = QtGui.QHBoxLayout()
        self.Help_Dial.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.Help_Dial.horizontalLayout.addItem(spacerItem)
        self.Help_Dial.CloseButton = QtGui.QPushButton(self.Help_Dial)
        self.Help_Dial.CloseButton.setObjectName("CloseButton")
        self.Help_Dial.CloseButton.setText(QtGui.QApplication.translate("Help", "Close", None, QtGui.QApplication.UnicodeUTF8))
        self.Help_Dial.horizontalLayout.addWidget(self.Help_Dial.CloseButton)
        self.Help_Dial.gridLayout.addLayout(self.Help_Dial.horizontalLayout, 1, 0, 1, 1)
        self.Help_Dial.setWindowTitle(QtGui.QApplication.translate("Help", "Help", None, QtGui.QApplication.UnicodeUTF8))        

        #Connections
        QtCore.QObject.connect(self.Help_Dial.CloseButton, QtCore.SIGNAL("clicked()"), self.Help_Dial, QtCore.SLOT("close()"))
        
    def show_HelpUi(self):
        """shows and updates help ui"""
        self.update_help()
        self.Help_Dial.show()
        
    def update_help(self):
        """updates text in help ui according to current node"""
        currentnode=str(self.SelectNodeBox.currentText())
        if currentnode:
            self.Help_Dial.textBrowser.setText(
                pySPACE.missions.nodes.NODE_MAPPING[currentnode].__doc__)
    
    ##################################################################
    ############## internal formatting and helper functions
    def get_node_type(self, node_str):
        """ Parse node string for the node type
        
        This function assigns the node string (2nd argument) to a type which is used
        for the type sorting in the GUI. The function takes the whole NODE_MAPPING string: The type is
        the string between "pySPACE.missions.nodes." and the next "." and corresponds to the directory in
        pySPACE/missions/nodes/
        
        :returns: node type string and a boolean reflecting the success of the assignment
        """
        current_node_str = str(pySPACE.missions.nodes.NODE_MAPPING[node_str])
        ref_str = 'pySPACE.missions.nodes.'
        startpos = current_node_str.find(ref_str)+len(ref_str)
        endpos = current_node_str.find('.',startpos)
        current_type = current_node_str[startpos:endpos]
 
        if 0 <= startpos < endpos:
            valid = True
        else:
            valid = False
        # the string of the current type is returned and a boolean
        # if the assignment worked
        return current_type, valid
    
    def insertImage(self, icon, filename):
        """insert image into label"""
        pixmap = QtGui.QPixmap()
        pixmap.load(filename)
        icon.setPixmap(pixmap.scaledToWidth(icon.width(),
                                            QtCore.Qt.SmoothTransformation))
    
    def formatNodeBox(self):
        """Format NodeBox according to type selection in TypeNodeBox"""
        selected_type=str(self.TypeNodeBox.currentText())
        self.SelectNodeBox.clear()
        
        if selected_type == 'all':
            self.SelectNodeBox.addItems(self.nodelist)
        else:
            temp_nodelist=[]
            
            for line in self.nodelist:
                current_type, valid=self.get_node_type(line)
                if current_type == selected_type:
                    temp_nodelist.append(line)
                    
            temp_nodelist.sort()
            self.SelectNodeBox.addItems(temp_nodelist)
       
    def formatParamBox(self):
        """Format ParamBox according to node selection in NodeBox. The parameter-list is derived 
        from the parameters of the __init__ function of the current node. The parameters "self" and 
        "args" are omitted, "kwargs" is translated into "user defined"  """
        currentnode=str(self.SelectNodeBox.currentText())
        
        if currentnode:
            self.ParamBox.clear()
            parameters=list(pySPACE.missions.nodes.NODE_MAPPING[currentnode].__init__.im_func.func_code.co_varnames)
        
            if 'args' in parameters:
                parameters.pop(parameters.index('args'))
        
            if 'kwargs' in parameters:
                parameters.pop(parameters.index('kwargs'))
                parameters.append('user defined')
        
            if 'self' in parameters:
                parameters.pop(parameters.index('self'))
        
            parameters.sort()
            self.ParamBox.addItems(parameters)
        
    def validateNodesandParams(self, specs):
        """This function checks the current specification for possible unspecified nodes. This is necessary when
        e.g. a file is loaded. The file is ignored, if a non-existing node is identified. When a parameter is
        not present a warning is printed in the notes section, since user defined parameters are allowed.
        """
        for specsnode in specs:
                    
            if specsnode['node'] in self.nodelist: #checking parameters
                if 'parameters' in specsnode:
                    parameters=list(
                        pySPACE.missions.nodes.NODE_MAPPING[specsnode['node']].__init__.im_func.func_code.co_varnames)
                    for param in specsnode['parameters']:
                        if not param in parameters:
                            self.note('Warning: File contains unspecified node parameter:Node <%s> has no default parameter <%s>!' % (specsnode['node'], param))
                            return True
            else:
                self.note('File contains unspecified node: <%s> not existent! File ignored.' % specsnode['node'])
                return False
                
        return True
        
    def eval_user_defined(self):
        """Evaluate a user defined parameter, so that it can be managed properly. This function
        is executed whenever a user defined parameter is set. It decomposes the entry into 
        <paramname>:<paramval> and returns both. If the format is wrong, nothing happens and
        a message is printed in the notes section.
        """
        userdef=str(self.ParamValue.text())
        separator=userdef.find(':')
        
        if separator>0:
            current_param=userdef[0:separator]
            current_value=userdef[separator+1:]
        else:
            current_param=''
            current_value=''
            self.note("Wrong input format: Use \'param:value\'!")
        
        return current_param, current_value
       
    def get_param_value(self):
        """Get the value of the current parameter (if given) in order to display it
        in the corresponding TextEdit (ParamValue). If a user defined parameter is
        selected, the function looks for undefined parameters and displays <name>:<value>
        instead of <value>.
        """
        
        if not self.PriorityLineEdit.text().isEmpty() and self.specs: #only if priority is specified
            currentnode=str(self.SelectNodeBox.currentText())
            selected_param=str(self.ParamBox.currentText())
            current_value=''
            current_pos=int(self.PriorityLineEdit.text())-1 #position=priority-1
             
            if selected_param == 'user defined':
                paramlist=[]
                for current_param in range(self.ParamBox.count()): #create a list of the parameters currently available
                    paramlist.append(str(self.ParamBox.itemText(current_param)))
            
                avail_params=self.specs[current_pos]['parameters'].keys() #take all parameters specified for current node
            
                for line in avail_params:
                    if line not in paramlist:
                        selected_param=line #found first user defined parameter, when specified parameter is not in paramlist
                        current_value=line + ':' #value now contains "<name>:" - the value is added later
                        break
        
            if current_pos<=len(self.specs) \
            and currentnode in self.specs[current_pos]['node'] \
            and 'parameters' in self.specs[current_pos] \
            and selected_param in self.specs[current_pos]['parameters']: #node and parameter exist
                current_value=current_value + str(self.specs[current_pos]['parameters'][selected_param]) #add current value
                self.ParamValue.setText(current_value)
            else: #parameter not specified => empty field
                self.ParamValue.setText('')
        else:#parameter not specified => empty field
            self.ParamValue.setText('')
                   
        
    def find_inactive_nodes(self, rawdata):
        """ This function takes raw YAML data (i.e. text not loaded with YAML function)
        and looks for commented nodes. The index of all of these nodes is returned, so 
        that these nodes later get the value False for node_is_active. 
        """
        pos=0
        nnodes=rawdata.count('node: ') #nodes are detected with string matching

        inactive_nodes=[]

        for nodenr in range(nnodes):
            nextnode=rawdata.find('node: ', pos) #next node position is also detected with string matching

            if nextnode and '#' in rawdata[rawdata.rfind('\n',0,nextnode):nextnode]: #if node is a comment
                inactive_nodes.append(nodenr)

            pos=nextnode+1

        return inactive_nodes
        
    def deactivate_nodes(self):
        """Function that (i) uses yaml.dump to format dict according to YAML
        specifications and (ii) deactivates nodes by adding the comment symbol "#".
        Here, a deepcopy is used, because the specs dict is changed (i.e. the entry
        node_is_active is deleted).
        The function checks and dumps one node after the other: Due to a difference 
        in formatting (by YAML) when dumping one in contrast to more than one node,
        additional spaces are also added here.
        Return value is the complete specification text in YAML format.
        """
        localspecs=copy.deepcopy(self.specs)
        
        rawdata=""
        
        inactive_node=[]
        
        for line in localspecs:
            node_is_active=line['node_is_active'] #store boolean value
            
            del line['node_is_active'] #delete entry
            
            tempdata=yaml.dump(line) #data of current node
            tempdata='- ' + tempdata #change YAML single format to format for multiple entries
            tempdata=tempdata.replace('\n','\n  ') #change YAML single format to format for multiple entries
            if tempdata[-2:] == '  ':
                tempdata=tempdata[0:-2] #remove possible spaces
            
            if not node_is_active: #add comment symbol, if necessary
                tempdata='#' + tempdata
                tempdata=tempdata.replace('\n','\n#')
                
                if tempdata.endswith('#'):
                    tempdata=tempdata[0:-1]
            
            rawdata=rawdata+tempdata #add current node
            
        return rawdata
        
    def showSpecs(self):
        """show the complete specification text in YAML format"""
        self.SpecsBox.setText(self.deactivate_nodes())
        
    def note(self, notestring):
        """add notestring to current notes"""
        self.Notes.setText(self.Notes.toPlainText()+notestring+"\n") 
        
    def import_comment(self, filename):
        """this function loads the given filename and stores all lines starting with '#'
        in the comment variable. These lines have to be at the beginning of the file and 
        must not start with a node specification (in YAML format)."""
        open_file=open(filename)
        lines=open_file.readlines()
        open_file.close()
        
        self.comment=''
        
        startyaml=False
        
        for line in lines:
            if line.startswith('#'):
                if line.startswith('#- node: ') or line.startswith('#- {node: '):
                    break
                else:
                    self.comment=self.comment + line
            else:
                break
                
    ############## all connections in GUI        

    def resetNoteConnections(self):
        """Each time something happens in the GUI, the notes are deleted. 
        The current function establishes these connections"""
        QtCore.QObject.connect(self.SelectNodeBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.resetNotes)
        QtCore.QObject.connect(self.PriorityLineEdit, QtCore.SIGNAL("textEdited(const QString&)"), self.resetNotes)
        QtCore.QObject.connect(self.NodeActive, QtCore.SIGNAL("clicked()"), self.resetNotes)
        QtCore.QObject.connect(self.InsertNodeButton, QtCore.SIGNAL("clicked()"), self.resetNotes)
        QtCore.QObject.connect(self.DeleteNodeButton, QtCore.SIGNAL("clicked()"), self.resetNotes)
        QtCore.QObject.connect(self.ParamBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.resetNotes)
        QtCore.QObject.connect(self.ParamValue, QtCore.SIGNAL("textEdited(const QString&)"), self.resetNotes)
        QtCore.QObject.connect(self.SetParamButton, QtCore.SIGNAL("clicked()"), self.resetNotes)
        QtCore.QObject.connect(self.DefaultParamButton, QtCore.SIGNAL("clicked()"), self.resetNotes)
        QtCore.QObject.connect(self.BrowseButton, QtCore.SIGNAL("clicked()"), self.resetNotes)
        QtCore.QObject.connect(self.FileEdit, QtCore.SIGNAL("textEdited(const QString&)"), self.resetNotes)
        QtCore.QObject.connect(self.LoadButton, QtCore.SIGNAL("clicked()"), self.resetNotes)
    
    def makeConnections(self):
        """All other connections that are necessary to make the GUI work."""        
        QtCore.QObject.connect(self.TypeNodeBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.formatNodeBox)
        QtCore.QObject.connect(self.SelectNodeBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.formatParamBox)
        QtCore.QObject.connect(self.SelectNodeBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.getPriority)
        QtCore.QObject.connect(self.SelectNodeBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.getState)
        QtCore.QObject.connect(self.SelectNodeBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.get_param_value)
        QtCore.QObject.connect(self.SelectNodeBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.update_help)
        QtCore.QObject.connect(self.NodeActive, QtCore.SIGNAL("clicked()"), self.setState)
        QtCore.QObject.connect(self.NodeActive, QtCore.SIGNAL("clicked()"), self.showSpecs)
        QtCore.QObject.connect(self.PriorityLineEdit, QtCore.SIGNAL("textEdited(const QString&)"), self.getState)
        QtCore.QObject.connect(self.PriorityLineEdit, QtCore.SIGNAL("textEdited(const QString&)"), self.get_param_value)
        QtCore.QObject.connect(self.HelpButton, QtCore.SIGNAL("clicked()"), self.show_HelpUi)
        
        QtCore.QObject.connect(self.InsertNodeButton, QtCore.SIGNAL("clicked()"), self.insertNode)
        QtCore.QObject.connect(self.DeleteNodeButton, QtCore.SIGNAL("clicked()"), self.delNode)
        
        QtCore.QObject.connect(self.ParamBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.note_user_defined)
        QtCore.QObject.connect(self.ParamBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.get_param_value)
        QtCore.QObject.connect(self.SetParamButton, QtCore.SIGNAL("clicked()"), self.setParam)
        QtCore.QObject.connect(self.DefaultParamButton, QtCore.SIGNAL("clicked()"), self.defaultParam)
        
        QtCore.QObject.connect(self.BrowseButton, QtCore.SIGNAL("clicked()"), self.selectFile)
        QtCore.QObject.connect(self.FileEdit, QtCore.SIGNAL("editingFinished()"), self.updateFileName)
        QtCore.QObject.connect(self.FileEdit, QtCore.SIGNAL("textEdited (const QString&)"), self.updateFileName)
        QtCore.QObject.connect(self.EditCommentButton, QtCore.SIGNAL("clicked()"), self.show_CommentUi)
        
        QtCore.QObject.connect(self.LoadButton, QtCore.SIGNAL("clicked()"), self.loadYAMLFile)
        QtCore.QObject.connect(self.SaveButton, QtCore.SIGNAL("clicked()"), self.saveYAMLFile)
        
        QtCore.QObject.connect(self.CloseButton, QtCore.SIGNAL("clicked()"), self.Comment_Dial, QtCore.SLOT("close()"))
        QtCore.QObject.connect(self.CloseButton, QtCore.SIGNAL("clicked()"), self.Help_Dial, QtCore.SLOT("close()"))
        
    ############## functions mainly executed by user using GUI
    def selectFile(self):
        """Opens the SelectFile Dialog and inserts filename in FileEdit field. If no file is selected,
        the filename is set to untitled.yaml"""
        self.filename = \
            QtGui.QFileDialog.getSaveFileName(None, 'Select File', 
                                              self.flow_directory, 
                                              'YAML files (*.yaml);;All files (*)')
        if self.filename.isEmpty():
            self.filename = QtCore.QString("untitled.yaml")
        self.FileEdit.setText(self.filename)
    
    def loadYAMLFile(self):
        """load YAML file specified by user:
        a dialog opens asking the user to specify a file
        then the file is decomposed into comment header, active and 
        inactive nodes
        the results are written into the comment and the specs variable
        """
        open_fname = QtGui.QFileDialog.getOpenFileName(None, 'Open File', '.', 'YAML files (*.yaml);;All files (*)')
    
        if not open_fname.isEmpty():
            self.import_comment(open_fname)        
            yamlfile=open(open_fname)
            
            raw_yaml=yamlfile.read()
            yamlfile.close()
            
            if self.comment:
                raw_yaml=raw_yaml[len(self.comment)-1:] #cut comment
            
            inactive_nodes=self.find_inactive_nodes(raw_yaml)
            
            #raw_yaml=self.del_comment_between(raw_yaml)
            
            specs=yaml.load(raw_yaml.replace('#', '')) 
            
            for nodenr in range(len(specs)):
                if nodenr in inactive_nodes:
                    specs[nodenr]['node_is_active']=False
                else:
                    specs[nodenr]['node_is_active']=True

            if self.validateNodesandParams(specs):
                self.filename=open_fname
                self.FileEdit.setText(self.filename)
                self.specs=specs
                self.showSpecs()
                
    def saveYAMLFile(self):
        """saves specs to specified file:
        if there is a comment, it is written in the header
        all inactive nodes are marked correspondingly"""
        if self.specs:
            savefile=open(self.filename, 'w')
            savefile.write(self.comment)
            savefile.write('\n')
            savefile.write(self.deactivate_nodes())
            savefile.close()
    
    def updateFileName(self):
        """new filename is stored internally"""
        self.filename = self.FileEdit.text()

    def setPriority(self, priority):
        """the given priority is displayed in the corresponding line edit"""
        self.PriorityLineEdit.setText(str(priority))
          
    def getPriority(self):
        """determine the priority of the selected node.
        if the node is specified multiple times, the priority 
        of the first node found is taken
        respective notes are displayed"""
        currentnode=str(self.SelectNodeBox.currentText())
        pos=1
        nodefound=False
        
        for entry in self.specs:
            if currentnode in entry['node']:
                nodefound=True
                break
            pos+=1
        
        if nodefound:
            self.note("If you use the selected node multiple times, edit the desired node by entering the appropriate priority.")
            self.PriorityLineEdit.setText(str(pos))
        else:
            self.note("Selected Node is new. Select desired priority and press >>Insert Node<<.")
            self.PriorityLineEdit.setText("")
            
    def getState(self):
        """determines if node is specified as active and sets check accordingly in GUI"""
        if self.specs:
            priority=self.PriorityLineEdit.text()
            if not priority.isEmpty() and int(priority)>0 and int(priority)-1<len(self.specs):
                self.NodeActive.setChecked(self.specs[int(priority)-1]['node_is_active'])
                
    def setState(self):
        """set state in specs (internally)"""
        checked=False
        if self.specs:
            priority=self.PriorityLineEdit.text()
            if not priority.isEmpty() and int(priority)>0 and int(priority)-1<len(self.specs):
                if self.NodeActive.checkState():
                    checked=True
                self.specs[int(priority)-1]['node_is_active']=checked
            else:
                raise(ValueError,'Warning: node_chain_GUI::setState: Function called without valid priority!') #should never happen

    def insertNode(self):
        """inserts a node in specifications according to where the user specified it
        function is executed when user presses <insert> button
        when priority is not specified (or wrong), the node is appended at the end"""
        currentnode=str(self.SelectNodeBox.currentText())
        
        if self.specs:
            priority=self.PriorityLineEdit.text()
            if not priority.isEmpty() and int(priority)>0:
                self.specs.insert(int(priority)-1, dict(node=currentnode))
            else:
                self.specs.append(dict(node=currentnode))
                self.setPriority(len(self.specs))
                self.note("No or wrong priority given! Node appended at the end.")
        else:
            self.specs.append(dict(node=currentnode))
            self.setPriority(len(self.specs))
        
        self.note_user_defined()
        self.setState()    
        self.showSpecs()
        
    def delNode(self):
        """delete node from current specifications.
        delete is ignored if priority is not specified correct"""    
        currentnode=str(self.SelectNodeBox.currentText())
        
        if self.specs:
            
            if self.PriorityLineEdit.text():
                current_priority=int(self.PriorityLineEdit.text())
    
                if currentnode in self.specs[current_priority-1]['node']:
                    del self.specs[current_priority-1]
                else:
                    self.note("No node at given priority! Delete ignored.")                
            
            else:
                self.note("No priority given! Delete ignored.")
        else:
            self.note("No specifications given! Delete ignored.")
        
        self.note_user_defined()
        self.getPriority()
        self.showSpecs()
        
    def setParam(self):
        """ insert parameter into current specifications
        
        this is only happening, if user specifies node and priority correctly
        if parameter is existing, the value is only changed, if not, a new entry
        is established in specs
        the user defined parameter case is also considered, given it is entered
        in the expected way: 'param:value'
        """
        currentnode=str(self.SelectNodeBox.currentText())
        
        if not self.PriorityLineEdit.text().isEmpty(): #priority is specified
            current_pos=int(self.PriorityLineEdit.text())-1 #position=priority-1
        
            if currentnode in self.specs[current_pos]['node']: #node exists
                
                if not self.ParamValue.text().isEmpty(): #parameter value is specified
                                    
                    if str(self.ParamBox.currentText()) == 'user defined':
                        selected_param, current_value=self.eval_user_defined()
                    else:
                        selected_param=str(self.ParamBox.currentText())
                        current_value=str(self.ParamValue.text())
                    
                    #todo: check this
                    if isinstance(current_value, basestring) and current_value.startswith("eval("):
                        current_value = eval(current_value[5:-1])
                        
                    if selected_param: #not the case if user specified 'user defined' parameter wrong
                
                        if not 'parameters' in self.specs[current_pos]: #node has no parameters so far
                            templist=[]
                            templist.append(selected_param)
                            self.specs[current_pos].update(dict(parameters=(dict.fromkeys(templist, current_value))))
                        else:
                            current_params=self.specs[current_pos]['parameters']
                        
                            if selected_param in current_params: #parameter exists
                                current_params[selected_param]=current_value
                            else: #insert new parameter
                                templist=[]
                                templist.append(selected_param)
                                current_params.update(dict.fromkeys(templist, current_value))
                else:
                    self.note("No parameter value entered! Set parameter ignored.")                
            else:
                self.note("No node at given priority! Please change priority appropriately.")                
        else:
            self.note("No priority given! Please change priority appropriately.")
        
        self.showSpecs()
        
    def defaultParam(self):
        """the default value is not set here. instead, the parameter is deleted from the
        specifications, so that the default values are used.
        .. note:: this shortcoming should be improved in future versions
        """
        currentnode=str(self.SelectNodeBox.currentText())

        if not self.PriorityLineEdit.text().isEmpty(): #priority is specified
            current_pos=int(self.PriorityLineEdit.text())-1 #position=priority-1

            if currentnode in self.specs[current_pos]['node']: #node exists
                if 'parameters' in self.specs[current_pos]:
                    
                    if str(self.ParamBox.currentText()) == 'user defined':
                        selected_param=str(self.ParamValue.text())
                    else:
                        selected_param=str(self.ParamBox.currentText())
                    
                    current_params=self.specs[current_pos]['parameters']
                    
                    if selected_param in current_params:
                        del current_params[selected_param]
                    else:
                        self.note("No parameter <%s> found." % selected_param)
                        
                    if len(current_params)==0:
                        del self.specs[current_pos]['parameters']
                        
            else:
                self.note("No node at given priority! Please change priority appropriately.")                
        else:
            self.note("No priority given! Please change priority appropriately.")

        self.showSpecs()
        
    def note_user_defined(self):
        """display a special note only for user defined parameters"""
        if str(self.ParamBox.currentText()) == 'user defined':
            self.note("You have selected to define an additional parameter. To set parameter please enter name and value, separated by \':\' (e.g. myparam:5). To set default value enter only parameter name.")
        
    def resetNotes(self):
        """reset text in Notes"""
        self.Notes.setText("Notes: ")

class NodeChainConfigurationWidget(QtGui.QWidget):
    """class which sets up GUI"""
    def __init__(self, flow_directory='.', parent=None):
        super(NodeChainConfigurationWidget, self).__init__()
        self.parent = parent
        self.confDialog = ConfDialog(flow_directory)
        self.confDialog.setupUi(self)
        
        self.connect(self.confDialog.CloseButton, QtCore.SIGNAL("clicked()"), 
                     self.close)
        
    def close(self):
        """ Close all gui components and gui """
        self.parent.close()
        super(NodeChainConfigurationWidget, self).close()
        

if __name__ == "__main__":
    """main: start GUI and close program, when all windows are closed"""
    app = QtGui.QApplication(sys.argv)
    configD = NodeChainConfigurationWidget()
    configD.show()
    configD.confDialog.DeleteNodeButton.setDefault(False) #without this, Button is automatically set to Default (for whatever reason)
    configD.confDialog.HelpButton.setDefault(False) #without this, Button is automatically set to Default (for whatever reason)
    
    app.connect(app, QtCore.SIGNAL("lastWindowClosed()"), app, QtCore.SLOT("quit()"))
    sys.exit(app.exec_())
