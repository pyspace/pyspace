# -*- coding: UTF-8 -*
""" Load the global configuration file

:Author: Timo Duchrow (timo.duchrow@dfki.de)
:Created: 2009/02/20
"""
from copy import deepcopy

import os
import sys
import logging
import yaml
import warnings
from os.path import expanduser
home = expanduser("~")

class Configuration(object):
    """ A global configuration class """
    def __init__(self, parent_dir, conf_dir=None):
        """ Setting of the default paths

        The files are searched in 'PYSPACE_CONF_DIR', if it is specified.
        Otherwise it is searched for in the pySPACEcenter in the user directory.

        **Parameters**
        
            :parent_dir:  the directory the pySPACE/ tree resides in
            :conf_dir: an alternative configuration directory
            
                       (*optional, default: 'home_dir/pySPACEcenter'*)
        """
        # Set parent_dir (and root for convenience)
        self.parent_dir = parent_dir
        self.root = os.path.join(parent_dir, "pySPACE")
        self.root_dir = os.path.join(parent_dir, "pySPACE")
        self.examples_conf_dir = os.path.join(self.parent_dir,"docs","examples", "conf")
        
        # Set configuration directory or default
        if conf_dir is None:
            conf_dir = os.path.join(home, "pySPACEcenter")
        self.conf_dir = conf_dir

        # Set other defaults
        self.examples_storage = os.path.join(self.parent_dir, "docs","examples", "storage")
        self.storage = os.path.join(home, "pySPACEcenter","storage")
        self.examples_spec_dir = os.path.join(self.parent_dir, "docs","examples", "specs")
        self.spec_dir = os.path.join(home, "pySPACEcenter","specs")
        self.log_dir = os.path.join(self.root_dir, "log")
        self.min_log_level = 0
        self.backend_com = None
        self.external_nodes = []


    first_call_epilog =\
    """
    Welcome to pySPACE
    ------------------

    This seems to be the first time, you are using pySPACE.

    For detailed documentation on pySPACE refer to the online documentation at
    http://pyspace.github.io/pyspace/index.html,
    the __init__ file in the pySPACE folder, or the index.rst in the docs folder.

    If you have already used the setup.py or another installation program all
    relevant configuration files should be found in the folder `pySPACEcenter`
    in your home directory.
    Otherwise it will be searched for in your `PYSPACE_CONF_DIR`.

    The main configuration is specified in the <config.yaml>. Please have a look
    at it and the therein specified environment parameters.
    """

    def load_configuration(self, conf_file_name=None):
        """ Load a configuration from the specified :ref:`YAML<yaml>` configuration file.
        
        Overwrites default directories and path with directories set
        in the specified YAML file.
        
        **Parameters**
        
            :conf_file_name: the name of the conf file that lies in the 'conf_dir'
        """
        if conf_file_name is None:
            conf_file_name = 'config.yaml'
            logging.debug("No configuration file given in %s. Defaulting to %s." %
                         (self.conf_dir,conf_file_name))

        conf_file_name = os.sep.join([self.conf_dir, conf_file_name])
        self.conf_file_name = conf_file_name    # store for later reference
        logging.debug("Configuration file: %s"%conf_file_name)
        print("--> Using configuration file: \n\t\t %s."%conf_file_name)
        
        self.python_path = None
        try:
            conf_file = open(conf_file_name, 'r')
            conf = yaml.load(conf_file)
            for k, v in conf.iteritems():
                if v is not None or (isinstance(v, str) and v.isspace()):
                    if isinstance(v,str) and '~' in v:
                        v = os.path.expanduser(v)
                    self.__dict__[k] = v
            conf_file.close()
            # check for first call
            if "first_call" in self.__dict__.keys() and self.first_call:
                conf_file = open(conf_file_name, 'r')
                lines = conf_file.readlines()
                conf_file.close()
                for line in lines:
                    if line.startswith("first_call : True"):
                        lines.remove(line)
                        print self.first_call_epilog
                        conf_file = open(conf_file_name, 'w')
                        conf_file.writelines(lines)
                        conf_file.close()
                        break
        except IOError, _:
            msg = "Unable to open configuration file " + str(conf_file_name)
            msg += " Put it to the pySPACEcenter folder or change "
            msg += "the 'PYSPACE_CONF_DIR' parameter in your shell."
            warnings.warn(msg)
        except AttributeError, _:
            warnings.warn("Your configuration file  is empty. Please check the installation documentation! Using defaults.")

        print("--> Using Python: \n\t\t %s"%sys.executable)
        # Setup python path as specified
        if self.python_path is not None:
            for i in range(len(self.python_path)):
                if '~' in self.python_path[i]:
                    self.python_path[i]=os.path.expanduser(self.python_path[i])
            new_python_path = deepcopy(self.python_path)
            python_path_set = set(new_python_path)
            unique_sys_path = [i for i in sys.path if i not in python_path_set and not python_path_set.add(i)]
            new_python_path.extend(unique_sys_path)
            sys.path = new_python_path
        else:
            self.python_path=[]
        print("--> Prepending to your PYTHONPATH: \n\t\t %s"%str(self.python_path))

        if len(self.external_nodes) > 0:
            print("--> Using external nodes: \n\t\t %s" %
                  str(self.external_nodes))

        # Append root directory to PYTHONPATH anew
        if not self.parent_dir in sys.path:
            sys.path.append(self.parent_dir)

        if "resources_dir" in self.__dict__.keys():
            self.__dict__["storage"] = self.__dict__["resources_dir"]
            warnings.warn("Change config parameter 'resources_dir' to 'storage'.")
        return self
        
    def __str__(self):
        return yaml.dump(self.__dict__)
        
  
