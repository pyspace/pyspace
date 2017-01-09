# -*- coding: UTF-8 -*
""" Load the global configuration file

:Author: Timo Duchrow (timo.duchrow@dfki.de)
:Created: 2009/02/20
"""

import logging
import os
import sys
import warnings
from os.path import expanduser

import yaml

HOME = expanduser("~")

LOGGER = logging.getLogger("pySPACE")
LOGGER.setLevel(logging.INFO)

HANDLER = logging.StreamHandler(stream=sys.stdout)
HANDLER.setFormatter(logging.Formatter(fmt="%(message)s"))
HANDLER.setLevel(logging.INFO)
LOGGER.addHandler(HANDLER)


class Configuration(dict):
    """ A global configuration class """

    FIRST_CALL_EPILOG = """
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

    def __init__(self, parent_dir, conf_dir=None):
        """ Setting of the default paths

        The files are searched in 'PYSPACE_CONF_DIR', if it is specified.
        Otherwise it is searched for in the pySPACEcenter in the user directory.

        **Parameters**
        
            :parent_dir:  the directory the pySPACE/ tree resides in
            :conf_dir: an alternative configuration directory
            
                       (*optional, default: 'home_dir/pySPACEcenter'*)
        """
        super(Configuration, self).__init__(parent_dir=parent_dir,
                                            root=os.path.join(parent_dir, "pySPACE"),
                                            root_dir=os.path.join(parent_dir, "pySPACE"),
                                            examples_conf_dir=os.path.join(parent_dir, "docs", "examples", "conf"),
                                            examples_storage=os.path.join(parent_dir, "docs", "examples", "storage"),
                                            storage=os.path.join(HOME, "pySPACEcenter", "storage"),
                                            examples_spec_dir=os.path.join(parent_dir, "docs", "examples", "specs"),
                                            spec_dir=os.path.join(HOME, "pySPACEcenter", "specs"),
                                            log_dir=os.path.join(parent_dir, "pySPACE", "log"),
                                            min_log_level=0,
                                            backend_com=None,
                                            external_nodes=[],
                                            blacklisted_nodes=[])

        # Set configuration directory or default
        if conf_dir is None:
            conf_dir = os.path.join(HOME, "pySPACEcenter")
        self["conf_dir"] = conf_dir

    def load_configuration(self, conf_file_name=None):
        """ Load a configuration from the specified :ref:`YAML<yaml>` configuration file.
        
        Overwrites default directories and path with directories set
        in the specified YAML file.
        
        :param conf_file_name: the name of the conf file that lies in the 'conf_dir'
        :type conf_file_name: basestring
        """
        if conf_file_name is None:
            conf_file_name = 'config.yaml'
            LOGGER.debug("No configuration file given in %s. Defaulting to %s." % (
                self["conf_dir"], conf_file_name))

        conf_file_name = os.sep.join([self["conf_dir"], conf_file_name])
        self["conf_file_name"] = conf_file_name    # store for later reference

        LOGGER.debug("Configuration file: %s" % conf_file_name)
        LOGGER.info("--> Using configuration file: \n\t\t %s." % conf_file_name)

        self["python_path"] = []
        try:
            with open(conf_file_name, 'r') as conf_file:
                conf = yaml.load(conf_file)
                for k, v in conf.iteritems():
                    if v is not None or (isinstance(v, basestring) and v.isspace()):
                        if isinstance(v, basestring) and '~' in v:
                            v = os.path.expanduser(v)
                        self[k] = v

            # check for first call
            if "first_call" in self and self["first_call"]:
                with open(conf_file_name, 'r') as conf_file:
                    lines = conf_file.readlines()

                for line in lines:
                    if line.startswith("first_call : True"):
                        lines.remove(line)
                        LOGGER.info(Configuration.FIRST_CALL_EPILOG)
                        with open(conf_file_name, 'w') as conf_file:
                            conf_file.writelines(lines)
                        break
        except IOError:
            msg = "Unable to open configuration file {conf_file_name}"\
                  " Put it to the pySPACEcenter folder or change "\
                  "the 'PYSPACE_CONF_DIR' parameter in your shell.".format(conf_file_name=conf_file_name)
            warnings.warn(msg)
        except AttributeError, _:
            warnings.warn("Your configuration file  is empty. "
                          "Please check the installation documentation! Using defaults.")

        LOGGER.info("--> Using Python: \n\t\t %s" % sys.executable)

        # Setup python path as specified
        python_path = set()
        for path in self["python_path"]:
            python_path.add(os.path.expanduser(path))

        for path in sys.path:
            python_path.add(path)
        sys.path = list(python_path)

        LOGGER.info("--> Prepending to your PYTHONPATH: \n\t\t %s" % self["python_path"])

        if self["external_nodes"]:
            LOGGER.info("--> Using external nodes: \n\t\t %s" % self["external_nodes"])

        # Append root directory to PYTHONPATH anew
        if self["parent_dir"] not in sys.path:
            sys.path.append(self["parent_dir"])

        if "resources_dir" in self:
            self["storage"] = self["resources_dir"]
            warnings.warn("Change config parameter 'resources_dir' to 'storage'.")
        return self

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __str__(self):
        return yaml.dump(self)
