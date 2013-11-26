""" Create pySPACE center (basic software configuration and data folder)

This file must be run with python.

.. note:: In future this script shall create not only the basic configuration
          folder, but also move the relevant code to site-packages
          and install the required or even the optional dependencies.

.. todo:: Make this script a real installation script.
"""

from distutils.core import setup
import os
import sys
import shutil
import warnings
from os.path import expanduser
home = expanduser("~")

email = 'ric-pyspace-dev@dfki.de'

classifiers = ["Development Status :: 1 - Production/Stable",
               "Intended Audience :: Developers",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "License :: None",
               "Operating System :: Linux",
               "Operating System :: OS X",
               "Programming Language :: Python",
               "Programming Language :: Python :: 2",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Benchmark",
               "Topic :: Scientific/Engineering :: Mathematics"]

def get_module_code():
    # keep old python compatibility, so no context managers
    init_file = open(os.path.join(os.getcwd(), 'pySPACE', '__init__.py'))
    module_code = init_file.read()
    init_file.close()
    return module_code

def throw_bug():
    raise ValueError('Can not get pySPACE parameters!\n'
                     'Please report a bug to ' + email)

try:
    import ast

    def get_extract_variable(tree, variable):
        for node in ast.walk(tree):
            if type(node) is ast.Assign:
                try:
                    if node.targets[0].id == variable:
                        print node.targets
                        return node.value.s
                except Exception,e:
                    raise
                    pass
        throw_bug()

    def get_ast_tree():
        return ast.parse(get_module_code())

    def get_version():
        tree = get_ast_tree()
        return get_extract_variable(tree, '__version__')

    def get_short_description():
        return "pySPACE is a **Signal Processing And Classification Environment** "
        tree = get_ast_tree()
        return get_extract_variable(tree, '__short_description__')

    def get_long_description():
        tree = get_ast_tree()
        return ast.get_docstring(tree)
except ImportError:
    import re

    def get_variable(pattern):
        m = re.search(pattern, get_module_code(), re.M + re.S + re.X)
        if not m:
            throw_bug()
        return m.group(1)

    def get_version():
        return get_variable(r'^__version__\s*=\s*[\'"](.+?)[\'"]')

    def get_short_description():
        text = get_variable(r'''^__short_description__\s*=\s*  # variable name and =
                            \\?\s*(?:"""|\'\'\')\\?\s*         # opening quote with backslash
                            (.+?)
                            \s*(?:"""|\'\'\')''')              # closing quote
        return text.replace(' \\\n', ' ')

    def get_long_description():
        return get_variable(r'''^(?:"""|\'\'\')\\?\s*          # opening quote with backslash
                            (.+?)
                            \s*(?:"""|\'\'\')''')              # closing quote

def create_directory(path):
    """ Create the given directory path recursively

    Copy from pySPACE.tools.filesystem
    """
    parts = path.split(os.sep)
    subpath = ""
    for part in parts[1:]:
        subpath += os.sep + part
        if not os.path.exists(subpath):
            try:
                os.mkdir(subpath)
            except OSError as (err_no, strerr):
                import errno
                # os.path.exists isn't secure on gpfs!
                if not err_no == errno.EEXIST:
                    raise

def save_copy(src, dest):
    """ Check if file exists and is new before copying

    Otherwise a new file name is created.
    """
    if not os.path.exists(dest):
        shutil.copy2(src, dest)
    elif dest.endswith(".yaml"):
        import yaml
        d = yaml.load(open(dest))
        s = yaml.load(open(src))
        if not d == s:
            dest = dest[:-5] + "_new.yaml"
            save_copy(src, dest)
    elif dest.endswith(".csv") or dest.endswith(".rst"):
        pass
    else:
        dest += ".new"
        save_copy(src, dest)

setup_message = """
Creating (or modifying) pySPACEcenter
-------------------------------------

The pySPACEcenter directory is created in your home directory.
It will include subdirectories for specs, storage with some basic examples and
a more elaborate example folder.
Furthermore, the basic configuration file *config.yaml* will be created.
You might want to have a look at this file to find out, what you can change
in your basic configuration.
To simplify the standard execution, links will be created to the main
software scripts called *launch.py*, *launch_live.py* and
*performance_results_analysis.py*.

It is not yet possible to use these commands outside the folders as normal
command line arguments. Moreover, this script is currently only creating
the pySPACE usage folder. It is not yet installing the needed dependencies,
mentioned in the documentation and it is leaving the software, where it is
and it is not moving it to site-packages, as it is usual for other setup
scripts. This will hopefully change in future.
"""


def setup_package():
    """ Copy configuration files to pySPACEcenter folder in the home dir

    If files are already existing, they are not replaced, but new versions are
    additionally stored.
    """
    print setup_message
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    create_directory(os.path.join(home, "pySPACEcenter"))
    #create_directory(os.path.join(home, "pySPACEcenter","examples"))
    create_directory(os.path.join(home, "pySPACEcenter","specs"))
    create_directory(os.path.join(home, "pySPACEcenter","storage"))
    src_conf_file = os.path.join(src_path, "docs", "examples", "conf",
                                 "example.yaml")
    dest_conf_file = os.path.join(home, "pySPACEcenter","config.yaml")
    save_copy(src_conf_file,dest_conf_file)
    
    try:
        os.symlink(os.path.join(src_path, "pySPACE", "run", "launch.py"),
                   os.path.join(home, "pySPACEcenter", "launch.py"))
    except:
        pass
    try:
        os.symlink(os.path.join(src_path, "pySPACE", "run", "launch_live.py"),
                   os.path.join(home, "pySPACEcenter", "launch_live.py"))
    except:
        pass
    try:
        os.symlink(os.path.join(src_path,"pySPACE", "run", "gui",
                   "performance_results_analysis.py"),
                   os.path.join(home, "pySPACEcenter",
                                "performance_results_analysis.py"))
    except:
        pass
        
    examples = os.path.join(src_path,"docs","examples")
    # copying examples folder
    for folder, _, files in os.walk(examples):
        new_folder=os.path.join(home, "pySPACEcenter", "examples",
                                folder[len(examples)+1:])
        for file in files:
            if not file.startswith("."):
                create_directory(new_folder)
                save_copy(os.path.join(src_path, folder, file),
                          os.path.join(new_folder, file))
    # copying important examples to normal structure to have a start
    for folder, _, files in os.walk(examples):
        new_folder=os.path.join(home, "pySPACEcenter",folder[len(examples)+1:])
        for file in files:
            if  "conf" in folder[len(examples)+1:] or \
                ".rst" in file or \
                file.startswith("."):
                    pass
            #"example" in file or "example_summary" in folder:
            elif "example" in file \
                    or "example_summary" in folder[len(examples)+1:] \
                    or file=="functions.yaml"\
                    or "template" in file\
                    or folder.endswith("templates") \
                    or folder.endswith("examples"):
                create_directory(new_folder)
                save_copy(os.path.join(src_path, folder,file),
                          os.path.join(new_folder, file))

    print "The pySPACEcenter should be available now."
#
#    try:
#        shutil.copytree(os.path.join(src_path,"docs","examples"),os.path.join(home, "pySPACEcenter","examples"))
#    except:
#        pass
#
#    # check that we have a version
#    version = get_version()
#    short_description = get_short_description()
#    long_description = get_long_description()
#    # Run build
#    os.chdir(src_path)
#    sys.path.insert(0, src_path)
#
#    setup(name = 'pySPACE', version=version,
#          author = 'pySPACE Developers',
#          author_email = email,
#          maintainer = 'pySPACE Developers',
#          maintainer_email = email,
#          license = "http://pyspace.github.io/pyspace/license.html",
#          platforms = ["Linux, OSX"],
#          url = 'https://github.com/pyspace/',
#          download_url = 'https://github.com/pyspace/pyspace/archive/master.zip',
#          description = short_description,
#          long_description = long_description,
#          classifiers = classifiers,
#          packages = ['pySPACE', 'pySPACE.missions', 'pySPACE.resources', 'pySPACE.environments',
#                      'pySPACE.tests', 'pySPACE.tools', 'pySPACE.run'],
#          package_data = {}
#          )


if __name__ == '__main__':
    setup_package()
