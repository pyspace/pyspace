""" File handling helper functions """
import os
import fnmatch
import tarfile
import warnings


def get_author():
    try:
        import platform
        CURRENTOS = platform.system()
        if CURRENTOS == "Windows":
            import getpass
            author = getpass.getuser()
        else:
            import pwd
            author = pwd.getpwuid(os.getuid())[4]
    except:
        author = "unknown"
        warnings.warn("Author could not be resolved.")

    return author

def create_directory(path):
    """ Create the given directory path recursively """
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

def common_path(path1, path2, common=[]):
    """ Compute the common part of two paths *path1* and *path2* """
    if len(path1) < 1: 
        return (common, path1, path2)
    if len(path2) < 1: 
        return (common, path1, path2)
    if path1[0] != path2[0]: 
        return (common, path1, path2)
    return common_path(path1[1:], path2[1:], common + [path1[0]])

def get_relative_path(path1, path2):
    """ Return the relative path of *path1* to *path2* """
    (common,l1,l2) = common_path(path1.split(os.path.sep), 
                                 path2.split(os.path.sep),)
    p = []
    if len(l1) > 0:
        p = [os.pardir + os.sep for i in range(len(l1)-1)]
    p = p + l2
    return os.path.join( *p )

def locate(pattern, root=os.curdir):
    """ Locate all files matching pattern in root directory.

    Locate all files matching supplied filename pattern in and below
    the supplied root directory.

    **Parameters**

        :pattern:
            The pattern (regular expression) the files, that are selected, must match

        :root:
            The root directory of the directory tree in which files are searched

    :Source: http://code.activestate.com/recipes/499305/

    .. todo:: transfer to file handling?
    """
    for path, dirs, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)


def create_source_archive(archive_path, packages=["pySPACE"],
                          patterns=["*.py", "*.yaml"]):
    """ Store the source code of important packages

    Locates all files in the directory structure of the given :packages:
    that match the given :patterns:. Add these files to an archive that is
    stored in :archive_path:.

    :Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
    :Created: 2010/08/12
    """
    # Create archive file
    archive_file = tarfile.open(archive_path + os.sep + "source_code_archive.tbz",
                                "w:bz2")
    # Find all packages the directories in which they are located
    package_root_dirs = [__import__(package).__path__[0] + os.sep + os.pardir
                            for package in packages]

    orig_dir = os.curdir
    for package, package_root_dir in zip(packages, package_root_dirs):
        os.chdir(package_root_dir)
        # Find all files in the package that match one of the patterns and add
        # them to the archive.
        for pattern in patterns:
            for file in locate(pattern, package):
                archive_file.add(file)

    archive_file.close()
    os.chdir(orig_dir)