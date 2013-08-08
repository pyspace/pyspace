
INSTALLATION / COMPILATION on WINDOWS-XP and WINDOWS-7
------------------------------------------------------

1: install minGW

http://sourceforge.net/projects/mingw/files/latest/download?source=files

- make sure, C++ compiler is checked in installation options
- ignore the error saying "no valid w32api.h in MinGW"
- add MinGW binary DIR to systems PATH variable (e.g. C:\MinGW\bin)

2: install Qt4/qmake

http://releases.qt-project.org/qt4/source/qt-win-opensource-4.8.4-mingw.exe

- reference your MinGW location during installation (e.g. C:\MinGW\)
- add Qt binary DIR to systems PATH variable (e.g. C:\Qt\4.8.4\bin)

3: compile xmlrpc++ and eegmanager

- navigate to $(repo)\pySPACE\tools\eegmanager-win32
- qmake && make

4: install Python, Python Package and bindings

http://www.python.org/ftp/python/2.7.3/python-2.7.3.msi

- add Python binary DIR to systems PATH (e.g. C:\Python27)

- install yaml package
http://pyyaml.org/download/pyyaml/PyYAML-3.10.win32-py2.7.exe

- install pyQT bindings
http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.10/PyQt4-4.10-gpl-Py2.7-Qt4.8.4-x32.exe

5: use the GUI

- navigate to $(repo)\pySPACE\tools\eegmanager-win32
- start the gui with python eegmanager_gui.py