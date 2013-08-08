######################################################################
# qmake build-script for eegmanager application, part of pySPACE
# this has been tested on windows 7 and Windows XP (32-Bit compatible).
######################################################################

DEPENDPATH += .
INCLUDEPATH += .
TEMPLATE = subdirs

CONFIG += release console ordered
CONFIG -= app_bundle qt

SUBDIRS = "deps\\xmlrpc++" "eegmanager"
