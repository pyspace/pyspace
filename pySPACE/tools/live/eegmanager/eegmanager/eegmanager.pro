######################################################################
# qmake build-script for eegmanager application, part of pySPACE
# this has been tested on windows 7 and Windows XP (32-Bit compatible).
######################################################################

TARGET = eegmanager
DEPENDPATH += .
INCLUDEPATH += .

CONFIG += release console
CONFIG -= app_bundle qt

# Input
HEADERS += glob_module.h \
		   acq_file.h \
           acq_net.h \
           acq_bp.h \
           ctrl_xmlrpc.h \
           eegmanager.h \
           EEGStream.h \
           global.h \
           out_net.h \
           out_file.h \
           util_mutex.h \
           util_thread.h \
		   util_ringbuffer.h \
           utils.h 
SOURCES += glob_module.cpp \
		   gitversion.c \
		   acq_file.cpp \
           acq_net.cpp \
           acq_bp.cpp \
           ctrl_xmlrpc.cpp \
           eegmanager.cpp \
           out_net.cpp \
           out_file.cpp \
           util_mutex.cpp \
           util_thread.cpp \
		   util_ringbuffer.cpp 

unix:!macx {
    DEFINES += __linux__
}

macx {
    DEFINES += __mac__
}

unix {
    
    HEADERS -= acq_bp.h
    SOURCES -= acq_bp.cpp

	INCLUDEPATH += . ../deps/xmlrpc++
	LIBS += -L../deps/xmlrpc++ -L/opt/local/lib
	LIBS += -lxmlrpc -lpthread
}


win32 {

    DEFINES += __windows__
	INCLUDEPATH += . "..\\deps\\xmlrpc++" "..\\deps\\pthreads-win32"
	LIBS += -L"..\\deps\\xmlrpc++" -L"..\\deps\\pthreads-win32\\dll\\x86" 
	LIBS += -lxmlrpc -lWs2_32 -lpthreadGC2
}
