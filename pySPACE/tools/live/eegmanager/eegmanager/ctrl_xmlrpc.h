#ifndef __CTRL_XMLRPC__
#define __CTRL_XMLRPC__

#include <map>
#include "global.h"

#ifdef WIN32
#include <windows.h>
#endif // WIN32

#include "XmlRpc.h"
#include "eegmanager.h"

class CRTLXmlrpc; // forward declaration

// ---

class showModulesCallback : public XmlRpc::XmlRpcServerMethod
{
public:
	showModulesCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
 	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
	std::string help ();
private:
	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class getUsageCallback : public XmlRpc::XmlRpcServerMethod
{
public:
	getUsageCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
 	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
	std::string help ();
private:
	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class applySetupCallback: public XmlRpc::XmlRpcServerMethod
{
public:
	applySetupCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
 	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
	std::string help ();
private:
	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class getProcessState : public XmlRpc::XmlRpcServerMethod
{
public:
	getProcessState (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
 	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
	std::string help ();
private:
	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class getCurrentSetup : public XmlRpc::XmlRpcServerMethod
{
public:
	getCurrentSetup (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
 	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
	std::string help ();
private:
	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class addModuleCallback : public XmlRpc::XmlRpcServerMethod
{
public:
	addModuleCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
  	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
  	std::string help ();
private:
  	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class startLoopbackCallback : public XmlRpc::XmlRpcServerMethod
{
public:
	startLoopbackCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
  	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
  	std::string help ();
private:
  	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class startWorkingCallback : public XmlRpc::XmlRpcServerMethod
{
public:
	startWorkingCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
  	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
  	std::string help ();
private:
  	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class stopWorkingCallback : public XmlRpc::XmlRpcServerMethod
{
public:
	stopWorkingCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
 	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
	std::string help ();
private:
	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class shutDownCallback : public XmlRpc::XmlRpcServerMethod
{
public:
	shutDownCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
 	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
	std::string help ();
private:
	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class stdoutReadCallback : public XmlRpc::XmlRpcServerMethod
{
public:
	stdoutReadCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
 	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
	std::string help ();
private:
	CRTLXmlrpc* ctrl_xmlrpc;
	long position;
	char name[64];
};

// ---

class selfTestCallback : public XmlRpc::XmlRpcServerMethod
{
public:
	selfTestCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* abriXmlServer);
 	void execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result);
	std::string help ();
private:
	CRTLXmlrpc* ctrl_xmlrpc;
};

// ---

class CRTLXmlrpc : public Thread
{

public:

	CRTLXmlrpc (int port = 16253);
	~CRTLXmlrpc ();

	void run ();
	void stop();
	int32_t startProcessing();
	void startEegServer (int port = 51244, int blockSize = 100);
	void startEegServer (std::string filename, int port = 51244, int blockSize = 100);
	void startLoopback (int acq_port, std::string acq_ip, int out_port);

	void getBufferState(XmlRpc::XmlRpcValue& result);
	void getBufferBandwidth(XmlRpc::XmlRpcValue& result);
	void getProcState(XmlRpc::XmlRpcValue& result);
	void getCurSetup(XmlRpc::XmlRpcValue& result);
	int32_t getUsage(std::string name);
	int32_t addModule(std::string name, std::string opts);

	void stopEegServer ();

private:

	int port;
	bool working;
	bool started;

	EEGManager* eegmanager;

	XmlRpc::XmlRpcServer* server;

	startWorkingCallback* start_working;
	stopWorkingCallback*  stop_working;
	shutDownCallback*    shut_down;
	startLoopbackCallback* start_loopback;
	showModulesCallback* show_modules;
	addModuleCallback* add_module;
	getUsageCallback* get_usage;
	getProcessState* get_state_cb;
	getCurrentSetup* get_setup_cb;
	applySetupCallback* apply_setup_cb;
	stdoutReadCallback* stdout_read_cb;
	selfTestCallback* self_test_cb;
	
};

#endif // __CTRL_XMLRPC__
