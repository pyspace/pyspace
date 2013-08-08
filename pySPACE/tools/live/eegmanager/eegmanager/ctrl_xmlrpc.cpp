#include "ctrl_xmlrpc.h"

//****************************************************************************//

showModulesCallback::showModulesCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("show_modules", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void showModulesCallback::execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{
	int i = 0;

	ModuleFactory::map_type* mods = ModuleFactory::getMap();
	ModuleFactory::map_type::iterator it;

	if(mods->size() == 0) {
		WTF("no modules are registered!");
		result = 0;
	} else {
		std::vector<std::string> input_mods;
		std::vector<std::string> flow_mods;
		std::vector<std::string> output_mods;

		for(it = mods->begin(); it != mods->end(); it++) {
			Module* m = ModuleFactory::createInstance((std::string)it->first);
			// input, output, or flow node?
			if(!m->isInput() && !m->isOutput()) {
				flow_mods.push_back((std::string)it->first);
			} else if(m->isInput()) {
				input_mods.push_back((std::string)it->first);
			} else {
				output_mods.push_back((std::string)it->first);
			}
			delete m;
		}

		result.setSize(int(mods->size()));
		std::vector<std::string>::iterator itm;

		for(itm = input_mods.begin(); itm != input_mods.end(); itm++) {
			result[i++] = (*itm);
		}
		for(itm = flow_mods.begin(); itm != flow_mods.end(); itm++) {
			result[i++] = (*itm);
		}
		for(itm = output_mods.begin(); itm != output_mods.end(); itm++) {
			result[i++] = (*itm);
		}

	}

	fflush(stdout);

}

std::string showModulesCallback::help()
{
	return std::string("Return a list of available modules in the eegmanager.");
}


//****************************************************************************//

getUsageCallback::getUsageCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("usage", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void getUsageCallback::execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{

	if(!params.valid()) {
		FYI("Invalid Params from XMLRPC");
		result = -1;
		return;
	}

	Module* temp = ModuleFactory::createInstance((std::string)params[0]);
	if(temp == NULL) {
		FYI("Could not create Object of %s", ((std::string)params[0]).c_str());
		result = -1;
		return;
	}

	result.setSize(4);

	// the usage string
	result[0] = temp->usage();

	// input type
	result[1] = temp->inputType();

	// output type
	result[2] = temp->outputType();

	// input, output, or flow node?
	if(!temp->isInput() && !temp->isOutput()) {
		result[3] = std::string("flow");
	} else if(temp->isInput()) {
		result[3] = std::string("input");
	} else {
		result[3] = std::string("output");
	}
	delete temp;

	fflush(stdout);

	return;

}

std::string getUsageCallback::help()
{
	return std::string("see usage of a module (e.g. usage(\"FILEAcquisition\")");
}

//****************************************************************************//

applySetupCallback::applySetupCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("apply_setup", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void applySetupCallback::execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{

	if(params.valid()) {

		if(params.size() != 1) {
			result = -1;
			return;
		}

		XmlRpc::XmlRpcValue p = params[0];
		if(p.size() % 2 != 0) {
			result = -1;
			fflush(stdout);
			return;
		}

		for(int i=0; i<p.size(); i+= 2) {
			if(0 > ctrl_xmlrpc->addModule(p[i], p[i+1])) {
				result = i+1;
				fflush(stdout);
				return;
			}
		}
	}
	result = 0;
	fflush(stdout);

	return;
}

std::string applySetupCallback::help()
{
	return std::string("apply a setup retrieved with \'get_setup\' call");
}

//****************************************************************************//

getProcessState::getProcessState (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("get_state", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void getProcessState::execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{

	this->ctrl_xmlrpc->getProcState(result);
	fflush(stdout);
	return;
}

std::string getProcessState::help()
{
	return std::string("Returns a string describing the state of the process (e.g. IDLE).");
}


//****************************************************************************//

getCurrentSetup::getCurrentSetup (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("get_setup", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void getCurrentSetup::execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{
	ctrl_xmlrpc->getCurSetup(result);
	fflush(stdout);
	return;
}

std::string getCurrentSetup::help()
{
	return std::string("Returns a string list with module-names and parameters.");
}

//****************************************************************************//

addModuleCallback::addModuleCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("add_module", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void addModuleCallback::execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{
	result = 0;
	if (params.valid())
	{
		result = ctrl_xmlrpc->addModule((std::string) params[0], (std::string) params[1]);
	} else {
		FYI("Error reading XMLRPC Message..");
	}
	fflush(stdout);
}

std::string addModuleCallback::help()
{
	return std::string("Add a module to the processing chain (e.g. add_module(\"<name>\", \"<options>\"))");
}

//****************************************************************************//

startLoopbackCallback::startLoopbackCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("start_loopback", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void startLoopbackCallback::execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{
	// if no arguments were received
	if (!params.valid())
	{
		ctrl_xmlrpc->startProcessing();
	}
	else
	{
		switch(params.size())
		{
			case 3:
				FYI("params[0]: %d\n\tparams[1]: %s\n\tparams[2]: %d\n\t", (int)params[0], ((std::string)params[1]).c_str(), (int)params[2]);
				ctrl_xmlrpc->startLoopback((int)params[0], (std::string) params[1], (int) params[2]);
				break;
			default:
				WTF("WARNING! Server got wrong number of arguments!");
				break;
		}
	}

	fflush(stdout);
  	result = 0;
}

std::string startLoopbackCallback::help()
{
	return std::string("Start the Loopback EEG server!");
}

//****************************************************************************//

startWorkingCallback::startWorkingCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("start", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void startWorkingCallback::execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{
	// if no arguments were received
	if (!params.valid())
	{
		result = ctrl_xmlrpc->startProcessing();
	}
	else
	{
		switch(params.size())
		{
			// start the server in online mode with port and blocksize
			case 2:
				ctrl_xmlrpc->startEegServer((int) params[0], (int) params[1]);
                result = 0;
				break;
			case 3:
				ctrl_xmlrpc->startEegServer((std::string) params[0], (int) params[1], (int) params[2]);
                result = 0;
				break;
			default:
				OMG("WARNING! Server got wrong number of arguments!");
				break;
		}
	}
	fflush(stdout);
}

std::string startWorkingCallback::help()
{
	return std::string("Start the configured eegmanager.");
}

//****************************************************************************//

stopWorkingCallback::stopWorkingCallback(XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("stop", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void stopWorkingCallback::execute(XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{
	ctrl_xmlrpc->stopEegServer();
	fflush(stdout);
  	result = 0;
}

std::string stopWorkingCallback::help()
{
	return std::string("Stop the running eegmanager.");
}

//****************************************************************************//

shutDownCallback::shutDownCallback(XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("shut_down", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void shutDownCallback::execute(XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{
	// if there is an EEG server running, stop it
	ctrl_xmlrpc->stopEegServer();
	// stop the aBRI-XmlRpc-server
	ctrl_xmlrpc->stop();

	fflush(stdout);
  	result = 0;
}

std::string shutDownCallback::help()
{
	return std::string("Shut down the eegmanager.");
}

//****************************************************************************//

stdoutReadCallback::stdoutReadCallback(XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("stdout", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
	position = 0;
	memset(name, 0, 64);
#ifdef __windows__
	sprintf(name, ".%d.stdout", GetCurrentProcessId());
#else
	sprintf(name, ".%d.stdout", getpid());
	//sprintf(name, ".%d.stdout", 42);
#endif
}

void stdoutReadCallback::execute(XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{
	FILE* stdout_handle = fopen(name, "r");
	fseek(stdout_handle, position, SEEK_SET);

	std::string output;
	char line[512];

	while(!feof(stdout_handle)) {
		memset(line, 0, 512);
		fread(line, sizeof(char), 511, stdout_handle);
		output += line;
	}
	result = output;

	position = ftell(stdout_handle);
	fclose(stdout_handle);
	return;
}

std::string stdoutReadCallback::help()
{
	return std::string("Forwards recent stdout output to the caller.");
}

//****************************************************************************//

selfTestCallback::selfTestCallback (XmlRpc::XmlRpcServer* s, CRTLXmlrpc* ctrl_xmlrpc)
	: XmlRpc::XmlRpcServerMethod("self_test", s)
{
	this->ctrl_xmlrpc = ctrl_xmlrpc;
}

void selfTestCallback::execute (XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{

	ModuleFactory::map_type* mods = ModuleFactory::getMap();
	ModuleFactory::map_type::iterator it;

	if(mods->size() == 0) {
		WTF("no modules are registered!");
		result = 0;
	} else {

		for(it = mods->begin(); it != mods->end(); it++) {
			printf("\n** creating node with name %s **\n", ((std::string)it->first).c_str());
			Module* m = ModuleFactory::createInstance((std::string)it->first);
			printf("description : %s\n", m->description().c_str());
			printf("input type  : %s\n", m->inputType().c_str());
			printf("is input    : %s\n", m->isInput() ? "INPUT" : "NO");
			printf("output type : %s\n", m->outputType().c_str());
			printf("is output   : %s\n", m->isOutput() ? "OUTPUT" : "NO");
			printf("is strem    : %s\n", m->isStream() ? "STREAM" : "NO");
			printf("usage       : %s\n", m->usage().c_str());
			delete m;
			printf("** module successfully deleted ** \n\n");
		}
	}
	fflush(stdout);
	result = 1;
}

std::string selfTestCallback::help()
{
	return std::string("Query the internal module list and create/destroy objects, testing needed methods.");
}

//****************************************************************************//

CRTLXmlrpc::CRTLXmlrpc (int port)
{
	this->port = port;
	this->eegmanager = NULL;
	started = false;
	// create new server
	server = new XmlRpc::XmlRpcServer;
	// create new callback objects
	start_working = new startWorkingCallback (server, this);
	stop_working  = new stopWorkingCallback (server, this);
	shut_down    = new shutDownCallback(server, this);
	start_loopback = new startLoopbackCallback(server, this);
	show_modules = new showModulesCallback(server, this);
	add_module = new addModuleCallback(server, this);
	get_usage = new getUsageCallback(server, this);
	get_state_cb = new getProcessState(server, this);
	get_setup_cb = new getCurrentSetup(server, this);
	apply_setup_cb = new applySetupCallback(server, this);
	stdout_read_cb = new stdoutReadCallback(server, this);
	self_test_cb = new selfTestCallback(server, this);

	// Create the server socket on the specified port
	if(!server->bindAndListen(port)) {
		exit(11);
	}
	// Enable introspection
	server->enableIntrospection(true);
}

CRTLXmlrpc::~CRTLXmlrpc ()
{
	if (working == true)
	{
		working = false;
		msleep(100);
	}

	server->XmlRpcServer::shutdown();

	if (start_working) delete start_working;
	if (stop_working) delete stop_working;
	if (shut_down) delete shut_down;
	if (start_loopback) delete start_loopback;
	if (add_module) delete add_module;
	if (get_usage) delete get_usage;
	if (get_state_cb) delete get_state_cb;
	if (get_setup_cb) delete get_setup_cb;
	if (apply_setup_cb) delete apply_setup_cb;
	if (stdout_read_cb) delete stdout_read_cb;
	if (self_test_cb) delete self_test_cb;

	if (server) delete server;
}

void CRTLXmlrpc::getProcState(XmlRpc::XmlRpcValue& result)
{

	result.setSize(3);
	result[1] = 0.0;
	result[2] = 0.0;

	if(eegmanager == NULL) {
		result[0] = std::string("IDLE");
		return;
	}

	// update local state variable
	if(!eegmanager->isRunning()) started = false;

	if(eegmanager->isRunning()) {
		result.setSize(eegmanager->bandwidth.size()*2 + 1);
		result[0] = std::string("RUNNING");
		result.setSize(eegmanager->bandwidth.size());
		for(uint32_t i=0; i<eegmanager->bandwidth.size(); i++) {
			result[(i*2)+1] = eegmanager->bandwidth[i];
			result[(i*2)+2] = eegmanager->fill[i];
		}
		return;
	}

	if(eegmanager->current_setup().size() > 0) {
		result[0] = std::string("CONFIGURED");
		return;
	}

	result[0] = std::string("IDLE");

	return;
}

void CRTLXmlrpc::getCurSetup(XmlRpc::XmlRpcValue& result)
{
	if(eegmanager == NULL) {
		result.setSize(0);
		return;
	}

	int i=0;
	std::vector< std::pair<std::string, std::string> > s;
	std::vector< std::pair<std::string, std::string> >::iterator it;

	s = eegmanager->current_setup();
	result.setSize(s.size()*2);

	for(it=s.begin(); it!=s.end(); it++) {
		result[i++] = (*it).first;
		result[i++] = (*it).second;
	}

	return;
}

int32_t CRTLXmlrpc::addModule(std::string name, std::string opts)
{
	if(eegmanager == NULL) {
		eegmanager = new EEGManager();
	}
	return eegmanager->add_module(name, opts);
}

void CRTLXmlrpc::startEegServer (int port, int blockSize)
{
	if(eegmanager != NULL && started){
		return;
	}

	eegmanager = new EEGManager();


	char buf[512];
	memset(buf, 0, 512);
	sprintf(buf, "--blocksize %d", blockSize);
	eegmanager->add_module("BPAcquisition", buf);
#ifdef __mac__
	memset(buf, 0, 512);
	eegmanager->add_module("FLOWRealtime", buf);
	memset(buf, 0, 512);
	sprintf(buf, "--port 51255");
	eegmanager->add_module("FLOWIpmarker", buf);
#endif
	memset(buf, 0, 512);
	sprintf(buf, "--port %d", port);
	eegmanager->add_module("NETOutput", buf);

	startProcessing();
}

void CRTLXmlrpc::startEegServer (std::string filename, int port, int blockSize)
{
	if(started) return;
	
	if(eegmanager == NULL) {
		eegmanager = new EEGManager();
	} else {
		if(eegmanager->isRunning()) stopEegServer();
	}

	char buf[512];
	memset(buf, 0, 512);
	sprintf(buf, "--filename %s --blocksize %d", filename.c_str(), blockSize);
	eegmanager->add_module("FILEAcquisition", buf);
	memset(buf, 0, 512);
	sprintf(buf, "--port %d --blocking", port);
	eegmanager->add_module("NETOutput", buf);

	startProcessing();
}

void CRTLXmlrpc::startLoopback (int acq_port, std::string acq_ip, int out_port)
{

	if(eegmanager != NULL){
		stopEegServer();
	} else {
		eegmanager = new EEGManager();
	}

	char buf[512];
	memset(buf, 0, 512);
	sprintf(buf, "--host %s --port %d", acq_ip.c_str(), acq_port);
	eegmanager->add_module("NETAcquisition", buf);
	eegmanager->add_module("FLOWProcessor", "--devicename /dev/flow");
	memset(buf, 0, 512);
	sprintf(buf, "--port %d --blocking", out_port);
	eegmanager->add_module("NETOutput", buf);

	startProcessing();
}

int32_t CRTLXmlrpc::startProcessing ()
{
	if(started) return -1;
	
	if(eegmanager == NULL) {
		OMG("no eegmanager here!?");
        return -1;
	} else {
		if(eegmanager->check() == 0) {
			((Thread*)eegmanager)->start();
			started = true;
		} else {
            return -1;
        }
	}
    
    return 0;
}

void CRTLXmlrpc::stopEegServer ()
{
	if (eegmanager != NULL)
	{
		if(started) {
			eegmanager->stop();
			FYI("stopping server");
			while(eegmanager->isRunning()) {
				FYI("waiting..");
				msleep(500);
//				eegmanager->wait();
			}
		} else {
			FYI("deleting server");
			delete eegmanager;
			eegmanager = NULL;
		}
	} else {
		FYI("no server present");
	}

	FYI("Done!");
	started = false;
}

void CRTLXmlrpc::run ()
{
	working = true;

	while (working) {
		server->work(10.0);
	}

	return;
}

void CRTLXmlrpc::stop ()
{
	working = false;
}


int main (int argc, char *argv[])
{
	CRTLXmlrpc* server = NULL;
//	XmlRpc::setVerbosity(100);

	char name[64];
	memset(name, 0, 64);
#ifdef __windows__
	sprintf(name, ".%d.stdout", GetCurrentProcessId());
#else
	sprintf(name, ".%d.stdout", getpid());
	//sprintf(name, ".%d.stdout", 42);
#endif
	freopen(name, "w", stdout);

	const char* BUILD_DATE = __DATE__;
	const char* BUILD_TIME = __TIME__;
	printf("BUILD: %s, %s\n", BUILD_DATE, BUILD_TIME);

	extern const char* gitversion;
	printf("GIT  : %s\n", gitversion);

	fflush(stdout);

	int port;

	switch(argc)
	{
		// standard call; XmlRpc-server running on port 16253
		case 1:
			server = new CRTLXmlrpc();
			server->start();
			break;

		// variable port; XmlRpc-server running on port <port>
		case 2:
			port = atoi(argv[1]);

			if (port == 0)
			{
				std::cerr << "ERROR! Given argument (" << argv[1] << ") must be an integer!" << std::endl;
				return -1;
			}

			server = new CRTLXmlrpc(port);
			server->start();
			break;

		default:
			std::cerr << "ERROR! Wrong number of arguments! The EEG-server takes no or just one integer argument!" << std::endl;
			return -1;
	}
	
    if(!(server->isRunning())) return -1;

	// wait here until the thread exits
	while(server->isRunning()) {
		msleep(3000);
	}

	// if there is an aBRI_XmlServer running, stop it
	if (server)
	{
		server->stop();
		server->wait();
		delete server;
		server = NULL;
	}

	fclose(stdout);

	return 0;
}

