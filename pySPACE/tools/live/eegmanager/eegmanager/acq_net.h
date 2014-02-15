#ifndef __ACQ_NET_H__
#define __ACQ_NET_H__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>

#ifndef WIN32
#include <arpa/inet.h>
#endif

#include "global.h"
#include "EEGStream.h"
#include "glob_module.h"
#include "util_thread.h"
#include "utils.h"


#define DEFAULT_PORT 61244
#define DEFAULT_IP "127.0.0.1"

class NETAcquisition : public Module
{
public:
	NETAcquisition();
	NETAcquisition(std::string host, uint32_t port);

	std::string inputType();
	std::string outputType();
	std::string description();

	int32_t setup(std::string opts);

	~NETAcquisition();

	// overwrite the default (IPC) getMessage function
	virtual int32_t getMessage(int type);

private:

	void run();

	int32_t _connect();
	int32_t _disconnect();
	int connected;

	int server_socket;

	char server_address[32];
	int server_port;
	int32_t num_timeouts;
	int32_t max_timeouts;

	REGISTER_DEC_TYPE(NETAcquisition);
};
#endif // __ACQ_NET_H__
