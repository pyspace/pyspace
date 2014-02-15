#ifndef __NETOUT_H__
#define __NETOUT_H__

#include "global.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>

#include "EEGStream.h"
#include "util_thread.h"
#include "glob_module.h"
#include "utils.h"


#define DEFAULT_LISTEN_PORT 61244
//#define DEFAULT_IP "10.250.3.43"


class NETOutput : public Module
{
public:
	NETOutput();
	NETOutput(uint32_t _listen_port, MessageHeader* _startmsg);
	~NETOutput();

	std::string inputType();
	std::string outputType();
	std::string description();

	int32_t setup(std::string opts);
	
	virtual int32_t putMessage(MessageHeader* pHeader);

protected:

	static void* _manage_sockets(void* ptr);
	pthread_t* socket_manager;
	int32_t _add_socket(int new_client);
	int32_t _send_start_msg(int new_client);
	int32_t _send_stop_msg(int old_client);
	int32_t _disconnect(int client_id);

	int32_t process(void);

	pthread_mutex_t server_mutex;
	int listen_socket;

	int* clients;
	uint32_t num_clients;

	bool socket_manager_should_work;

private:

	void run();

	int listen_port;
	struct sockaddr_in server_address;

	int blocking;

	REGISTER_DEC_TYPE(NETOutput);

};

#endif // __NETOUT_H__
