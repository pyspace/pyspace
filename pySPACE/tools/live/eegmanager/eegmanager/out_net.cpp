#include "out_net.h"

NETOutput::NETOutput()
{
	// Options this module understands
	struct option long_options[] =
	{
		{"blocking",	no_argument,    &blocking, 1},
		{"port",	required_argument,       0, 'p'},
		{0, 0, 0, 0}
	};

	merge_options(long_options, sizeof(long_options));

	blocking = 0;
	working = false;
	num_clients = 0;
	clients = NULL;
	listen_socket = 0;
	listen_port = 0;

	socket_manager_should_work = false;
	socket_manager = NULL;

    pthread_mutex_init(&server_mutex, NULL);
}

NETOutput::~NETOutput()
{
	socket_manager_should_work = false;

	if(socket_manager != NULL) {
		pthread_join(*socket_manager, NULL);
		delete socket_manager;
	}

	pthread_mutex_lock(&server_mutex); // vvvvvvvvv
	if(clients != NULL) freee(clients);
	pthread_mutex_unlock(&server_mutex); // vvvvvvvvv

	if(listen_socket != 0) {
		// clear possible data, which may be still in some deeper osi-layers
		shutdown(listen_socket, 2);
#ifdef __windows__
		closesocket(listen_socket);
#else
		close(listen_socket);
#endif
		listen_socket = 0;
	}
}


int32_t NETOutput::setup(std::string opts)
{

	parameters = new std::string(opts);
	if(opts.size() == 0) return -1;

    int option_index = 0;
    // force reset of the getopt subsystem
    optind = 0;

	int argc;
	char** argv;

	string2argv(&opts, &argc, argv);

	int c;

    // Parse Arguments
	while (1) {

        c = getopt_long(argc, argv, "p:", module_options, &option_index);
        if(c == -1) break;


		switch (c) {
		case 'p':
			listen_port = atoi(optarg);
			break;
		case '?':
			if (optopt == 'p') {
				OMG("Option -%c requires an argument.\n", optopt);
			} else if (isprint (optopt)) {
				WTF("Unknown option `-%c'.\n", optopt);
			} else {
				WTF("Unknown option character `\\x%x'.\n", optopt);
			}
			break;
		default:
			break;
		}
	}

    // cleanup std::string conversion
    for(int j=0; j<argc; j++) {
        freee(argv[j]);
    }
    freee(argv);
    argc = 0;

	int opt = 1;
	int ret = 0;

    if(listen_port < 1024 || listen_port == 0) return -1;

#ifdef __windows__
	WSADATA wsaData;
    int err;
    err = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (err != 0) {
        /* Tell the user that we could not find a usable */
        /* Winsock DLL.                                  */
        OMG("WSAStartup failed with error: %d\n", err);
    }
#endif

    listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(listen_socket < 0) {
		WTF("socket creation failed");
	}

	if(setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt)) <= SOCKET_ERROR) {
#ifdef __windows__
		OMG("SO_REUSEADDR failed for listen_socket with error %d", WSAGetLastError());
#else
		OMG("SO_REUSEADDR failed for listen_socket with error %d", errno);
#endif	
	}

	server_address.sin_family = AF_INET;
	server_address.sin_addr.s_addr = INADDR_ANY;
	server_address.sin_port = htons(listen_port);

	// connect the normal socket
	ret = bind(listen_socket, (struct sockaddr *)&server_address, sizeof(server_address));
	if(0 > ret) {
#ifdef __windows__
		WTF("bind failed with Error %d!", WSAGetLastError());
#else
		WTF("bind failed with Error %d!", errno);
#endif
		return -1;
	}

	listen(listen_socket, 5);

	if(socket_manager != NULL) {
		delete socket_manager;
		socket_manager = NULL;
	}
	socket_manager = new pthread_t;
	if( 0 > pthread_create(socket_manager, NULL, &NETOutput::_manage_sockets, this)) {
		OMG("failed to start socket_manager thread");
		return -1;
	}

	socket_manager_should_work = true;

	return 0;
}


void NETOutput::run()
{
	working = true;

	getMessage(1);
	if(in_data_description == NULL) {
		WTF("IPC Error: no startmessage");
		goto error;
	}
	if(in_data_description->message_type != 1) {
		WTF("IPC Error: message was no startmessage, was type %d", in_data_description->message_type);
		goto error;
	}

	// since we do not change any data properties
	// BUT endianess is possible reverted
	if(_e32((int32_t)1) != (int32_t)1) {
		out_data_description = (EEGStartMessage*)malloc(in_data_description->message_size);
		if(out_data_description == NULL) {
			WTF("Not enough memory to allocate out_data_description of size %d", in_data_description->message_size);
			goto error;
		}
		memcpy(out_data_description, in_data_description, in_data_description->message_size);
		endianflip((MessageHeader**)&out_data_description, TO_NETWORK);
		out_data_description->message_size = _e32(out_data_description->message_size);
		out_data_description->message_type = _e32(out_data_description->message_type);
	} else {
		out_data_description = in_data_description;
	}

	while(working) {
		getMessage(4);

		process();

		while(0 == putMessage((MessageHeader*)out_data_message)) msleep(1);
	}

	socket_manager_should_work = false;
	pthread_join(*socket_manager, NULL);

error:
	// clear possible data, which may be still in some deeper osi-layers
	shutdown(listen_socket, 2);
#ifdef __windows__
	closesocket(listen_socket);
#else
	close(listen_socket);
#endif
	listen_socket = 0;

	working = false;

	return;
}


int32_t NETOutput::putMessage(MessageHeader* pHeader)
{
	if(pHeader == NULL || working == false) return 1;

	if(pHeader->message_type == 3) working = false;

	if(num_clients == 0) {
		if(blocking) {
			return 0;
		} else {
			return pHeader->message_size;
		}
	}

	while(0 != pthread_mutex_trylock(&server_mutex)) {
		FYI("trylock failed!");
		msleep(0);
	}

	int buffer[num_clients];
	int b = 0;

	int* c = clients;
	char rdy;
	uint32_t sent;
	int ret = 0;
	uint32_t current_length = 0;

	char* msgp;

	// if we are on microblaze
	if(_e32((int32_t)1) != (int32_t)1) {
		endianflip(&pHeader, TO_NETWORK);
		pHeader->message_size = _e32(pHeader->message_size);
		pHeader->message_type = _e32(pHeader->message_type);
	}

	for(uint32_t i = 0; i<num_clients; i++) {
		rdy = 0;
		ret = recv(*c, (char*)&rdy, sizeof(char), 0);
		if(0 > ret) {
			OMG("Error %d during recv", ret);
			buffer[b++] = i;
			c++;
			continue;
		}

		if(rdy == 0) {
			OMG("client not reachable");
			buffer[b++] = i;
		} else {
			sent = 0;
			msgp = (char*)pHeader;
			current_length = 0;
			while(sent < pHeader->message_size) {
				current_length = std::min(pHeader->message_size - sent, (uint32_t)1500);
				ret = send(*c, msgp, current_length, 0);
				if(0 > ret) {
					OMG("Error %d during send", ret);
					buffer[b++] = i;
					break;
				}
				sent += ret;
				msgp += ret;
			}
		}
		c++;
	}

	pthread_mutex_unlock(&server_mutex);

	// cleanup list of clients due to possible errors
	for(int i=0; i<b; i++) _disconnect(buffer[i]);

	if(pHeader->message_type == 3) working = false;

	return (int32_t)sent;
}

void* NETOutput::_manage_sockets(void* ptr)
{
	FYI("running");
	int accept_socket;
	int sel_ret;
	int err = 0;
	fd_set sockets;

	struct sockaddr_in client_address;
	uint32_t client_address_len = sizeof(client_address);

	struct timeval sock_timeout;

//	while(!((NETOutput*)ptr)->socket_manager_should_work) msleep(500);

	while(((NETOutput*)ptr)->out_data_description == NULL \
			&& ((NETOutput*)ptr)->socket_manager_should_work) msleep(500);

	while(((NETOutput*)ptr)->socket_manager_should_work) {
		FD_ZERO(&sockets);
		FD_SET(((NETOutput*)ptr)->listen_socket, &sockets);
		sock_timeout.tv_sec = 1;
		sock_timeout.tv_usec = 0;
		sel_ret = select(((NETOutput*)ptr)->listen_socket+1, &sockets, 0, 0, &sock_timeout);

		if(sel_ret > 0) {
#ifndef __windows__
			accept_socket = accept(((NETOutput*)ptr)->listen_socket, \
					(struct sockaddr *)&client_address, \
					(socklen_t*)&client_address_len);
#else
			accept_socket = accept(((NETOutput*)ptr)->listen_socket, \
					(sockaddr * ) &client_address, \
					(int*) &client_address_len);
#endif
			if(accept_socket != -1) {
				FYI("new connection from %s", inet_ntoa(client_address.sin_addr));
				((NETOutput*)ptr)->_add_socket(accept_socket);
			}
		}
	}

	((NETOutput*)ptr)->_disconnect(-1);
	FYI("Done!");

	return 0;
}

int32_t NETOutput::_add_socket(int new_client)
{
	int *c;
	int ret = 0;
	int opt = 1;

	if(setsockopt(new_client, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt)) == -1) {
		WTF("SO_REUSEADDR failed for new_client");
	}

	if(0 > _send_start_msg(new_client)) {
		OMG("new connection stalled!");
		goto error;
	}

	while(0 != pthread_mutex_trylock(&server_mutex)) { // vvvvvvvvv
		msleep(0);
	}
	clients = (int*)realloc(clients, (num_clients + 1)*sizeof(int));
	if(clients == NULL) {
		WTF("cannot realloc clients to %d bytes", (num_clients + 1)*sizeof(int));
		ret = -1;
		goto error;
	}
	c = clients;
	c += num_clients;
	memcpy(c, (void*)&new_client, sizeof(int));
	num_clients++;

	error:
	pthread_mutex_unlock(&server_mutex); // ^^^^^^^^

	return 0;
}

int32_t NETOutput::_send_start_msg(int new_client)
{
	int ret = -1;
	uint32_t total_length = in_data_description->message_size;

	char* cstart = (char*)out_data_description;
	uint32_t sent = 0;

	while(sent < total_length) {
		ret = send(new_client, cstart, (total_length - sent), 0);
		if(0 > ret) {
			OMG("Error %d during send", ret);
			break;
		}
		sent += ret;
		cstart += ret;
	}

	return ret;
}

int32_t NETOutput::_disconnect(int client_id)
{
	if(0 >= num_clients || client_id >= (int)num_clients) return 0;

	pthread_mutex_lock(&server_mutex); // vvvvvvvvv

	int *c;
	int buffer[num_clients-1];
	int32_t b = 0;

	if(0 > client_id) {
		// disconnect from all clients
		c = clients;
		while(num_clients > 0) {
			_send_stop_msg(*c);
			shutdown(*c, 2);
#ifdef __windows__
			closesocket(*c);
#else
			close(*c);
#endif
			c++;
			num_clients--;
		}
		if(clients != NULL) freee(clients);
		clients = NULL;
	} else {
		// disconnect from specific client
		c = clients;
		for(int32_t i=0; i<(int32_t)num_clients; i++) {
			if(i==client_id) {
				_send_stop_msg(*c);
				shutdown(*c, 2);
#ifdef __windows__
				closesocket(*c);
#else
				close(*c);
#endif
			} else {
				buffer[b] = *c;
				b++;
			}
			c++;
		}

		num_clients--;

		if(num_clients > 0) {
			clients = (int*)realloc(clients, num_clients*sizeof(int));
			memcpy(clients, buffer, num_clients*sizeof(int));
		} else {
			freee(clients);
			clients = NULL;
		}
	}

	pthread_mutex_unlock(&server_mutex); // ^^^^^^^^

	return 0;
}

int32_t NETOutput::process(void)
{
	if(in_data_message == NULL) {
		OMG("%s: did not receive data to process", description().c_str());
		working = false;
		return -1;
	}

	out_data_message = in_data_message;

	return 0;
}

int32_t NETOutput::_send_stop_msg(int old_client)
{
	int ret = -1;
	uint32_t sent = 0;
	char* p_stop;

	EEGStopMessage* stop = NULL;
	stop = (EEGStopMessage*)malloc(sizeof(EEGStopMessage));
	if(stop == NULL) {
		OMG("Error allocating stopmessage");
		return -1;
	}

	p_stop = (char*)stop;
	stop->message_type = 3;
	stop->message_size = sizeof(EEGStopMessage);


	while(sent < stop->message_size) {
		ret = send(old_client, p_stop, (stop->message_size - sent), 0);
		if(0 > ret) {
			OMG("Error while sending stop-message");
			break;
		}
		sent += ret;
		p_stop += ret;
	}

	freee(stop);

	return ret;
}

std::string NETOutput::inputType() {
	return std::string("stream");
}

std::string NETOutput::outputType() {
	return std::string("");
}

std::string NETOutput::description()
{
	return std::string("NETOutput");
}

REGISTER_DEF_TYPE(NETOutput);

