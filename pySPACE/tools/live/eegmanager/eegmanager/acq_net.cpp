#include "acq_net.h"

// constructor
NETAcquisition::NETAcquisition()
{
	// Options this module understands
	struct option long_options[] =
	{
		{"host",	required_argument,       0, 'h'},
		{"port",	required_argument,       0, 'p'},
//		{"timeouts",	required_argument,       0, 't'},
		{0, 0, 0, 0}
	};

	merge_options(long_options, sizeof(long_options));

	// set default parameters
	std::string host = std::string(DEFAULT_IP);
	memset(server_address, 0, sizeof(server_address));
	memcpy(server_address, (char*)host.c_str(), std::min(host.length(), sizeof(server_address)));

	server_port = DEFAULT_PORT;

	// state = unconnected
	connected = 0;

	num_timeouts = 0;
	max_timeouts = 20;

}

// deconstructor
NETAcquisition::~NETAcquisition()
{

}

// parse options
int32_t NETAcquisition::setup(std::string opts)
{



	parameters = new std::string(opts);

    int option_index = 0;
    // force reset of the getopt subsystem
    optind = 0;

	int argc;
	char** argv;

	string2argv(&opts, &argc, argv);

	int c;

    // Parse Arguments
	while (1) {

        c = getopt_long(argc, argv, "h:p:", module_options, &option_index);
        if(c == -1) break;


		switch (c) {
		case 'p':
			server_port = atoi(optarg);
			break;
		case 'h':
			strcpy(server_address, optarg);
			break;
		case 't':
			max_timeouts = atoi(optarg);
			break;
		case '?':
			if (optopt == 'h' || optopt == 'p') {
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
	
	if(max_timeouts == 0 || max_timeouts == -1) {
		max_timeouts = 0xefffffff;
		FYI("set timeouts to maximum value %d", max_timeouts);
	}

    // cleanup std::string conversion
    for(int j=0; j<argc; j++) {
        freee(argv[j]);
    }
    freee(argv);
    argc = 0;

	return 0;
}


// connect to the server
int32_t NETAcquisition::_connect()
{
	int32_t ret;
	
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

	server_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (server_socket < 0)
	{
		WTF("can't create socket!");
	}

	// {send,receive}-timeout (SO_{SND,RCV}TIMEO)
	timeval timeout;
	timeout.tv_sec = 1;
	timeout.tv_usec = 0;
	setsockopt(server_socket, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout));

	//Set Portnumber and TCP/IP parameter
	struct sockaddr_in addr;
	memset((char*)&addr, 0, sizeof(sockaddr_in));

	addr.sin_family = AF_INET;
	addr.sin_port = htons(server_port);
	addr.sin_addr.s_addr = inet_addr(server_address);

	//Connect the client to the server
	while(true) {
		ret = connect(server_socket, (struct sockaddr *)&addr, sizeof(struct sockaddr_in));
		if (ret == -1){
#ifdef __windows__
			closesocket(server_socket);
#else
			close(server_socket);
#endif
			return ret;
		} else if (ret < 0) {
#ifdef __windows__
			closesocket(server_socket);
#else
			close(server_socket);
#endif
			return ret;
		} else {
			break;
		}
	}

	connected = 1;

	return 0;
}

// the main run-loop
void NETAcquisition::run(void)
{
	int32_t ret;
	working = true;

	// connect
	while(connected==0 && working) {
		_connect();
		msleep(1);
	}

	ret = getMessage(1);

	// Handle receive Errors:
	if (ret == 3) {
		FYI("Server has closed the connection.");
		goto error;
	}
	if (ret < 0) {
		FYI("An error occured during message receiving.");
		_disconnect();
		goto error;
	}

	// Only allow Startmessage
	if(in_data_description == NULL) {
		FYI("Server sent no startmessage!");
		_disconnect();
		goto error;
	}

	// Only allow Startmessage
	if(in_data_description->message_type != 1) {
		FYI("Server sent no startmessage but message of type %d!", in_data_description->message_type);
		_disconnect();
		goto error;
	}

	connected = 1;
	FYI("connected to server %s", server_address);

	// we do not change any data properties
	out_data_description = in_data_description;

	// setup {int,out}_data_description
	if(0 > putMessage((MessageHeader*)out_data_description)){
		WTF("IPC: failed sending data_description");
		goto error;
	}

	// process messages here
	while(connected==1 && working) {
		getMessage(4);

		process();

		putMessage((MessageHeader*)out_data_message);
	}

error:
	if(connected==1) {
		working = false;
		_disconnect();
	}
	connected = 0;

	return;
}

// get one message from the server
int32_t NETAcquisition::getMessage(int type)
{
	if(!working) return -1;

	int32_t ret;
	uint32_t total_length = 0;
	uint32_t current_length = 0;
	char* pData;

	// send ready message
	char readyMessage[1] = {'1'};
	ret = send(server_socket, (char*)&readyMessage, sizeof(readyMessage), 0);
	if(ret < 0) {
		OMG("Error sending readymessage");
		return ret;
	}
	if(ret != sizeof(readyMessage)) {
		FYI("readymessage partially sent: %d/%d", ret, sizeof(readyMessage));
	}

	num_timeouts = 0;

	// reserve space for header and receive it
	MessageHeader tempHeader;
	pData = (char*)&tempHeader;
	while(total_length < sizeof(MessageHeader))
	{
		current_length = sizeof(MessageHeader) - total_length;
		ret = recv(server_socket, (char*)pData, current_length, 0);
		if(!working || num_timeouts  > max_timeouts) {
			if(num_timeouts  > max_timeouts) OMG("Stopping due to too many (%d) timeouts!", num_timeouts);
			// inject stop-message
			tempHeader.message_type = 3;
			tempHeader.message_size = sizeof(MessageHeader);
			total_length = sizeof(MessageHeader);
		}

		if(ret < 0) {
			if(errno == EWOULDBLOCK) {
				num_timeouts += connected;
				continue;
			}
			OMG("%s: An Error occured while reading from socket %d", description().c_str(), server_socket);
			return ret;
		}

		total_length += ret;
		pData += ret;
	}

	// precautious conversion of byte-order (only on mb)
	if(_e32((int32_t)1) != (int32_t)1) {
		tempHeader.message_size = _e32(tempHeader.message_size);
		tempHeader.message_type = _e32(tempHeader.message_type);
	}

	switch(tempHeader.message_type) {
	case 1: // start-message
		if(in_data_description == NULL) {
			in_data_description = (EEGStartMessage*)malloc(tempHeader.message_size);
			if(in_data_description == NULL) {
				WTF("could not allocate memory for in_data_description!");
				working = false;
				return 0;
			}
			FYI("receiving msg of type %d with size %d", tempHeader.message_type, tempHeader.message_size);
		}
		memcpy(in_data_description, &tempHeader, sizeof(MessageHeader));
		pData = (char*)in_data_description;
		break;
	case 2: // normal-message
		WTF("receiving of normal eeg-stream messages not implemented!");
		break;
	case 3: // stop-message
	case 4: // mux-message
		if(in_data_message == NULL) {
			in_data_message = (EEGMuxDataMarkerMessage*)malloc(tempHeader.message_size);
			if(in_data_message == NULL) {
				WTF("could not allocate memory for in_data_message!");
				working = false;
				return 0;
			}
		}

		memcpy((char*)in_data_message, (char*)&tempHeader, sizeof(MessageHeader));
		pData = (char*)in_data_message;
		break;
	default:
		OMG("message of type %d not recognized", tempHeader.message_type);
		return -1;
	}

	pData += sizeof(MessageHeader);

	// receive data
	while(total_length < tempHeader.message_size)
	{
		current_length = tempHeader.message_size - total_length;
		ret = recv(server_socket, (char*)pData, current_length, 0);

		if(!working || num_timeouts  > max_timeouts) {
			if(num_timeouts  > max_timeouts) OMG("Stopping due to too many (%d) timeouts!", num_timeouts);
			// inject stop-message
			in_data_message->message_type = 3;
			in_data_message->message_size = tempHeader.message_size;
			total_length = tempHeader.message_size;
		}

		if(ret < 0) {
			if(errno == EWOULDBLOCK) {
				num_timeouts += connected;
				continue;
			}
			OMG("%s: An Error occured while reading from socket %d", description().c_str(), server_socket);
			return ret;
		}

		total_length += ret;
		pData += ret;
//		FYI("%d/%d", total_length, tempHeader.message_size);
	}

	// if we are on microblaze
	if(_e32((int32_t)1) != (int32_t)1) {
		switch(tempHeader.message_type) {
		case 1:
			endianflip((MessageHeader**)&(in_data_description), TO_MICROBLAZE);
			break;
		case 2:
			OMG("normal messages not implemented!");
			break;
		case 4:
			endianflip((MessageHeader**)&(in_data_message), TO_MICROBLAZE);
			break;
		case 3:
			// stopmessage
			break;
		default:
			WTF("message type (%d) not recognized", tempHeader.message_size);
			break;
		}
	}

	if(abs_start_time == 0 && tempHeader.message_type == 1) {
		abs_start_time = in_data_description->abs_start_time[0];
        abs_start_time = abs_start_time << 32;
        abs_start_time += in_data_description->abs_start_time[1];
		printf("%s: set start time to %llu\n", description().c_str(), abs_start_time);
	}

	return tempHeader.message_type;
}

// disconnect from the server
int32_t NETAcquisition::_disconnect()
{
	char stopMessage[1] = {'0'};
	send(server_socket, (char*)&stopMessage, sizeof(stopMessage), 0);
#ifdef __windows__
	closesocket(server_socket);
#else
	close(server_socket);
#endif
	connected = 0;
	return 0;
}

std::string NETAcquisition::inputType() {
	return std::string("");
}

std::string NETAcquisition::outputType() {
	return std::string("stream");
}

std::string NETAcquisition::description()
{
	return std::string("NETAcquisition");
}

REGISTER_DEF_TYPE(NETAcquisition);
