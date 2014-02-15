#include "glob_module.h"

ModuleFactory::map_type * ModuleFactory::map = NULL;

Module::Module()
{
	working = false;

	in_data_description = NULL;
	out_data_description = NULL;
	in_data_message = NULL;
	out_data_message = NULL;

	parameters = NULL;
	

	prev = NULL;
	next = NULL;

	module_options = NULL;
	option_storage = NULL;

	abs_start_time = 0;

	meta = 0;

}

Module::~Module()
{

	if(in_data_description == out_data_description) {
		if(in_data_description != NULL) freee(in_data_description);
	} else {
		if(out_data_description != NULL) freee(out_data_description);
		if(in_data_description != NULL) freee(in_data_description);
	}
	in_data_description = NULL;
	out_data_description = NULL;

	if(in_data_message == out_data_message) {
		if(in_data_message != NULL) freee(in_data_message);
	} else {
		if(out_data_message != NULL) freee(out_data_message);
		if(in_data_message != NULL) freee(in_data_message);
	}
	in_data_message = NULL;
	out_data_message = NULL;

	prev = NULL;
	next = NULL;

	if(parameters != NULL) delete parameters;
	parameters = NULL;

	if(module_options != NULL) free(module_options);
	module_options = NULL;
	if(option_storage != NULL) free(option_storage);
	option_storage = NULL;

}

void Module::setPrev(RingBuffer* p)
{
	if(prev != NULL) {
		OMG("already attached to buffer at %16lu", (uintptr_t)prev);
	}
	prev = p;

}

void Module::setNext(RingBuffer* n)
{
	if(next != NULL) {
		OMG("already attached to buffer at %16lu", (uintptr_t)next);
	}
	next = n;
}


float Module::i_bandwith()
{
	return 23.0;
}

float Module::o_bandwith()
{
	return 42.0;
}

/*
 * This function retrieves a message of type type from shared memory
 * @return: type of message when prev was readable
 * 			< 0  when error
 * 			else the message-type
 */

int32_t Module::getMessage(uint32_t type)
{
	uint32_t total_length = 0;
	uint32_t ret = 0;

	MessageHeader header;
	memset(&header, 0, sizeof(MessageHeader));
	char* p_data = (char*)&header;

	while(total_length < sizeof(MessageHeader) && working) {
		ret = prev->read(p_data, sizeof(MessageHeader)-total_length);
		total_length += ret;
		p_data += ret;
	}

	switch(header.message_type) {
	case 1: // start-message
		if(in_data_description == NULL) {
			in_data_description = (EEGStartMessage*)malloc(header.message_size);
			if(in_data_description == NULL) {
				WTF("could not allocate memory!");
				working = false;
				return 0;
			}
		}
		memcpy(in_data_description, &header, sizeof(MessageHeader));
		p_data = (char*)in_data_description;
		p_data += sizeof(MessageHeader);
		break;
	case 2: // normal-message
		OMG("receiving of normal eeg-data-msgs not implemented");
		break;
	case 3: // stop-message
		if(in_data_message == NULL) {
			in_data_message = (EEGMuxDataMarkerMessage*)malloc(header.message_size);
			if(in_data_message == NULL) {
				WTF("could not allocate memory for in_data_message!");
				working = false;
				return 0;
			}
		}
		memcpy((char*)in_data_message, (char*)&header, sizeof(MessageHeader));
		return header.message_type;
		break;
	case 4: // mux-message
		if(in_data_message == NULL) {
			in_data_message = (EEGMuxDataMarkerMessage*)malloc(header.message_size);
			if(in_data_message == NULL) {
				WTF("could not allocate memory for in_data_message!");
				working = false;
				return 0;
			}
		}
		memcpy((char*)in_data_message, (char*)&header, sizeof(MessageHeader));
		p_data = (char*)in_data_message;
		p_data += sizeof(MessageHeader);
		break;
	default:
		OMG("%s: message of type %d not recognized", description().c_str(), header.message_type);
		working = false;
		return -1;
	}

	while(total_length < header.message_size && working) {
		ret = prev->read(p_data, header.message_size - total_length);
		p_data += ret;
		total_length += ret;
	}

	// set absolute start time
	if(abs_start_time == 0 && header.message_type == 1) {
		abs_start_time = in_data_description->abs_start_time[0];
        abs_start_time = abs_start_time << 32;
        abs_start_time += in_data_description->abs_start_time[1];
		printf("%s: set start time to %llu\n", description().c_str(), abs_start_time);
	}
	
	if(type != header.message_type) {
		FYI("requested MSGTYPE(%u) but received %u", type, header.message_type);
	}

	return header.message_type;
}

/*
 * This function is the minimum processing required
 */

int32_t Module::process(void) {
	if(in_data_message == NULL) {
		OMG("%s: did not receive data to process", description().c_str());
		working = false;
		return -1;
	}

	out_data_message = in_data_message;

	return 0;
}

/*
 * This functions pushes a message along the shared memory
 * returns ret < 0 when error, else number of bytes sent
 */

int32_t Module::putMessage(MessageHeader* header)
{
	uint32_t total_length = 0;
	int32_t ret = 0;

	if(header == NULL) {
		OMG("%s: no message provided to send!", description().c_str());
		return -1;
	}

	char* p_data = (char*)header;

	while(total_length < header->message_size && working) {
		ret = next->write(p_data, header->message_size-total_length);
		total_length += ret;
		p_data += ret;
	}

	if(header->message_type == 3) {
		stop();
	}
	// invalidate current message
	header->message_type = 3;

	return total_length;
}

void Module::stop() {
	if(working) {
		working = false;
		FYI("%s: working = false", description().c_str());
	}
}

int32_t Module::block_length_ms_out()
{
	if(out_data_description == NULL) return -1;
	return (out_data_description->n_observations*1000)/out_data_description->frequency;
}

int32_t Module::block_length_ms_in()
{
	if(in_data_description == NULL) return -1;
	return (in_data_description->n_observations*1000)/in_data_description->frequency;
}

int32_t Module::block_length_ms()
{
	if(in_data_description != out_data_description) return -1;
	return block_length_ms_out();
}

void Module::string2argv(std::string* opts, int* argc, char** &argv)
{
	// convert std::string to argc & argv
    std::istringstream iss(*opts);
    std::vector<std::string> tokens;
    tokens.push_back("/bin/eegmanager"); // dummy token
    copy(std::istream_iterator<std::string>(iss),
    		std::istream_iterator<std::string>(),
    		std::back_inserter< std::vector<std::string> >(tokens));

    int i=0;

    *argc = tokens.size();
    argv = (char**)malloc(tokens.size()*sizeof(char*));

    for (std::vector<std::string>::iterator it = tokens.begin(); it!=tokens.end(); ++it) {
        argv[i] = (char*)malloc(it->size());
        strcpy(argv[i++], it->c_str());
    }

    // tokens is on the stack - it will be destroyed anyway
    // just cleanup a bit
    tokens.clear();

    return;
}


void Module::endianflip(MessageHeader** header, int dir) {
	// convert data to big-endian format
	// note: message_type and message_size have to be converted before!

	EEGStartMessage* s = (EEGStartMessage*)*header;
	EEGMuxDataMarkerMessage* d = (EEGMuxDataMarkerMessage*)*header;
	int16_t* d16;
	int32_t* d32;

	char* temp;
	uint32_t* marker_names_size;

	switch(s->message_type) {
	case 1: // startmessage
		s->n_variables = _e32(s->n_variables);
		s->n_observations = _e32(s->n_observations);
		s->frequency = _e32(s->frequency);
		s->sample_size = _e32(s->sample_size);
//		s->abs_start_time = _e64(s->abs_start_time);
		s->protocol_version = _e32(s->protocol_version);
//		uint8_t resolutions[256]; // as defined in BrainAmpIoCtl.h
		if(dir == TO_MICROBLAZE) s->variable_names_size = _e32(s->variable_names_size);
//		char variable_names[1];
		temp = s->variable_names;
		temp += s->variable_names_size;
		marker_names_size = (uint32_t*)temp;
		*marker_names_size = _e32(*marker_names_size);
//		char marker_names[1];
		if(dir == TO_NETWORK) s->variable_names_size = _e32(s->variable_names_size);
		break;

	case 2: // datamessage
		OMG("processing of normal eegstream messages not implemented!");
		break;

	case 3: // stopmessage
		// nothing to do here
		break;

	case 4: // muxed-datamessage
		if(dir == TO_MICROBLAZE) {
			d->time_code = _e32(d->time_code);
			d->sample_size = _e32(d->sample_size);
		}

		// nothing to do if we only got characters
		if(d->sample_size == 1) break;

		d16 = (int16_t*)d->data;
		d32 = (int32_t*)d->data;

		if(d->sample_size == 2) {
			for(uint32_t i=0; i<in_data_description->n_observations; i++) {
				for(uint32_t j=0; j<in_data_description->n_variables; j++) {
					*d16 = _e16(*d16);
					d16++;
				}
			}
		} else if(d->sample_size == 4) {
			for(uint32_t i=0; i<in_data_description->n_observations; i++) {
				for(uint32_t j=0; j<in_data_description->n_variables; j++) {
					*d32 = _e32(*d32);
					d32++;
				}
			}
		} else {
			WTF("cannot work with a sample-size of %d!", d->sample_size);
		}

		if(dir == TO_NETWORK) {
			d->time_code = _e32(d->time_code);
			d->sample_size = _e32(d->sample_size);
		}

		break;

	default:
		WTF("Message-Type (%d) not understood", (*header)->message_type);
		break;
	}
}

void Module::datadescription2text(MessageHeader* dd)
{
	EEGStartMessage* start = (EEGStartMessage*)dd;
	printf("-- dump of %s's data_description --\n", description().c_str());
	printf("type                 %6d\n", dd->message_type);
	printf("size                 %6d\n", dd->message_size);
	printf("n_variables          %6d\n", start->n_variables);
	printf("n_observations       %6d\n", start->n_observations);
	printf("frequency            %6d\n", start->frequency);
	printf("sample_size          %6d\n", start->sample_size);
	printf("protocol_version     %6d\n", start->protocol_version);
	printf("abs_start_time       %6lu\n", start->abs_start_time);
	printf("resolutions          ");
	for(unsigned long i=0; i<sizeof(start->resolutions); i++) {
		printf("%2x ", start->resolutions[i]);
	}
	printf("\n");
	printf("variable_names_size  %6d\n", start->variable_names_size);
	printf("variable_names       ");
	char* worker = start->variable_names;
	for(uint32_t i=0; i<start->variable_names_size; i++) {
		if(*worker == '\0') printf(" ");
		else printf("%c", *worker);
		worker++;
	}
	printf("\n");
	worker = start->variable_names;
	worker += start->variable_names_size;
	uint32_t* marker_names_size = (uint32_t*)worker;
	printf("marker_names_size    %6d\n", *marker_names_size);
	printf("marker_names         ");
	worker += sizeof(uint32_t);
	for(uint32_t i=0; i<*marker_names_size; i++) {
		if(*worker == '\0') printf(" ");
		else printf("%c", *worker);
		worker++;
	}
	printf("\n");
	printf ("-- -  -   -    -     -      -       -        -\n");
}

// this methods incorporates the data pointed to by
// long_options into the local memory
void Module::merge_options(option* long_options, size_t size)
{
	option* o = long_options;
	char* option_storage_;
	size_t string_size = 0;

    while(o->name != 0) {
        string_size += strlen(o->name)+1;
        o++;
    }
    // allocate space for strings pointed to by
    // option struct fields
    option_storage = (char*)malloc(string_size);
    memset((char*)option_storage, 0, string_size);
    option_storage_ = option_storage;

    // allocate space for option structs
	module_options = (option*)malloc(size);
	memset((char*)module_options, 0, size);

    // copy options structs
    memcpy(module_options, long_options, size);

    o = module_options;
    while(o->name != 0) {
    	// copy strings
    	string_size = strlen(o->name)+1;
    	fflush(stdout);
        sprintf(option_storage_, "%s", o->name);
        // re-point pointer to local memory
        o->name = option_storage_;
        option_storage_ += string_size;
        o++;
    }

	return;
}

// is this module a input module?
// e.g. generates data or does acquisition?
bool Module::isInput()
{
	return inputType().empty();
}

// is this module a output module?
// e.g. consumes data?
bool Module::isOutput()
{
	return outputType().empty();
}

bool Module::isStream()
{
	return (!outputType().empty()) && (!inputType().empty());
}

// construct a 'usage' string
std::string Module::usage()
{
    struct option* o;
    o = module_options;

    if(o == NULL) return std::string("");
    if(o->name == NULL) return std::string("");

    char buf[1024];
    char* bf;
    memset(buf, 0, 1024);

    while(o->name != NULL) {
        bf = buf+strlen(buf);
        if(o->has_arg) {
        	sprintf(bf, "[-%c|--%s] <%s> ", o->val, o->name, o->name);
        } else {
        	sprintf(bf, "--%s ", o->name);
        }
        o++;
    }

	return std::string(buf);
}

// returns the parameter of the configure module
// (currently unused)
std::string Module::getParameters()
{
	if(parameters == NULL) return std::string("");
	return *parameters;
}
