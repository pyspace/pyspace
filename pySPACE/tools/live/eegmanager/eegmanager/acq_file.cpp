#include "acq_file.h"



FILEAcquisition::FILEAcquisition()
{
	// Options this module understands
	struct option long_options[] =
	{
		{"filename",	required_argument,       0, 'f'},
		{"blocksize",	required_argument,   0, 'b'},
		{"meta", no_argument, &meta, 1}, // argument is used by the GUI to provide useful interaction
		{0, 0, 0, 0}
	};

	merge_options(long_options, sizeof(long_options));
    
	filename = NULL;

	vhdr = NULL;
	eeg = NULL;
	vmrk = NULL;

	vhdr_name = NULL;
	eeg_name = NULL;
	vmrk_name = NULL;

	position = 0;

	blocksize = 0;
	working = false;

	current_marker_name = std::string("");
	current_marker_position = 0;
	current_marker_value = 0;
}

FILEAcquisition::~FILEAcquisition()
{
	if(filename != NULL) delete filename;

	if(eeg_name != NULL) delete eeg_name;
	if(vhdr_name != NULL) delete vhdr_name;
	if(vmrk_name != NULL) delete vmrk_name;

	// close files
	if(eeg != NULL) eeg->close();
	if(vhdr != NULL) vhdr->close();
	if(vmrk != NULL) vmrk->close();
}

int32_t FILEAcquisition::setup(std::string opts)
{

	parameters = new std::string(opts);
	if(opts.size() == 0) return -1;

	int option_index = 0;
	// force reset of the getopt subsystem
	optind = 0;

	int argc;
	char** argv;

	string2argv(&opts, &argc, argv);

	int c = 0;

	// Parse Arguments
	while (1) {

		c = getopt_long(argc, argv, "f:b:", module_options, &option_index);
		if(c == -1) break;

		switch (c) {
		case 'f':
			filename = new std::string(optarg, strlen(optarg));
			break;
		case 'b':
			blocksize = atoi(optarg);
			break;
		case '?':
			if (optopt == 'f' || optopt == 'b') {
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
		free(argv[j]);
	}
	free(argv);
	argc = 0;

	// do we have enough options?
	if(filename == NULL) {
		WTF("No Filename provided!");
		return -1;
	}
	// switch to default blocksize
	if(blocksize == 0) {
		blocksize = 100;
	}

	// tolerate .eeg or basename parameters
	if(filename->rfind(".") != std::string::npos) {
		if(filename->substr(filename->rfind(".")) == std::string(".eeg")) {
			filename->resize(filename->rfind("."));
		}
	}

	// instant check for files
	char buf[512];

	// header
	memset(buf, 0, 512);
	sprintf(buf, "%s.vhdr", filename->c_str());

//	FYI("Now trying file %s", buf);
	fflush(stdout);

	if(!_exists(buf)) {
		WTF("Error %d when opening file %s.", errno, buf);
		return -1;
	}
	vhdr = new std::ifstream(buf, std::ifstream::in);
	vhdr_name = new std::string(buf, strlen(buf));

	// marker
	memset(buf, 0, 512);
	sprintf(buf, "%s.vmrk", filename->c_str());

//	FYI("Now trying file %s", buf);
	fflush(stdout);

	if(!_exists(buf)) {
		WTF("Error %d when opening file %s.", errno, buf);
		return -1;
	}
	vmrk = new std::ifstream(buf, std::ifstream::in);
	vmrk_name = new std::string(buf, strlen(buf));

	// data
	memset(buf, 0, 512);
	sprintf(buf, "%s.eeg", filename->c_str());

//	FYI("Now trying file %s", buf);
	fflush(stdout);

	if(!_exists(buf)) {
		OMG("Error %d when opening file %s.", errno, buf);
		memset(buf, 0, 512);
		sprintf(buf, "%s.dat", filename->c_str());
		if(!_exists(buf)) {
			WTF("Error %d when opening file %s.", errno, buf);
			return -1;
		}
	}
	eeg = new std::ifstream(buf, std::ifstream::in | std::ifstream::binary);
	eeg_name = new std::string(buf, strlen(buf));

	return 0;
}


void FILEAcquisition::run()
{

	int32_t ret;
	working = true;

	ret = getMessage(1);

	if(in_data_description == NULL) {
		WTF("failed to generate startmessage");
		return;
	}

	// we do not change data properties
	out_data_description = in_data_description;

	// forward IPC message
	putMessage((MessageHeader*)out_data_description);

	while(working) {
		getMessage(4);

		process();

		putMessage((MessageHeader*)out_data_message);
	}

	// close files
	if(eeg != NULL) eeg->close();
	if(vhdr != NULL) vhdr->close();
	if(vmrk != NULL) vmrk->close();

	return;
}


int32_t FILEAcquisition::getMessage(uint32_t type)
{
	int32_t ret;

	if(type == 1) {
		ret = _get_data_description();
	} else {
		// allocate space for in_data_message
		if(in_data_message == NULL) {
			size_t msg_size = sizeof(EEGMuxDataMarkerMessage);
			msg_size += in_data_description->n_observations*(in_data_description->n_variables)*in_data_description->sample_size;
			in_data_message = (EEGMuxDataMarkerMessage*)malloc(msg_size);
			if(in_data_message == NULL) {
				WTF("could not allocate memory for in_data_message! (%lu bytes)", msg_size);
				working = false;
				return 0;
			}
		}
		ret = _get_data_message();
	}

	return ret;
}

int32_t FILEAcquisition::_get_data_description()
{
	// stuff to construct the message
	uint32_t n_variables = 0;
	uint32_t frequency = 0;
	uint32_t sample_size = 0;
	uint8_t resolutions[256];
	uint8_t* p_res = resolutions;
	uint32_t variable_names_size = 0;
	char* variable_names = NULL;
	char* p_variable_names = NULL;
	uint32_t marker_names_size = 0;
	char* marker_names = NULL;
	char* p_marker_names = NULL;

	std::string datafile;
	std::string markerfile;

	std::string line;
	std::string name;
	std::string reso;
	std::string unit;
	std::string mrkn;
	size_t pos;

	// first: extract relevant information from header file
	while(vhdr->good()) {
		std::getline(*vhdr, line);
		// skip comments
		if(line[0] == '#' || line[0]==';' || line[0]=='[') {
			if(line.find("Comment") != std::string::npos) break;
			continue;
		}

		// datafile name
		if(line.find("DataFile") != std::string::npos) {
			pos = line.find("=");
			datafile = line.substr(pos+1, std::string::npos);
//			printf("datafile=%s\n", datafile.c_str());
		}
		// markerfile name
		if(line.find("MarkerFile") != std::string::npos) {
			pos = line.find("=");
			markerfile = line.substr(pos+1, std::string::npos);
//			printf("markerfile=%s\n", markerfile.c_str());

		}
		// format of datafile
		if(line.find("DataFormat") != std::string::npos) {
			pos = line.find("=");
			if(line.compare(pos+1, 6, "BINARY") == 0) {
//				printf("dataformat=BINARY\n");
			} else if (line.compare(pos+1, 5, "ASCII") == 0) {
				OMG("dataformat=ASCII is unsupported\n");
				return -1;
			} else {
				OMG("unsupported dataformat: %s\n", line.substr(pos+1, std::string::npos).c_str());
				return -1;
			}
		}
		// structure of datafile
		if(line.find("DataOrientation") != std::string::npos) {
			pos = line.find("=");
			if(line.compare(pos+1, 11, "MULTIPLEXED") == 0) {
//				printf("dataorientation=MULTIPLEXED\n");
			} else if (line.compare(pos+1, 10, "VECTORIZED") == 0) {
				WTF("dataorientation=VECTORIZED is unsupported\n");
				return -1;
			} else {
				WTF("unsupported dataorientation: %s\n", line.substr(pos+1, std::string::npos).c_str());
				return -1;
			}
		}
		// number of channels
		if(line.find("NumberOfChannels") != std::string::npos) {
			pos = line.find("=");
			n_variables = atoi(line.substr(pos+1, std::string::npos).c_str());
			if(n_variables == 0) {
				WTF("unsupported number of channels: %s\n", line.substr(pos+1, std::string::npos).c_str());
				return -1;
			}
		}
		// sampling interval / frequency
		if(line.find("SamplingInterval") != std::string::npos) {
			pos = line.find("=");
			int32_t interval = atoi(line.substr(pos+1, std::string::npos).c_str());
			if(interval == 0) {
				OMG("unknown sampling interval: %s\n", line.substr(pos+1, std::string::npos).c_str());
			} else {
				frequency = 1000000/interval;
			}
		}
		// sample-wise format
		if(line.find("BinaryFormat") != std::string::npos) {
			pos = line.find("=");
			if(line.compare(pos+1, 6, "INT_16") == 0) {
//				printf("binaryformat=INT_16\n");
				sample_size = sizeof(int16_t);
			} else if (line.compare(pos+1, 6, "INT_32") == 0) {
//				printf("binaryformat=INT_32\n");
				sample_size = sizeof(int32_t);
			} else if (line.compare(pos+1, 13, "IEEE_FLOAT_32") == 0) {
//				printf("binaryformat=IEEE_FLOAT_32\n");
				sample_size = sizeof(float);
			} else {
				OMG("unknown binaryformat: %s\n", line.substr(pos+1, std::string::npos).c_str());
				return -1;
			}
		}
		// channel parameters (name, resolution)
		if(line[0]=='C' && line[1]=='h') {
			pos = line.find("=");

			// channel name
			name = line.substr(pos+1);
			name = name.substr(0, name.find(","));
			variable_names = (char*)realloc(variable_names, variable_names_size+name.size()+1);
			p_variable_names = variable_names + variable_names_size;
			strcpy(p_variable_names, name.c_str());
			p_variable_names += name.size();
			*p_variable_names = '\0';
			variable_names_size += name.size()+1;

//			printf("%s\n", name.c_str());

			// channel resolution and resolution unit
			reso = line.substr(pos+1);
			reso = reso.substr(reso.find(",")+1, std::string::npos);
			reso = reso.substr(reso.find(",")+1, std::string::npos);

			unit = reso.substr(reso.find(",")+1, std::string::npos);

			double res_temp = 0.0;
			res_temp = (double)strtod(reso.substr(0,reso.find(",")).c_str(), 0);
			if(unit.compare(0, 2, "µV") == 0) res_temp = res_temp * 1000.0;
			// 0 = 100 nV, 1 = 500 nV, 2 = 10 µV, 3 = 152.6 µV
			if(res_temp == 0.0) {
				OMG("Resolution set to Default (152600 nV) on Channel %s\n", name.c_str());
				*p_res = 3;
			} else if(res_temp <= 100) {
				*p_res = 0;
			} else if(res_temp <= 500) {
				*p_res = 1;
			} else if(res_temp <= 10000) {
				*p_res = 2;
			} else if(res_temp <= 152600) {
				*p_res = 3;
			} else {
				OMG("Resolution set to Default (152600 nV) on Channel %s\n", name.c_str());
				*p_res = 3;
			}
			p_res++;
		}
	}
	// rewind vhdr file
	vhdr->clear();
	vhdr->seekg(0, std::ios::beg);


//	char* display = variable_names;
//	for(int i=0; i<variable_names_size; i++) {
//		if(*display == '\0') {
//			printf(" // ");
//		}
//		printf("%c", *display);
//		display++;
//	}
//
//	for(int i=0; i<256; i++) {
//		printf("%d, ", resolutions[i]);
//	}

	// second: extract marker-names from marker-file
	while(vmrk->good()) {
		std::getline(*vmrk, line);
		// skip comments
		if(line[0] == '#' || line[0]==';' || line[0]=='[') {
			// TODO: Comment section is not parsed
			if(line.find("Comment") != std::string::npos) break;
			continue;
		}
		// skip the new segment entry
		if(line.find("New Segment") != std::string::npos) {
			// TODO: some valuable information are coded here - extract them!
			continue;
		}

		// datafile name
		if(line.find("DataFile") != std::string::npos) {
			pos = line.find("=");
			datafile = line.substr(pos+1, line.size()-pos);
//			printf("datafile=%s\n", datafile.c_str());
		}

		// marker names
		if(line[0] == 'M' && line[1] == 'k') {
			pos = line.find(",");
			line = line.substr(pos+1, std::string::npos);
			name = line.substr(0, line.find(","));

			// check existing markers
			p_marker_names = marker_names;
			uint32_t i = 0;
			while(i < marker_names_size) {
				mrkn = std::string(p_marker_names);
				if(mrkn.compare(name) == 0) {
					// marker is already in the list
					break;
				}
				i += strlen(p_marker_names)+1;
				p_marker_names += strlen(p_marker_names)+1;
			}

			// if we read all markers and didn't found
			// the current in it we are adding  the
			// current marker to the list
			if(i >= marker_names_size) {
				marker_names = (char*)realloc(marker_names, marker_names_size+name.size()+1);
				p_marker_names = marker_names + marker_names_size;
				strcpy(p_marker_names, name.c_str());
				p_marker_names += name.size();
				*p_marker_names = '\0';
				marker_names_size += name.size()+1;
			}
		}
	}
	// rewind vmrk file
	vmrk->clear();
	vmrk->seekg(0, std::ios::beg);

	// one extra channel for markers
	name = std::string("marker");
	variable_names = (char*)realloc(variable_names, variable_names_size+name.size()+1);
	p_variable_names = variable_names + variable_names_size;
	strcpy(p_variable_names, name.c_str());
	p_variable_names += name.size();
	*p_variable_names = '\0';
	variable_names_size += name.size()+1;
	n_variables += 1;

	// StartMessage, Type = 1
	//	struct EEGStartMessage : MessageHeader {
	//		uint32_t n_variables; // = number of channels
	//		uint32_t n_observations; // = number of samples
	//		uint32_t frequency;
	//		uint32_t sample_size; 	// size of one sample (1, 2, 4, maybe even 8 Bytes)
	//		uint8_t resolutions[256]; // as defined in BrainAmpIoCtl.h
	//		uint32_t variable_names_size; //Number of character or size in byte???
	//		char variable_names[1];
	//		uint32_t marker_names_size; //Number of character or size in byte???
	//		char marker_names[1];
	//	};

	// put it all together
	in_data_description = (EEGStartMessage*)malloc(sizeof(EEGStartMessage) + marker_names_size + variable_names_size);
	if(in_data_description == NULL) {
		WTF("could not allocate memory for in_data_description (%lu Bytes)", sizeof(EEGStartMessage) + marker_names_size + variable_names_size);
		return -1;
	}

	in_data_description->message_type = 1;
	in_data_description->message_size = sizeof(EEGStartMessage) + marker_names_size + variable_names_size;
	in_data_description->n_variables = n_variables;
	in_data_description->n_observations = blocksize;
	in_data_description->frequency = frequency;
	in_data_description->sample_size = sample_size;
	in_data_description->protocol_version = 2;
	abs_start_time = getTime();
	in_data_description->abs_start_time[0] = (uint32_t)((abs_start_time&0xffffffff00000000)>>32);
    in_data_description->abs_start_time[1] = (uint32_t)(abs_start_time&0xffffffff);
	memcpy(in_data_description->resolutions, resolutions, 256*sizeof(uint8_t));
	in_data_description->variable_names_size = variable_names_size;
	p_variable_names = (char*)(in_data_description->variable_names);
	memcpy(p_variable_names, variable_names, variable_names_size);
	p_marker_names = p_variable_names + variable_names_size;
	memcpy(p_marker_names, &marker_names_size, sizeof(int32_t));
	p_marker_names += sizeof(int32_t);
	memcpy(p_marker_names, marker_names, marker_names_size);

	// cleanup
	if(variable_names != NULL) freee(variable_names);
	if(marker_names != NULL) freee(marker_names);

	return 1;
}


int32_t FILEAcquisition::_get_data_message()
{
	EEGStartMessage* idd = in_data_description;
	EEGMuxDataMarkerMessage* idm = (EEGMuxDataMarkerMessage*)in_data_message;

	size_t size = idd->n_variables*idd->n_observations*idd->sample_size;

	// data samples for one block
	int32_t n_samples = idd->n_variables-1;

	char* p_data = idm->data;
	memset(p_data, 0, size);

	uint16_t* d16;
	uint32_t* d32;

	if(eeg->eof() || !eeg->good() || !working) {
		// create stop message
		FYI("Reached EOF!");
		idm->message_size = sizeof(EEGStopMessage);
		idm->message_type = 3;
		return 0;
	}

	for(uint32_t i=0; i<idd->n_observations; i++) {
		// all channels for one sample
		eeg->read(p_data, n_samples*idd->sample_size);
		// this happens if we read exactly to the end of a file
		if(i==0 && eeg->gcount()==0) {
			// create stop message
			FYI("reached exact EOF!");
			idm->message_size = sizeof(EEGStopMessage);
			idm->message_type = 3;
			return 0;
		}
		p_data += n_samples*idd->sample_size;

		// corresponding marker or zero;
		if(idd->sample_size == 2) {
			d16 = (uint16_t*)p_data;
			*d16 = _marker_at_position(i);
		} else {
			d32 = (uint32_t*)p_data;
			*d32 = (uint32_t)_marker_at_position(i);
		}
		p_data += idd->sample_size;
	}

	// set abs start time (currently not used here)
	if(abs_start_time == 0) {
		abs_start_time = getTime();
	}

	position += idd->n_observations;

	size_t msg_size = 0;
	msg_size += sizeof(EEGMuxDataMarkerMessage);
	msg_size += idd->n_observations*idd->n_variables*idd->sample_size;
	idm->message_size = msg_size;
	idm->message_type = 4;
	idm->sample_size = idd->sample_size;
	idm->time_code = (uint32_t)((position*1000)/idd->frequency);
//	idm->time_code = (uint32_t)getTimeDiff(abs_start_time);

	return idm->message_type;
}

uint16_t FILEAcquisition::_marker_at_position(int32_t p)
{
	// file completely parsed
	if(vmrk->eof() || !vmrk->good()) {
		return 0;
	}

	std::string line;

	// position==0 -> we didn't read any data yet
	// happens only at the very first call
	if(position+p == 0) {
		while(vmrk->good()) {
			std::getline(*vmrk, line);
			if(line[0] == '#' || line[0]==';' || line[0]=='[') continue;
			if(line[0]=='M' && line[1]=='k') {
				break;
			}
		}
	}

	size_t pos;
	uint16_t mrk = 0;

	// get new marker from file
	if(current_marker_position < (uint32_t)(position + p)) {
		// extract a new one from the file
		std::getline(*vmrk, line);
		if(line.size() == 0) return 0;

		pos = line.find(",");
		line = line.substr(pos+1, std::string::npos);
		pos = line.find(",");

		current_marker_name = line.substr(0, pos).c_str();
		current_marker_value = atoi(current_marker_name.substr(1, std::string::npos).c_str());
		switch(current_marker_name[0]) {
		case 'S':
			current_marker_value = (uint16_t)(current_marker_value&0xff);
			break;
		case 'R':
			current_marker_value = (uint16_t)((current_marker_value&0xff)<<8);
			break;
		default:
			OMG("Marker type with name %s unknown", current_marker_name.c_str());
		}
		line = line.substr(pos+1, std::string::npos);
		pos = line.find(",");
		current_marker_position = atoi(line.substr(0, pos).c_str());
	}

	// if position of current marker matches put it in
	if(current_marker_position == (uint32_t)(position + p)) {
		mrk = current_marker_value;
	} else {
		// no marker
		return 0;
	}

	return mrk;
}

std::string FILEAcquisition::inputType() {
	return std::string("");
}

std::string FILEAcquisition::outputType() {
	return std::string("stream");
}

std::string FILEAcquisition::description()
{
	return std::string("FILEAcquisition");
}


REGISTER_DEF_TYPE(FILEAcquisition);


