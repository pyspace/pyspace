#include "out_file.h"

	// Constants for writing header- and markerfiles
	const char* markerfiletemplate = "Brain Vision Data Exchange Marker File, Version 1.0"
			"\n;Recorded with EEGmanager built on %s, %s from"
			"\n;Git-Revision %s\n\n"
			"\n[Common Infos]"
			"\nCodepage=UTF-8"
			"\nDataFile=%s"
			"\n\n[Marker Infos]"
			"\n; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,"
			"\n; <Size in data points>, <Channel number (0 = marker is related to all channels)>,"
			"\n; <Date (YYYYMMDDhhmmssuuuuuu)>"
			"\n; Fields are delimited by commas, some fields might be omited (empty)."
			"\n; Commas in type or description text are coded as \"\\1\"."
			"\nMk1=New Segment,,0,1,0,%04d%02d%02d%02d%02d%02d0000\n";

	const char* headerfiletemplate = "Brain Vision Data Exchange Header File Version 1.0"
			"\n;Recorded with EEGmanager built on %s, %s from"
			"\n;Git-Revision %s\n\n"
			"\n[Common Infos]"
			"\nCodepage=UTF-8"
			"\nDataFile=%s"
			"\nMarkerFile=%s"
			"\nDataFormat=BINARY"
			"\nDataOrientation=MULTIPLEXED"
			"\nNumberOfChannels=%d"
			"\nSamplingInterval=%d\n"
			"\n[Binary Infos]"
			"\nBinaryFormat=%s\n"
			"\n[Channel Infos]\n";

FILEOutput::FILEOutput()
{

	// Options this module understands
	struct option long_options[] =
	{
		{"online",	no_argument,      &online, 1},
		{"subject",	required_argument,       0, 's'},
		{"dir", 	required_argument,       0, 'd'},
		{"filename",	required_argument,       0, 'f'},
		{"trial",	required_argument,       0, 't'},
		{"meta", no_argument, &meta, 1}, // argument is used by the GUI to provide useful interaction
		{0, 0, 0, 0}
	};

	merge_options(long_options, sizeof(long_options));

	//	0 = 100 nV, 1 = 500 nV, 2 = 10 µV, 3 = 152.6 µV
	nv[0] =    100.0;
	nv[1] =    500.0;
	nv[2] =  10000.0;
	nv[3] = 152600.0;

	vhdr = NULL;
	eeg = NULL;
	vmrk = NULL;

	vhdr_name = NULL;
	eeg_name = NULL;
	vmrk_name = NULL;

	set_number = 1;
	marker_number = 2;
	position = 0;
	online = 0;

	experiment = NULL;
	subject = NULL;
	basename = NULL;
	filename = NULL;
	directory = NULL;
}

int32_t FILEOutput::setup(std::string opts)
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

		c = getopt_long(argc, argv, "s:d:f:t:", module_options, &option_index);
		if(c == -1) break;

		switch (c) {
		case 'f':
			filename = new std::string(optarg, strlen(optarg));
			break;
		case 'd':
			directory = new std::string(optarg, strlen(optarg));
			break;
		case 't':
			experiment = new std::string(optarg, strlen(optarg));
			break;
		case 's':
			subject = new std::string(optarg, strlen(optarg));
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

	char buf[512];
	memset(buf, 0, 512);

	if(directory == NULL) {
		char* path = NULL;
		path = getcwd(path, 0);
		if(path != NULL) {
			FYI("data gets saved in %s", path);
			directory = new std::string(path);
			free(path);
		} else {
			WTF("Error obtaining working directory!");
			return -1;
		}
	}
	if(filename != NULL) {
		memset(buf, 0, 512);
		sprintf(buf, "%s/%s.vhdr", directory->c_str(), filename->c_str());
		vhdr = new std::ofstream(buf, std::ofstream::out);
		vhdr_name = new std::string(*filename);
		*vhdr_name += ".vhdr";

		memset(buf, 0, 512);
		sprintf(buf, "%s/%s.vmrk", directory->c_str(), filename->c_str());
		vmrk = new std::ofstream(buf, std::ifstream::out);
		vmrk_name = new std::string(*filename);
		*vmrk_name += ".vmrk";

		memset(buf, 0, 512);
		sprintf(buf, "%s/%s.eeg", directory->c_str(), filename->c_str());
		eeg = new std::ofstream(buf, std::ifstream::out | std::ifstream::binary);
		eeg_name = new std::string(*filename);
		*eeg_name += ".eeg";

		return 0;
	}
	if(experiment == NULL) {
		return -1;
	}
	if(subject == NULL) {
		return -1;
	}

	// create files instantly
	memset(buf, 0, 512);

	time_t rawTime;
	time(&rawTime);
	tm* now;
	now = localtime(&rawTime);

	sprintf(buf, "%04d%02d%02d_r_%s_%s", now->tm_year+1900, now->tm_mon+1, now->tm_mday, subject->c_str(), experiment->c_str());
	basename = new std::string(buf);

	// determine the set number according to existing header files
	bool set_number_found = false;
	while(!set_number_found) {
		// header
		memset(buf, 0, 512);
		sprintf(buf, "%s/%s_set%d.vhdr", directory->c_str(), basename->c_str(), set_number);

		if(_exists(buf)) {
			FYI("Header from set %d already there..", set_number);
			set_number++;
			continue;
		}

		set_number_found = true;
	}

	// continue to search for next set number in online case
	// according to the naming scheme (https://svn.hb.dfki.de/IMMI-Trac/wiki/vibotstudy)
	if(online) {
		set_number_found = false;

		memset(buf, 0, 512);
		delete basename;

		sprintf(buf, "%04d%02d%02d_r_%s_%s_online", now->tm_year+1900, now->tm_mon+1, now->tm_mday, subject->c_str(), experiment->c_str());
		basename = new std::string(buf);

		while(!set_number_found) {
			// header
			memset(buf, 0, 512);
			sprintf(buf, "%s/%s_set%d.vhdr", directory->c_str(), basename->c_str(), set_number);

			if(_exists(buf)) {
				// FYI("Header from set %d already there..", set_number);
				set_number++;
				continue;
			}

			set_number_found = true;
		}
	}

	vhdr = new std::ofstream(buf, std::ofstream::out);
	vhdr_name = new std::string(*basename);
	*vhdr_name += "_set";
	memset(buf, 0, 512);
	sprintf(buf, "%d", set_number);
	*vhdr_name += buf;
	*vhdr_name += ".vhdr";

	// marker
	memset(buf, 0, 512);
	sprintf(buf, "%s/%s_set%d.vmrk", directory->c_str(), basename->c_str(), set_number);

	if(_exists(buf)) {
		WTF("File %s already exists!", buf);
		return -1;
	}
	vmrk = new std::ofstream(buf, std::ifstream::out);
	vmrk_name = new std::string(*basename);
	*vmrk_name += "_set";
	memset(buf, 0, 512);
	sprintf(buf, "%d", set_number);
	*vmrk_name += buf;
	*vmrk_name += ".vmrk";

	// data
	memset(buf, 0, 512);
	sprintf(buf, "%s/%s_set%d.eeg", directory->c_str(), basename->c_str(), set_number);

	if(_exists(buf)) {
		WTF("File %s already exists!", buf);
		return -1;
	}
	eeg = new std::ofstream(buf, std::ifstream::out | std::ifstream::binary);
	eeg_name = new std::string(*basename);
	*eeg_name += "_set";
	memset(buf, 0, 512);
	sprintf(buf, "%d", set_number);
	*eeg_name += buf;
	*eeg_name += ".eeg";

	return 0;
}


FILEOutput::~FILEOutput()
{

	if(experiment != NULL) delete experiment;
	if(subject != NULL) delete subject;
	if(basename != NULL) delete basename;

	if(eeg_name != NULL) delete eeg_name;
	if(vhdr_name != NULL) delete vhdr_name;
	if(vmrk_name != NULL) delete vmrk_name;

	// close files
	if(eeg != NULL) {
		eeg->flush();
		eeg->close();
	}
	if(vhdr != NULL) {
		vhdr->flush();
		vhdr->close();
	}
	if(vmrk != NULL) {
		vmrk->flush();
		vmrk->close();
	}

}

void FILEOutput::run() {

	working = true;

	int ret = getMessage(1);
	if(ret != 1) {
		OMG("Error receiving IPC-Startmessage, was type %d", ret);
		working = false;
		goto error;
	}

	// we will not change anything here..
	out_data_description = in_data_description;

	// write header file
	_write_data_description();

	while(working) {
		getMessage(4);

		process();

		_write_data_message((MessageHeader*)out_data_message);
	}

error:

	eeg->close();
	vhdr->close();
	vmrk->close();

	FYI("Done!");

}

int32_t FILEOutput::_write_data_description()
{
	const char* BUILD_DATE = __DATE__;
	const char* BUILD_TIME = __TIME__;
	extern const char* gitversion;

	char buf[4096];
	char binary_format[16];

	// write marker title
	memset(buf, 0, 4096);

	time_t rawTime;
	time(&rawTime);
	tm* now;
	now = localtime(&rawTime);

	sprintf(buf, markerfiletemplate,
			BUILD_DATE, BUILD_TIME,
			gitversion,
			eeg_name->c_str(),
			now->tm_year+1900, now->tm_mon+1,
			now->tm_mday, now->tm_hour,
			now->tm_min, now->tm_sec);
	vmrk->write(buf, strlen(buf));

	// binary format string
	memset(binary_format, 0, 16);
	switch(out_data_description->sample_size) {
	case 1:
		sprintf(binary_format, "INT_8");
		break;
	case 2:
		sprintf(binary_format, "INT_16");
		break;
	case 4:
		sprintf(binary_format, "INT_32");
		break;
	default:
		OMG("cannot make sense of sample_size %d", out_data_description->sample_size);
		sprintf(binary_format, "UNKNWON");
		break;
	}

	// write header title
	memset(buf, 0, 4096);

	sprintf(buf, headerfiletemplate,
			BUILD_DATE, BUILD_TIME,
			gitversion,
			eeg_name->c_str(),
			vmrk_name->c_str(),
			out_data_description->n_variables-1,
			1000000/out_data_description->frequency,
			binary_format);

	vhdr->write(buf, strlen(buf));

	// write channel information
	char* name = out_data_description->variable_names;
	for(int ch=0; ch<out_data_description->n_variables-1; ch++) {
		memset(buf, 0, 4096);
		sprintf(buf, "Ch%d=%s,,%.1f,%cV\n", ch+1, name, nv[out_data_description->resolutions[ch]]/1000, 181);
		vhdr->write(buf, strlen(buf));
		while(*name != '\0') name++;
		name++;
	}

	return 0;
}

int32_t FILEOutput::_write_data_message(MessageHeader* pHeader)
{
	EEGStartMessage* idd = in_data_description;
	EEGStartMessage* odd = out_data_description;
	EEGMuxDataMarkerMessage* idm = (EEGMuxDataMarkerMessage*)in_data_message;
	EEGMuxDataMarkerMessage* odm = (EEGMuxDataMarkerMessage*)out_data_message;

	if(pHeader->message_type == 3) {
		stop();
		return 0;
	}

	uint16_t *d16;
	uint32_t *d32;
	char buf[512];

	int blocksize = (odd->n_variables-1)*odd->sample_size;
	uint32_t mrk;

	char* data = odm->data;
	for(uint32_t o=0; o<odd->n_observations; o++) {
		// write data
		eeg->write(data, blocksize);
		data += blocksize;

		switch(odd->sample_size) {
		case 2:
			d16 = (uint16_t*)data;
			mrk = (uint32_t)*d16;
			break;
		case 4:
			d32 = (uint32_t*)data;
			mrk = (uint32_t)*d32;
			break;
		default:
			OMG("Sample Size %d is unknown - cannot read marker!", odd->sample_size);
			mrk = 0;
		}

		// write S marker
		if((mrk&0xff) != 0) {
			memset(buf, 0, 512);
			sprintf(buf, "Mk%d=Stimulus,S%3d,%d,1,0\n", marker_number, (mrk&0xff), position);
			vmrk->write(buf, strlen(buf));
			marker_number++;
		}
		// write R marker
		if(((mrk>>8)&0xff) != 0) {
			memset(buf, 0, 512);
			sprintf(buf, "Mk%d=Response,R%3d,%d,1,0\n", marker_number, ((mrk>>8)&0xff), position);
			vmrk->write(buf, strlen(buf));
			marker_number++;
		}
		data += odd->sample_size;
		position++;
	}

	return 0;
}

std::string FILEOutput::inputType() {
	return std::string("stream");
}

std::string FILEOutput::outputType() {
	return std::string("");
}

std::string FILEOutput::description() {
	return std::string("FILEOutput");
}

REGISTER_DEF_TYPE(FILEOutput);
