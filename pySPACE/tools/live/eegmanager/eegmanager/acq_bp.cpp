#include "acq_bp.h"

const char* BPAcquisition::channelnames[128] = {"Fp1", "Fp2", "F7", "F3",
										"Fz", "F4", "F8", "FC5",
										"FC1", "FC2", "FC6", "T7",
										"C3", "Cz", "C4", "T8",
										"TP9", "CP5", "CP1", "CP2",
										"CP6", "TP10", "P7", "P3",
										"Pz", "P4", "P8", "PO9",
										"O1", "Oz", "O2", "PO10",
										"AF7", "AF3", "AF4", "AF8",
										"F5", "F1", "F2", "F6",
										"FT9", "FT7", "FC3", "FC4",
										"FT8", "FT10", "C5", "C1",
										"C2", "C6", "TP7", "CP3",
										"CPz", "CP4", "TP8", "P5",
										"P1", "P2", "P6", "PO7",
										"PO3", "POz", "PO4", "PO8",
										"Fpz","F9","AFF5h","AFF1h",
										"AFF2h","AFF6h","F10","FTT9h",
										"FTT7h","FCC5h","FCC3h","FCC1h",
										"FCC2h","FCC4h","FCC6h","FTT8h",
										"FTT10h","TPP9h","TPP7h","CPP5h",
										"CPP3h","CPP1h","CPP2h","CPP4h",
										"CPP6h","TPP8h","TPP10h","POO9h",
										"POO1","POO2","POO10h","Iz",
										"AFp1","AFp2","FFT9h","FFT7h",
										"FFC5h","FFC3h","FFC1h","FFC2h",
										"FFC4h","FFC6h","FFT8h","FFT10h",
										"TTP7h","CCP5h","CCP3h","CCP1h",
										"CCP2h","CCP4h","CCP6h","TTP8h",
										"P9","PPO9h","PPO5h","PPO1h",
										"PPO2h","PPO6h","PPO10h","P10",
										"I1","OI1h","OI2h","I2"};

BPAcquisition::BPAcquisition()
{

	// Options this module understands
	struct option long_options[6] =
	{
		{"calibrate",	no_argument,      &calibration, 1},
//		{"devicename",	required_argument,   0, 'd'},
		{"blocksize",	required_argument,   0, 'b'},
		{"resolution",	required_argument,   0, 'r'},
		{0, 0, 0, 0}
	};


	merge_options(long_options, sizeof(long_options));

	working = false;

	calibration = 0;

	blocksize = 0;
	resolution = 0;
	position = 0;

	driverversion = 0;
	bp_device = 0L;

	number_of_amps = 4;
	amplifiers[0] = eNoAmp;
	amplifiers[1] = eNoAmp;
	amplifiers[2] = eNoAmp;
	amplifiers[3] = eNoAmp;

	errorCode = 0;
	countCorrect = 0;
	countNotCorrect = 0;

	lastmarker = 0;

	devicename = NULL;
	
#ifndef __windows__
	bua = NULL;
#endif
}


BPAcquisition::~BPAcquisition()
{
	if(devicename != NULL) delete devicename;
	_stop_acquisition();
#ifndef __windows__
	if(bua != NULL) delete bua;
#endif
}

int32_t BPAcquisition::setup(std::string opts)
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

		c = getopt_long(argc, argv, "b:r:", module_options, &option_index);
		if(c == -1) break;


		switch (c) {
		case 'b':
			blocksize = atoi(optarg);
			break;
		case 'r':
			resolution = atoi(optarg);
			break;
		case '?':
			if (optopt == 'd' || optopt == 'b' || optopt == 'r') {
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


	if(devicename == NULL) {
		devicename = new std::string(DEVICE_USB);
	}
	if(blocksize == 0) {
		blocksize = 100;
	}
	if(resolution == 0) {
		// keep it at zero (100 nV)
	}
	if(calibration) {
		FYI("Calibration mode");
	}

	// check for available adapters
#ifdef __windows__
	bp_device = INVALID_HANDLE_VALUE;	// Amplifier device
	std::wstring wdevicename = std::wstring(devicename->begin(), devicename->end());
	bp_device = CreateFile(wdevicename.c_str(),
							GENERIC_READ | GENERIC_WRITE,
							0,
							NULL,
							OPEN_EXISTING,
							FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH,
							NULL);

	if (bp_device != INVALID_HANDLE_VALUE) {
		FYI("found %s", devicename->c_str());
	} else {
		devicename->erase();
		devicename->append(DEVICE_PCI);
		wdevicename.erase();
		wdevicename = std::wstring(devicename->begin(), devicename->end());
		bp_device = CreateFile(wdevicename.c_str(),
								GENERIC_READ | GENERIC_WRITE,
								0,
								NULL,
								OPEN_EXISTING,
								FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH,
								NULL);

		if(bp_device != INVALID_HANDLE_VALUE) {
			FYI("found %s", devicename->c_str());
		} else {
			WTF("No Brainproducts Adapter found! Error %d", GetLastError());
			return -1;
		}
	}

	if (bp_device != INVALID_HANDLE_VALUE)
	{
		DWORD dwBytesReturned;
		DeviceIoControl(bp_device,
							IOCTL_BA_DRIVERVERSION,
							NULL,
							0,
							&driverversion,
							sizeof(driverversion),
							&dwBytesReturned,
							NULL);

		unsigned nModule = driverversion % 10000;
		unsigned nMinor = (driverversion % 1000000) / 10000;
		unsigned nMajor = driverversion / 1000000;

		FYI("DriverVersion: %d.%d.%d", nMajor, nMinor, nModule);

	}
#endif
	return 0;
}


int32_t BPAcquisition::getMessage(uint32_t type)
{
	int32_t ret;

	if(type == 1) {
		ret = _get_data_description();
	} else {
		// allocate space for in_data_message
		if(in_data_message == NULL) {
			size_t msg_size = sizeof(EEGMuxDataMarkerMessage);
			msg_size += in_data_description->n_observations*(in_data_description->n_variables+1)*in_data_description->sample_size;
			in_data_message = (EEGMuxDataMarkerMessage*)malloc(msg_size);
			if(in_data_message == NULL) {
				WTF("could not allocate memory for in_data_message! (%u bytes)", msg_size);
				working = false;
				return 0;
			}
			memset(in_data_message, 0, msg_size);
		}
		ret = _get_data_message();
	}

	return ret;
}


int32_t BPAcquisition::_get_data_description()
{
	// stuff to construct the message
	uint32_t n_variables = 0;
	uint32_t n_observations = 0;
	uint32_t frequency = 0;
	uint32_t sample_size = 0;
	uint8_t resolutions[256];
	uint32_t variable_names_size = 0;
	char* variable_names = NULL;
	char* p_variable_names = NULL;
	uint32_t marker_names_size = 0;
	char* marker_names = NULL;
	char* p_marker_names = NULL;

	int32_t ret = 0;
	uint16_t pullup = 0;
	char mrk_name_buffer[5];

	// check for connected amplifiers
	uint16_t amps[4];
#ifdef __windows__
	DWORD dwBytesReturned;
	DeviceIoControl(bp_device,
					IOCTL_BA_AMPLIFIER_TYPE,
					NULL,
					0,
					amps,
					sizeof(amps),
					&dwBytesReturned,
					NULL);
#endif

	// determine type of found amplifiers
	for (int32_t i = 0; i < 4; i++)
	{
		amplifiers[i] = (AmpType)amps[i];
		if (amplifiers[i] == eNoAmp && i < number_of_amps) {
			number_of_amps = i;
		}
	}

	if(number_of_amps == 0) {
		WTF("Amp(s) not connected or switched off!");
		working = false;
		return -1;
	}

	n_variables = number_of_amps*32;
	frequency = 5000; 				// HW-sampling-frequency is fixed
	sample_size = sizeof(int16_t); 	// HW-sampling-resolution is fixed
	n_observations = blocksize;
	for(int32_t i=0; i<256; i++) {
		if(i < n_variables) resolutions[i] = resolution;
		else resolutions[i] = 0;
	}

	// fill the setup structure and send it to the device
	memset(&bp_setup, 0, sizeof(bp_setup));
	bp_setup.nHoldValue = 0x0;
	bp_setup.nLowImpedance = 1;
	bp_setup.nPoints = n_observations;
	bp_setup.nChannels = n_variables;
	for(int32_t i=0; i<bp_setup.nChannels; i++) {
		bp_setup.nDCCoupling[i] = 0;
		bp_setup.nChannelList[i] = i; // TODO: apply workspace here?
		bp_setup.n250Hertz[i] = 0; // Set HW-LP-Filter (1=250Hz or 0=1000Hz)
		bp_setup.nResolution[i] = resolution;
	}

#ifdef __windows__
	// bp_setup
	if (!DeviceIoControl(bp_device, IOCTL_BA_SETUP, &bp_setup, sizeof(bp_setup), NULL, 0, (DWORD*)&ret, NULL)) {
		WTF("Setup failed, Error code: %u\n", errno);
		return -1;
	}
	// Pulldown input resistors for trigger input, (active high)
	if (!DeviceIoControl(bp_device, IOCTL_BA_DIGITALINPUT_PULL_UP, &pullup, sizeof(pullup), NULL, 0, (DWORD*)&ret, NULL)) {
		WTF("Cannot set pulldown resistors, Error code: %u\n", errno);
		return -1;
	}
#endif


	// how many channel-names do we have to include?
    for(uint32_t i=0; i<n_variables; i++) {
    	variable_names = (char*)realloc(variable_names, variable_names_size+strlen(channelnames[i])+1);
		p_variable_names = variable_names + variable_names_size;
		strcpy(p_variable_names, channelnames[i]);
		p_variable_names += strlen(channelnames[i]);
		*p_variable_names = '\0';
    	variable_names_size += strlen(channelnames[i])+1;
    }
    // muxed messages
	char mrk_chan_name[] = "marker";
	variable_names = (char*)realloc(variable_names, variable_names_size+strlen(mrk_chan_name)+1);
	p_variable_names = variable_names + variable_names_size;
	strcpy(p_variable_names, mrk_chan_name);
	p_variable_names += strlen(mrk_chan_name);
	*p_variable_names = '\0';
	variable_names_size += strlen(mrk_chan_name)+1;
	// resulting message will have one more channel
	n_variables += 1;

    // include all marker-names (S{1..255} and R{1..255})
    for(int32_t i=1; i<256; i++) {
    	memset(mrk_name_buffer, 0, 5);
    	sprintf(mrk_name_buffer, "S%3d", i);
    	marker_names = (char*)realloc(marker_names, marker_names_size+strlen(mrk_name_buffer)+1);
    	p_marker_names = marker_names + marker_names_size;
    	strcpy(p_marker_names, mrk_name_buffer);
    	p_marker_names += strlen(mrk_name_buffer);
    	*p_marker_names = '\0';
    	marker_names_size += strlen(mrk_name_buffer)+1;
    }
    for(int32_t i=1; i<256; i++) {
		memset(mrk_name_buffer, 0, 5);
		sprintf(mrk_name_buffer, "R%3d", i);
		marker_names = (char*)realloc(marker_names, marker_names_size+strlen(mrk_name_buffer)+1);
		p_marker_names = marker_names + marker_names_size;
		strcpy(p_marker_names, mrk_name_buffer);
		p_marker_names += strlen(mrk_name_buffer);
		*p_marker_names = '\0';
		marker_names_size += strlen(mrk_name_buffer)+1;
	}

    // put it all together
    in_data_description = (EEGStartMessage*)malloc(sizeof(EEGStartMessage) + (marker_names_size + variable_names_size)*sizeof(char));
    if(in_data_description == NULL) {
		WTF("could not allocate memory for in_data_description (%u Bytes)", sizeof(EEGStartMessage) + marker_names_size + variable_names_size);
		return -1;
    }

    // fill content
    in_data_description->message_type = 1;
    in_data_description->message_size = sizeof(EEGStartMessage) + marker_names_size + variable_names_size;
    in_data_description->n_variables = n_variables;
    in_data_description->n_observations = n_observations;
    in_data_description->frequency = frequency;
    in_data_description->sample_size = sample_size;
    in_data_description->protocol_version = 2;
    memcpy(in_data_description->resolutions, resolutions, 256*sizeof(uint8_t));
    in_data_description->variable_names_size = variable_names_size;
	p_variable_names = (char*)(in_data_description->variable_names);
	memcpy(p_variable_names, variable_names, variable_names_size);
	p_marker_names = p_variable_names + variable_names_size;
	memcpy(p_marker_names, &marker_names_size, sizeof(int32_t));
	p_marker_names += sizeof(int32_t);
	memcpy(p_marker_names, marker_names, marker_names_size);

	//Calibration Settings:
	// nWaveForm -> 0 = ramp, 1 = triangle, 2 = square, 3 = sine wave
	// nFrequency -> frequency in mHz (10^-3Hz)
	bp_calibration.nWaveForm = 3;
	bp_calibration.nFrequency = 10000;

	// start acquisition
	int32_t acquisitiontype = 0;
#ifdef __windows__
	if(calibration) {
		acquisitiontype = 2;
		if (!DeviceIoControl(bp_device,
				IOCTL_BA_CALIBRATION_SETTINGS,
				&bp_calibration,
				sizeof(bp_calibration),
				NULL,
				0,
				(DWORD*)&dwBytesReturned,
				NULL))
		{
			WTF("Calibration sending failed, error-code: %u\n", errno);
			return -1;
		}
	} else {
		acquisitiontype = 1;
	}

	if (!DeviceIoControl(bp_device,
			IOCTL_BA_START,
			&acquisitiontype,
			sizeof(acquisitiontype),
			NULL,
			0,
			(DWORD*)&dwBytesReturned,
			NULL))
	{
		WTF("Start failed, error-code: %u\n", errno);
		return -1;
	}

#endif
	return 1;
}


int32_t BPAcquisition::_get_data_message()
{
	EEGStartMessage* idd = in_data_description;
	EEGMuxDataMarkerMessage* idm = (EEGMuxDataMarkerMessage*)in_data_message;

	char* p_data = idm->data;
	lasttimestamp = idm->time_code;
	uint32_t transfersize = idd->n_variables*idd->n_observations*idd->sample_size;

	int32_t dwBytesReturned = 0;
	while(true)
	{

#ifdef __windows__
		if (!ReadFile(bp_device, p_data, transfersize, (DWORD*)&dwBytesReturned, NULL)) {
			errorCode = GetLastError();
			if(errorCode == 234){ // the 'more data available' error
				countNotCorrect++;
				if (!ReadFile(bp_device, p_data, transfersize, (DWORD*)&dwBytesReturned, NULL)) {
					WTF("Acquisition error, error-code: %d\n", errno);
				}
			}else{
				WTF("Acquisition error, error-code: %d\n", errorCode);
			}
			return -1;
		}else{
			countCorrect++;
		}
		if (dwBytesReturned == 0) {
			msleep(0);
			continue;
		}
#endif // __windows__
		break;
	}

	if(dwBytesReturned == 0) return -1;

	// set abs start time (currently not used here)
	if(abs_start_time == 0) {
		abs_start_time = (uint64_t)getTime();
		return 0;
	}

	lasttime = getTime();

	uint16_t* marker = (uint16_t*)idm->data;
	marker += idd->n_variables-1;
	uint16_t marker_temp;
	for(int i=0; i<idd->n_observations; i++) {
		if(*marker == lastmarker) {
			*marker = (uint16_t)0x0;
		} else {
			marker_temp = *marker;
			*marker &= ~lastmarker;
			lastmarker = marker_temp;
		}
		marker += idd->n_variables;
	}

	idm->message_size = sizeof(EEGMuxDataMarkerMessage) + transfersize;
	idm->message_type = 4;
	idm->sample_size = idd->sample_size;
	
#ifdef __windows__
	idm->time_code = (uint32_t)getTimeDiff(abs_start_time, lasttime);
#endif

	position += idd->n_observations;

	return 0;
}

int32_t BPAcquisition::_stop_acquisition()
{

	int32_t ret = 0;

	// stop acquisition
#ifdef WIN32
	// something running?
	if(bp_device == NULL) return 0;
	FYI("%u/%u Failed at first attempt\n", countNotCorrect, countCorrect);
	int32_t dwBytesReturned;
	if (!DeviceIoControl(bp_device, IOCTL_BA_STOP, NULL, 0, NULL, 0, (DWORD*) &dwBytesReturned, NULL)) {
		WTF("Stop failed, error-code: %u\n", GetLastError());
		ret = GetLastError();
	}
	CloseHandle(bp_device);
	bp_device = INVALID_HANDLE_VALUE;
#endif

	// forward stopmessage
	if(in_data_message != NULL) {
		in_data_message->message_type=3;
		in_data_message->message_size=sizeof(EEGStopMessage);
	}

	return ret;
}


void BPAcquisition::run(void)
{
	int32_t ret;
	working = true;

	ret = getMessage(1);

	if(in_data_description == NULL || ret < 0) {
		WTF("failed to generate startmessage");
		goto error;
	}

	// TODO: get start time directly from "driver"
	while(abs_start_time == 0 && working) {
		msleep(0);
		getMessage(4);
	}

	in_data_description->abs_start_time = abs_start_time;

	// we do not change data properties
	out_data_description = in_data_description;

	// forward IPC message
	putMessage((MessageHeader*)out_data_description);

	while(working) {
		ret = getMessage(4);

		process();

		if(0 > putMessage((MessageHeader*)out_data_message)) break;
	}
	FYI("left run loop");

	error:
	_stop_acquisition();
	if(in_data_message != NULL) putMessage((MessageHeader*)in_data_message);

	working = false;

	return;
}

std::string BPAcquisition::inputType() {
	return std::string("");
}

std::string BPAcquisition::outputType() {
	return std::string("stream");
}

std::string BPAcquisition::description()
{
	return std::string("BPAcquisition");
}

REGISTER_DEF_TYPE(BPAcquisition);
