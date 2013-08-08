#ifndef __ACQ_BP_H__
#define __ACQ_BP_H__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <fstream>

#ifndef __windows__
#include <sys/ioctl.h>
#include "util_bua.h"
#endif

#include "global.h"
#include "EEGStream.h"
#include "glob_module.h"
#include "util_thread.h"
#include "utils.h"
#include "BrainAmpIoCtl.h"

#ifdef __windows__
#define DEVICE_PCI	"\\\\.\\BrainAmp"
#define DEVICE_USB	"\\\\.\\BrainAmpUSB1"
#else
#define DEVICE_USB 	"/dev/usb/brainamp_usb0" // to be replaced..
#define BUA_VENDOR	0x1103
#define BUA_DEVICE	0x0001
#endif

// Different amplifier types
//enum AmpTypes {
//	None = 0, Standard = 1, MR = 2, DCMRplus = 3
//};

// Number of ELements
#define NEL(x)  (sizeof(x) / sizeof(x[0]))

// The Hardware Samples with 5000Hz
const int32_t g_deviceSamplingFrequency = 5000;

class BPAcquisition : public Module
{
public:
	BPAcquisition();
	~BPAcquisition();

	std::string inputType();
	std::string outputType();
	std::string description();

	int32_t setup(std::string opts);

	int32_t getMessage(uint32_t type);
	
	void run();

private:

	int32_t _get_data_description();
	int32_t _get_data_message();
	int32_t _stop_acquisition();

	struct BA_SETUP bp_setup;							// Setup structure
	struct BA_CALIBRATION_SETTINGS	bp_calibration;	// Calibration Structure (optional)

	uint32_t blocksize;
	uint8_t resolution;

	uint64_t position;

	int32_t errorCode;				// Error Code for failure handling (234 on windows)
	uint32_t countCorrect;
	uint32_t countNotCorrect;

	std::string* devicename;
	long driverversion;
	AmpType amplifiers[4];
	int32_t number_of_amps;
	uint16_t lastmarker;

	uint64_t lasttime;
	uint32_t lasttimestamp;

	static const char* channelnames[];
	int calibration;

#ifdef __windows__
	HANDLE	bp_device; 				// Amplifier device
#else
	int bp_device;					// Amplifier device
	// bua-wrapper
	BUAWrapper* bua;
#endif


	REGISTER_DEC_TYPE(BPAcquisition);
};

#endif //__ACQ_BP_H__
