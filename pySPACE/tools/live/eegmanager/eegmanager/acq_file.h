#ifndef __ACQ_FILE__
#define __ACQ_FILE__

#include "global.h"
#include "glob_module.h"

#include "util_thread.h"
#include "utils.h"

#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

class FILEAcquisition : public Module {

public:
	FILEAcquisition();
	~FILEAcquisition();

	std::string inputType();
	std::string outputType();
	std::string description();

	int32_t setup(std::string opts);
	
	virtual int32_t getMessage(uint32_t type);

	void run();

private:

	int32_t _get_data_description();
	int32_t _get_data_message();

	std::string current_marker_name;
	uint32_t current_marker_position;
	uint16_t current_marker_value;
	uint16_t _marker_at_position(int32_t p);

	uint32_t position;
	uint32_t blocksize;

	std::string* filename;

	std::ifstream* vhdr;
	std::string* vhdr_name;
	std::ifstream* eeg;
	std::string* eeg_name;
	std::ifstream* vmrk;
	std::string* vmrk_name;

	REGISTER_DEC_TYPE(FILEAcquisition);


};


#endif //__ACQ_FILE__
