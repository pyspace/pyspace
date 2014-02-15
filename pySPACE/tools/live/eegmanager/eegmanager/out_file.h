#ifndef __OUT_FILE_H__
#define __OUT_FILE_H__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <fstream>

#include "global.h"
#include "EEGStream.h"
#include "glob_module.h"
#include "util_thread.h"
#include "utils.h"

class FILEOutput : public Module
{
public:
	FILEOutput();

	std::string inputType();
	std::string outputType();
	std::string description();

	int32_t setup(std::string opts);

	~FILEOutput();


private:

	void run();

	std::string* directory;
	std::string* subject;
	std::string* experiment;
	std::string* basename;
	std::string* filename;

	int16_t set_number;
	uint32_t marker_number;
	uint32_t position;

	double nv[4];

	std::ofstream* vhdr;
	std::string* vhdr_name;
	std::ofstream* eeg;
	std::string* eeg_name;
	std::ofstream* vmrk;
	std::string* vmrk_name;

	int32_t _write_data_description();
	int32_t _write_data_message(MessageHeader* pHeader);

	int online;

	REGISTER_DEC_TYPE(FILEOutput);
};
#endif // __OUT_FILE_H__
