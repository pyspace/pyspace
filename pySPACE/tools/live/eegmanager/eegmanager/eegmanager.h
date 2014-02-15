#ifndef __EEGMANAGER_H__
#define __EEGMANAGER_H__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <vector>

#ifndef __windows__
#include <sys/time.h>
#include <sys/ioctl.h>
#endif

#include "global.h"
#include "glob_module.h"
#include "util_thread.h"
#include "util_ringbuffer.h"

#ifdef __windows__
#include "acq_bp.h"
#endif
#include "acq_file.h"
#include "out_net.h"



class EEGManager : public Thread
{
public:
	EEGManager();
	~EEGManager();

	int32_t add_module(std::string name, std::string opts);

	int32_t check();
	void run();
	void stop();

	std::vector<float> fill;
	std::vector<float> bandwidth;
	std::vector< std::pair<std::string, std::string> > current_setup();


private:

	int32_t add_module_internal(std::string name, std::string opts);
	int32_t connect_modules();
	void start_modules();
	void stop_modules();
	void cleanup();

	std::string running_modules();

	// list of current modules
	std::vector<Module*> modules;
	// IPC-Buffers
	std::vector<RingBuffer*> buffers;
	// setup of the current flow
	std::vector< std::pair<std::string, std::string> > setup;

	bool working;
};
#endif // __EEGMANAGER_H__
