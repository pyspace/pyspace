#include "eegmanager.h"

EEGManager::EEGManager()
{
	fill.resize(1);
	bandwidth.resize(1);
}

EEGManager::~EEGManager()
{
	// stop possible running modules
	stop_modules();

	// delete all modules
	cleanup();

	// delete setup history
	while(setup.size() > 0) {
		setup.erase(setup.begin());
	}
}

int32_t EEGManager::check()
{
	// do we have modules in the list?
	if(modules.size() == 0) {
		if(setup.size() == 0) {
			WTF("No Modules in current Setup!");
			return -1;
		}
		FYI("replaying setup..");
		for(int s=0; s<setup.size(); s++) {
			add_module_internal(setup[s].first, setup[s].second);
		}
	}

	// do we now have enough modules?
	if(modules.size() < 2) {
		OMG("cannot work with %d module(s)!", modules.size());
		return -1;
	}

	// sanity check:
	// first module has to be input, last output and so on
	for(int i=0; i<modules.size(); i++) {
		if(i == 0) {
			if(!(modules[i]->isInput())) {
				WTF("First module (%s) is not of type input.", modules[i]->description().c_str());
				cleanup();
				return -1;
			}
		} else if(i == modules.size()-1) {
			if(!(modules[i]->isOutput())) {
				WTF("Last module (%s) is not of type output.", modules[i]->description().c_str());
				cleanup();
				return -1;
			}
		} else {
			if(!(modules[i]->isStream())) {
				WTF("Module at index %d (%s) is not of type stream.", i, modules[i]->description().c_str());
				cleanup();
				return -1;
			}
		}
	}

	return 0;


}

void EEGManager::run()
{


	uint64_t startTime;
	std::vector<Module*>::iterator itm;
//	RingBuffer* rb;

	startTime = getTime();

	if(0 > connect_modules()) goto error;

	fill.resize(buffers.size());
	bandwidth.resize(buffers.size());

	start_modules();

	working = true;

	while(working) {
		// perform some monitoring tasks on the running modules
		// ->sleep exactly one second.
		//   this way it will gernerate kB/sec. in the bandwith vector
		//   and wont create much overhead
		msleep(1000);

		// check buffer filling state and bandwidth
		for(int i=0; i<buffers.size(); i++) {
			fill[i] = buffers[i]->fill();
			bandwidth[i] = buffers[i]->bandwidth();
		}

		// check for state of all modules
		for(itm=modules.begin(); itm!=modules.end(); itm++) {
			if(!((Thread*)(*itm))->isRunning()) {
				printf("%s -> stopped running!\n", (*itm)->description().c_str());
				working = false;
			}
		}

		fflush(stdout);
	}

//	msleep(1000);

	for(itm=modules.begin(); itm!=modules.end(); itm++) {
		if(((Thread*)(*itm))->isRunning()) {
			FYI("waiting for thread %s ..", (*itm)->description().c_str());
			while(((Thread*)(*itm))->isRunning()) msleep(100);
			((Thread*)(*itm))->wait();
			FYI("thread %s finished!", (*itm)->description().c_str());
		}
	}

	FYI("All modules terminated");

error:
	cleanup();

	// final output
	FYI("eegmanager done - runtime %d ms", getTimeDiff(startTime));

	return;
}

int32_t EEGManager::add_module(std::string name, std::string opts)
{
	// is it safe to add a module?
	if(working == true && isRunning()) {
		FYI("please stop first!");
		return -1;
	}

	// clear previous setup when new call
	// to this function occurs
	if(setup.size() > 0) setup.clear();

	return add_module_internal(name, opts);
}

int32_t EEGManager::add_module_internal(std::string name, std::string opts)
{
	// create the instance
	modules.push_back(ModuleFactory::createInstance(name));
	if(modules.at(modules.size()-1) == 0) {
		modules.pop_back();
		return -1;
	}

	// setup the instance
	if(0 > modules.at(modules.size()-1)->setup(opts)) {
		WTF("failed to setup module [%s] with options [%s]\n", modules.at(modules.size()-1)->description().c_str(), opts.c_str());
		modules.pop_back();
		return -1;
	}

	return 0;
}


void EEGManager::stop()
{
	stop_modules();
}


int32_t EEGManager::connect_modules()
{

	RingBuffer* link = new RingBuffer(FIFO_SIZE);
	if(link->get_size() != FIFO_SIZE) {
		OMG("Error allocating ringbuffer!");
		return -1;
	}
	buffers.push_back(link);

	for(int i=0; i<modules.size(); i++) {

		if(i == 0) {
			modules.at(i)->setNext(link);
		} else if(i == modules.size()-1) {
			modules.at(i)->setPrev(link);
		} else {
			modules.at(i)->setPrev(link);
			link = NULL;
			link = new RingBuffer(FIFO_SIZE);
			if(link->get_size() != FIFO_SIZE) {
				OMG("Error allocating ringbuffer!");
				return -1;
			}
			buffers.push_back(link);
			modules.at(i)->setNext(link);
		}
	}

	FYI("successfully connected all modules");
	return 0;

}



void EEGManager::start_modules()
{
	std::vector<Module*>::iterator it;

	// update current setup,
	while(setup.size() > 0) {
		setup.erase(setup.begin());
	}
	for(it=modules.begin(); it!=modules.end(); it++) {
		setup.push_back(make_pair((*it)->description(), (*it)->getParameters()));
	}

	for(it=modules.begin(); it!=modules.end(); it++) {
		((Thread*)(*it))->start();
	}
	FYI("started all modules");

}

void EEGManager::stop_modules()
{
	if(modules.size() == 0) return;

	std::vector<Module*>::iterator it;
	for(it=modules.begin(); it!=modules.end(); it++) {
		(*it)->stop();
	}

	// wait for threads to finish
	for(it=modules.begin(); it!=modules.end(); it++) {
		if(!(*it)->isRunning()) continue;
		(*it)->wait();
	}

	// clean buffers
	for(int i=0; i<buffers.size(); i++) {
		buffers[i]->reset();
	}

	FYI("stopped all modules");
}


std::vector< std::pair<std::string, std::string> > EEGManager::current_setup()
{
    std::vector<Module*>::iterator it;
    
	// update current setup,
    if(modules.size() > 0) {
		while(setup.size() > 0) {
			setup.erase(setup.begin());
		}
		for(it=modules.begin(); it!=modules.end(); it++) {
			setup.push_back(make_pair((*it)->description(), (*it)->getParameters()));
		}
    }
    
	return setup;
}


void EEGManager::cleanup()
{
	while(modules.size() > 0) {
		delete modules.front();
		modules.erase(modules.begin());
	}

	RingBuffer* buffer = NULL;
	while(buffers.size() > 0) {
		delete buffers.front();
		buffers.erase(buffers.begin());
	}
}
