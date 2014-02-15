#ifndef __GLOB_MODULE__
#define __GLOB_MODULE__

#include <map>
#include <typeinfo>
#include <string>
#include <cstddef>
#include <getopt.h>

#include "global.h"
#include "util_thread.h"
#include "utils.h"
#include "util_ringbuffer.h"

class Module : public Thread
{

public:
	Module();
	virtual ~Module();

	virtual int32_t setup(std::string opts) = 0;

	virtual std::string description() = 0;
	std::string getParameters();
	std::string usage();

	// setting IPC-pointers
	void setPrev(RingBuffer* p);
	void setNext(RingBuffer* n);

	float i_bandwith();
	float o_bandwith();

	// characterize modules
	virtual std::string inputType() = 0;
	virtual std::string outputType() = 0;
	bool isInput();
	bool isOutput();
	bool isStream();

	void stop();

protected:

	// pointer to shared memories
	RingBuffer* prev;
	RingBuffer* next;

	EEGStartMessage* in_data_description;
	EEGStartMessage* out_data_description;
	MessageHeader* in_data_message;
	MessageHeader* out_data_message;

	virtual int32_t getMessage(uint32_t type);
	virtual int32_t process(void);
	virtual int32_t putMessage(MessageHeader* pHeader);

	int32_t block_length_ms_out();
	int32_t block_length_ms_in();
	int32_t block_length_ms();

	// helper functions
	void string2argv(std::string *opts, int *argc, char** &argv);
	void endianflip(MessageHeader** header, int dir);
	void datadescription2text(MessageHeader* dd);

	// helper fields and functions for
	// option parsing
	struct option* module_options;
	char* option_storage;
	std::string* parameters;
	void merge_options(option* module_options, size_t size);

	bool working;
	uint64_t abs_start_time;

	int meta; // flag for the gui to trigger extra buttons etc.

};


#define REGISTER_DEC_TYPE(NAME) \
    static DerivedRegister<NAME> reg

#define REGISTER_DEF_TYPE(NAME) \
    DerivedRegister<NAME> NAME::reg(#NAME)

template<typename T> Module * createT() { return (Module*)((T*)( new T )); }

struct ModuleFactory {
    typedef std::map<std::string, Module*(*)()> map_type;

    static Module * createInstance(std::string const& s) {
        map_type::iterator it = getMap()->find(s);
        if(it == getMap()->end()) {
        	WTF("did not find module with name [%s]!", s.c_str());
        	return 0;
        }
        return it->second();
    }

    static map_type * getMap() {
        // never delete'ed. (exist until program termination)
        // because we can't guarantee correct destruction order
        if(!map) { map = new map_type; }
        return map;
    }

private:
    static map_type * map;
};

template<typename T>
struct DerivedRegister : ModuleFactory {
    DerivedRegister(std::string const& s) {
        getMap()->insert(std::make_pair(s, &createT<T>));
    }
};



#endif //__GLOB_MODULE__
