#ifndef __UTIL_FILEBUFFER__
#define __UTIL_FILEBUFFER__

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>

#include "util_basebuffer.h"
#include "util_mutex.h"
#include "global.h"
#include "utils.h"

class FileBuffer : public BaseBuffer {

public:
	FileBuffer(size_t size);
    ~FileBuffer();

    int32_t read(char* dst, uint32_t s);
    int32_t blocking_read(char* dst, uint32_t s, bool* r = NULL);

    int32_t write(char* src, uint32_t s);
    int32_t blocking_write(char* src, uint32_t s, bool* r = NULL);

	int32_t flush(uint32_t s);

    float bandwidth();
    float fill();

    uint32_t bytes_used(void);
    uint32_t bytes_left(void);

    void reset();

private:

    int p[2];
    FILE* i;
    FILE* o;
    fd_set r_set;
    fd_set w_set;
    struct timeval r_timeout;
    struct timeval w_timeout;

    float total_data;
    uint32_t used;

    bool check(bool* r);

#ifdef PROFILE
	FILE* logfile;
	std::string* logfilename;
#endif
};

#endif //__UTIL_FILEBUFFER__
