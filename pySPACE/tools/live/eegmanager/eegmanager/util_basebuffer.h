#ifndef __UTIL_BASEBUFFER__
#define __UTIL_BASEBUFFER__

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include "global.h"

class BaseBuffer {

public:
	BaseBuffer(size_t size);
    ~BaseBuffer();

    virtual int32_t read(char* dst, uint32_t s) = 0;
    virtual int32_t blocking_read(char* dst, uint32_t s, bool* r = NULL) = 0;

    virtual int32_t write(char* src, uint32_t s) = 0;
    virtual int32_t blocking_write(char* src, uint32_t s, bool* r = NULL) = 0;

	virtual int32_t flush(uint32_t s) = 0;

    virtual float bandwidth() = 0;
    virtual float fill() = 0;
    uint32_t get_size(void);

    virtual uint32_t bytes_used(void) = 0;
    virtual uint32_t bytes_left(void) = 0;

    virtual void reset() = 0;

protected:
    uint32_t bytes_read;
    uint32_t bytes_written;
    uint32_t total_data;
    uint32_t buffer_size;
};

#endif //__UTIL_BASEBUFFER__
