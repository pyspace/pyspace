#ifndef __UTIL_RINGBUFFER__
#define __UTIL_RINGBUFFER__

#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "util_mutex.h"
#include "global.h"
#include "utils.h"

class RingBuffer {
    
public:
    RingBuffer(size_t size);
    ~RingBuffer();
    
    int32_t read(char* dst, uint32_t s);
    int32_t blocking_read(char* dst, uint32_t s, bool* r = NULL);

    int32_t write(char* src, uint32_t s);
    int32_t blocking_write(char* src, uint32_t s, bool* r = NULL);

	int32_t flush(uint32_t s);
    
    float bandwidth();
    float fill();

    uint32_t get_size(void);
    uint32_t get_used(void);
    
    void reset();

private:
    Mutex* m;
    char* data;
    
    uint32_t in;
    uint32_t out;
    uint32_t used;
    uint32_t buffer_size;

    float total_data;
    
    uint32_t read_space(void);
    uint32_t write_space(void);

    bool check(bool* r);
};

#endif //__UTIL_RINGBUFFER__
