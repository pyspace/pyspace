#include "util_ringbuffer.h"

RingBuffer::RingBuffer(size_t size) 
{
    // init
    data = NULL;
    buffer_size = 0;
    total_data = 0;
    m = NULL;
    
    in = 0;
    out = 0;
    used = 0;

    // allocate space
    data = (char*)malloc(size);
    if(data != NULL) {
        buffer_size = (uint32_t)size;
    }
    
    // get mutex
    m = new Mutex(MUTEX_TYPE_NORMAL);
    if(m == NULL) {
        // error getting mutex..
    	OMG("Error creating Mutex!");
    }
}

RingBuffer::~RingBuffer()
{
    // free space
    if(data != NULL) free(data);
    
    // destroy mutex
    if(m != NULL) delete m;
}

bool RingBuffer::check(bool* r) {
	if(r == NULL) return true;
	return *r;
}

int32_t RingBuffer::blocking_read(char* dst, uint32_t s, bool* r)
{
	int32_t size = 0;
	while(size < (int32_t)s && check(r)) {
		size += read((char*)(dst+size), s-size);
		msleep(0);
	}
	return size;
}

int32_t RingBuffer::read(char* dst, uint32_t s) 
{   
    // calculate available data
    size_t current_length = std::min(read_space(), s);
	if(current_length == 0) {
		msleep(1);
		return 0;
	}
	
	// copy block of data
    char* o = data + out;
	memcpy(dst, o, current_length);
	
	// update pointers
	while(m->tryLock() != 0) msleep(0);
	used -= current_length;
	out += current_length;
	if(out >= buffer_size) {
		out -= buffer_size;
	}
	m->unlock();

    return current_length;
}

int32_t RingBuffer::blocking_write(char* src, uint32_t s, bool* r)
{
	uint32_t size = 0;
	while(size < s && check(r)) {
		size += write((char*)(src+size), s-size);
		msleep(0);
	}
	return size;
}

int32_t RingBuffer::write(char* src, uint32_t s) 
{
    // calculate available data
    size_t current_length = std::min(write_space(), s);
	if(current_length == 0) {
		msleep(1);
		return 0;
	}
	
	// copy block of data
    char* i = data + in;
	memcpy(i, src, current_length);
	
	// update pointers
	while(m->tryLock() != 0) msleep(0);
	used += current_length;
	in += current_length;
	if(in >= buffer_size) {
		in -= buffer_size;
	}
	total_data += current_length;
	m->unlock();
    
    return current_length;
}

int32_t RingBuffer::flush(uint32_t s)
{
    // calculate available data
    size_t current_length = std::min(read_space(), s);
	if(current_length == 0) return 0;
	
	// update pointers
	while(m->tryLock() != 0) msleep(0);
	used -= current_length;
	out += current_length;
	if(out >= buffer_size) {
		out -= buffer_size;
	}
	m->unlock();

    return current_length;
}

uint32_t RingBuffer::read_space(void)
{
	uint32_t s = 0;
	uint32_t i = 0;
	uint32_t o = 0;
	uint32_t u = 0;

	while(m->tryLock() != 0) msleep(0);
	i = in;
	o = out;
	u = used;
	m->unlock();

	// space until the wrap around has to occur
	if(o > i) {
		s = buffer_size - o;
	}

	// space until out meet in (again)
	if(o < i) {
		s = i - o;
	}

	// buffer is either full or empty
	if(o == i) {
		if(u > 0) s = buffer_size - o;
		else s = 0;
	}
	return s;
}

uint32_t RingBuffer::write_space(void)
{
	uint32_t s = 0;
	uint32_t i = 0;
	uint32_t o = 0;
	uint32_t u = 0;

	while(m->tryLock() != 0) msleep(0);
	i = in;
	o = out;
	u = used;
	m->unlock();

	// space until the wrap around has to occur
	if(i > o) {
		s = buffer_size - i;
	}

	// space until in meets out (again)
	if(i < o) {
		s = o - i;
	}

	// buffer is either empty or full
	if(i == o) {
		if(used > 0) s = 0;
		else s = buffer_size - i;
	}
	return s;
}

void RingBuffer::reset(void)
{
	while(m->tryLock() != 0) msleep(0);
	in = 0;
	out = 0;
	used = 0;
	m->unlock();
}

uint32_t RingBuffer::get_size(void)
{
    return buffer_size;
}

uint32_t RingBuffer::get_used(void)
{
	return used;
}

float RingBuffer::fill(void) 
{
    return ((float)used*100)/buffer_size;
}

float RingBuffer::bandwidth(void) 
{
    m->lock(); 
    float ret = total_data/1024.0; 
    total_data = 0; 
    m->unlock(); 
    return ret;
}
