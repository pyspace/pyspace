#include "util_filebuffer.h"

FileBuffer::FileBuffer(size_t size) : BaseBuffer(size)
{
	pipe(p);

	i = fdopen(p[0], "w");
	o = fdopen(p[1], "r");

    // FYI("openend named pipe: in: %d, out: %d", p[0], p[1]);

	FD_ZERO(&w_set);
	FD_ZERO(&r_set);

	buffer_size = size;

    used = 0;
}

FileBuffer::~FileBuffer()
{
    close(p[0]);
    close(p[1]);
}

int32_t FileBuffer::read(char* dst, uint32_t s)
{
	int32_t size = 0;
	r_timeout.tv_sec = 0;
	r_timeout.tv_usec = 80;

    // delay execution on read attempt from empty buffer
    if(used == 0) {
        select(0, NULL, NULL, NULL, &r_timeout);
        return 0;
    }

	FD_ZERO(&r_set);
	FD_SET(p[0], &r_set);

	if(select(p[0]+1, &r_set, 0, 0, &r_timeout) > 0) {
		if(FD_ISSET(p[0], &r_set)) {
			size = ::read(p[0], (void*)dst, (size_t)s);
			if(0 > size) {
				FYI("Error %d when reading!", errno);
				return 0;
			}
		}
	}

    __sync_fetch_and_sub(&used, size);
	bytes_read += size;

	return size;
}

int32_t FileBuffer::write(char* src, uint32_t s)
{
	int32_t size = 0;
	w_timeout.tv_sec = 0;
	w_timeout.tv_usec = 1000;

    // put a logarithmic soft margin on the buffer size
    if(used > buffer_size) {
        float factor = std::max(1.0, log((used-buffer_size)/512));
        s = s /(int)factor;
    }

	FD_ZERO(&w_set);
	FD_SET(p[1], &w_set);

	if(select(p[1]+1, 0, &w_set, 0, &w_timeout) > 0) {
		if(FD_ISSET(p[1], &w_set)) {
			size = ::write(p[1], (void*)src, (size_t)s);
			if(0 > size) {
				FYI("Error %d when writing!", errno);
				return 0;
			}
		}
	}

    __sync_fetch_and_add(&used, size);
	bytes_written += size;

	return size;

}

int32_t FileBuffer::flush(uint32_t s)
{
    char data[s];
    return read(data, s);
}

void FileBuffer::reset(void)
{

}

uint32_t FileBuffer::bytes_used(void)
{
	return used;
}

float FileBuffer::fill(void)
{
    return ((float)bytes_used()*100)/buffer_size;
}

uint32_t FileBuffer::bytes_left()
{
	return buffer_size-bytes_used();
}

float FileBuffer::bandwidth(void)
{
    float ret = (float)(bytes_read-total_data)/1024.0;
    total_data = bytes_read;

    return ret;
}

// convenience functions
int32_t FileBuffer::blocking_write(char* src, uint32_t s, bool* r)
{
	size_t size = 0;
	while(size < (size_t)s && check(r)) {
		size += write((char*)(src+size), s-size);
	}
	return size;
}

int32_t FileBuffer::blocking_read(char* dst, uint32_t s, bool* r)
{
	size_t size = 0;
	while(size < (size_t)s && check(r)) {
		size += read((char*)(dst+size), s-size);
	}
	return size;
}

bool FileBuffer::check(bool* r)
{
	if(r == NULL) return true;
	return *r;
}
