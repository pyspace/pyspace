#include "util_basebuffer.h"

BaseBuffer::BaseBuffer(size_t size)
{
	bytes_read = 0;
	bytes_written = 0;
	total_data = 0;
    buffer_size = 0;
}

BaseBuffer::~BaseBuffer()
{

}

uint32_t BaseBuffer::get_size(void)
{
    return buffer_size;
}
