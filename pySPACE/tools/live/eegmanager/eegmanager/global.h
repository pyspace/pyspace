#ifndef __GLOBAL_H__
#define __GLOBAL_H__



#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <iterator>
#include <getopt.h>
#include <assert.h>

#ifdef __windows__
#include <windows.h>
#include <Time.h>
// socket compatability
#include <winsock.h>
#include <wchar.h>
#define EWOULDBLOCK WSAEWOULDBLOCK
#else
#define SOCKET_ERROR -1
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <errno.h>
#endif

#include "EEGStream.h"
#include "util_mutex.h"
#include "utils.h"

#define DEBUG 1
#define WARN 1

#define TO_MICROBLAZE 1
#define TO_NETWORK 2

#define FIFO_SIZE (uint32_t)256000

#define WTF(...) printf("\nFATAL: %s\n\t",__PRETTY_FUNCTION__);printf(__VA_ARGS__);printf("\n")

#ifdef WARN
#define OMG(...) printf("%s\t",__PRETTY_FUNCTION__);printf(__VA_ARGS__);printf("\n")
#else
#define OMG(...)
#endif

#ifdef DEBUG
#define FYI(...) printf("%s\n\t",__PRETTY_FUNCTION__);printf(__VA_ARGS__);printf("\n")
#else
#define FYI(...)
#endif


#ifdef MY_MICROBLAZE
#define __ARCH__ "microblaze"
#define _e64(x) (((x & 0x00000000000000FFULL) << 56) + ((x & 0x000000000000FF00ULL) << 40) + ((x & 0x0000000000FF0000ULL) << 24) + ((x & 0x00000000FF000000ULL) <<  8) + ((x & 0x000000FF00000000ULL) >>  8) + ((x & 0x0000FF0000000000ULL) >> 24) + ((x & 0x00FF000000000000ULL) >> 40) + ((x & 0xFF00000000000000ULL) >> 56))
#define _e32(x) (((x&0xff000000)>>24) + ((x&0xff0000)>>8) + ((x&0xff00)<<8) + ((x&0xff)<<24))
#define _e16(x) (((x&0xff00)>>8) + ((x&0x00ff)<<8))
#else
#define __ARCH__ "other"
#define _e64(x) x
#define _e32(x) x
#define _e16(x) x
#endif

// TIME helper functions
// diff = t2 - t1 [ms]
inline uint32_t diff_ms(struct timeval t1, struct timeval t2)
{
	return (t2.tv_sec-t1.tv_sec)*1000 + (t2.tv_usec-t1.tv_usec)/1000;
}

// diff = t2 - t1 [us]
inline uint32_t diff_us(struct timeval t1, struct timeval t2)
{
	return (t2.tv_sec-t1.tv_sec)*1000000 + (t2.tv_usec-t1.tv_usec);
}

// MEMORY debugging
//#define freee(x) printf("freeing pointer at %#08x\n", x);fflush(stdout);free(x);
#define freee(x) free(x);

#endif // __GLOBAL_H__
