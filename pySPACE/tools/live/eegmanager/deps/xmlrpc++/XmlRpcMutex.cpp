#if defined(XMLRPC_THREADS)

#include "XmlRpcMutex.h"

#ifdef WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#else
# include <pthread.h>
#endif

using namespace XmlRpc;


//! Destructor.
XmlRpcMutex::~XmlRpcMutex()
{
  if (_pMutex)
  {
#ifdef WIN32
    ::CloseHandle((HANDLE)_pMutex);
#else
    ::pthread_mutex_destroy((pthread_mutex_t*)_pMutex);
    delete _pMutex;
#endif
    _pMutex = 0;
  }
}

//! Wait for the mutex to be available and then acquire the lock.
void XmlRpcMutex::acquire()
{
#ifdef WIN32
  if ( ! _pMutex)
    _pMutex = ::CreateMutex(0, TRUE, 0);
  else
    ::WaitForSingleObject(_pMutex, INFINITE);
#else
  if ( ! _pMutex)
  {
    _pMutex = new pthread_mutex_t;
    ::pthread_mutex_init((pthread_mutex_t*)_pMutex, 0);
  }
  ::pthread_mutex_lock((pthread_mutex_t*)_pMutex);
#endif
}

//! Release the mutex.
void XmlRpcMutex::release()
{
  if (_pMutex)
#ifdef WIN32#ifdef WIN32
    ::ReleaseMutex(_pMutex);
#else
    ::pthread_mutex_unlock((pthread_mutex_t*)_pMutex);
#endif
}

#endif // XMLRPC_THREADS

