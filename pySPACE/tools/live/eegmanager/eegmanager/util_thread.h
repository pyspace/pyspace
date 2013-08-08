/**
 *  Copyright 2011, DFKI GmbH Robotics Innovation Center
 *
 *  This file is part of the MARS simulation framework.
 *
 *  MARS is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License
 *  as published by the Free Software Foundation, either version 3
 *  of the License, or (at your option) any later version.
 *
 *  MARS is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public License
 *   along with MARS.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef MARS_THREAD_H
#define MARS_THREAD_H

#include <pthread.h>
#include <cstddef> // for std::size_t
#include <list>

#include "util_mutex.h"

//namespace mars {

  class Thread {
  public:
    Thread();
    virtual ~Thread();

    void setStackSize(std::size_t stackSize);
    void start();
    bool wait();
    bool wait(unsigned long timeoutMilliseconds);
    bool isRunning() const;
    bool isFinished() const;
    std::size_t getStackSize() const;
    static Thread* getCurrentThread();

  protected:
    virtual void run() = 0;

  private:
    // disallow copying
    Thread(const Thread &);
    Thread &operator=(const Thread &);
    
    static void *runHelper(void *context);
    pthread_t myThread;
    std::size_t myStackSize;
    bool running;
    bool finished;
    static Mutex threadListMutex;
    static std::list<Thread*> threads;
  }; // class Thread

//}; // namespace mars


#endif /* MARS_THREAD_H */
