""" This module contains a class that provides memoization support for
generator. Most likely there would be cleaner and more general
ways of doing this but for the moment is suffices... 

Meaning of this method:
https://secure.wikimedia.org/wikipedia/en/wiki/Memoization

:Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
:Created: 2008/11/25
"""

import itertools

class MemoizeGeneratorNotRefreshableException(Exception):
    pass

class MemoizeGenerator(object):
    """ Object to encapsulate a generator so that one iterate over
    the generators output several time. The output is computed
    only once and then stored in a cache. Be careful in cases where
    the generator might produce memory-intensive outputs!
    """
    
    def __init__(self, generator, caching=False):
        """ Stores the generator and creates an empty cache
        
        .. note:: 
                Since the output of the generator is ordered,
                the cache is an ordered sequence of variable length like a list
        """
        self.generator = generator
        self.caching = caching
        self.refreshable = True
        if self.caching:
            self.cache = []
        
    def _fetch_from_generator(self):
        """
        Fetches one fresh value from the generator, store it in the
        cache and yield it
        """ 
        while True:
            nextValue = self.generator.next()
            if self.caching:
                self.cache.append(nextValue)
            else:
                self.refreshable = False
                
            yield nextValue
        
    def fresh(self):
        """ Return one generator that yields the same values
        like the internal one that was passed to __init__. 
        
        .. note:: It does not recompute values that have already
            been requested before but just uses these from the internal cache.
        
        .. note:: Calling fresh invalidates all existing 
                generators that have been created before using this method,
                i.e. there can only be one generator at a time 
        """
        if self.caching:
            return itertools.chain(self.cache,
                                   self._fetch_from_generator())
        else:
            if not self.refreshable:
                raise MemoizeGeneratorNotRefreshableException( "This MemoizeGenerator does not cache elements from the generator and can thus not be reset") 
                
            return self._fetch_from_generator()
        
