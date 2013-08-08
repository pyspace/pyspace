""" This module contains contains methods for prime factorization and
related methods.

:Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
:Created: 2010/11/23
"""

import copy

def factorize(n):
    """
    Simple function for prime factorization using a recursive approach.
    Suitable for small n.
    """
    prime_factors = []
    result = n
    
    # 1 is always a factor...
    if n == 1:
        return [1]
    
    while True:
        if result == 1:
            break
        divisor = 2
        while True:
            if result % divisor == 0:
                break
            divisor += 1
        prime_factors.append(divisor)
        result /= divisor
    return prime_factors


def next_least_nice_integer_divisor(n,d):
    """
    Finds the biggest number dsmall, 
    that is smaller or equal to d, such that n / dsmall is an int 
    """
    # get prime numbers
    prime_factors = factorize(n)
    
    # find dsmall recursively 
    # removes number after number out of a list of prime factors
    # to find the list with the prime factors that has the desired property
    def _multiply_factors_if_to_big(factors, d):
        # empty => no further recursion possible
        if len(factors) == 0:
            return float('inf')
        
        # get product of all prime factors
        product = reduce(lambda x, y: x*y, factors)
        
        # product of prime factors is 
        if product < d:
            return product
        
        
        if product > d:
            subset_products = []
            for i in factors:
                temp_factors = copy.copy(factors)
                temp_factors.remove(i)
                subset_products.append(_multiply_factors_if_to_big(temp_factors, d))
                
            product = max(subset_products)
            
        return product
    
    return _multiply_factors_if_to_big(prime_factors,d)
    
    