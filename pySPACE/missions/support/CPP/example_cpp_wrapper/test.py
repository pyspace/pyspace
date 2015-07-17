import numpy as np

import example_wrapper as ew

array1=np.arange(100,dtype=float)
array2=np.zeros(100,dtype=float)
multiplier = 100
divisor = 5.


# Should return one and array2 should be equal to:
# array1*multiplier/divisor
return_value = ew.example(array1, array2, multiplier, divisor)

checkArray = array1*multiplier/divisor

print "The calculations are equal:", np.array_equal(array2,checkArray)