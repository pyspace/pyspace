#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// class for handling shared memory access
class VarianceTools {

private:
    /* .... C matrix utility functions ..................*/
    double **pyvector_to_Carrayptrs(PyArrayObject *arrayin);
    double **pyvector(PyObject *objin);
    int  not_doublevector(PyArrayObject *vec);

};