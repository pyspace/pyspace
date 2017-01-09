#include "example_wrapper.h"

/**************************************************************************************************************
* THIS IS AN EXAMPLE FOR WRAPPING C IN PYTHON USING THE PYTHON C-API
* FOR FURTHER INFORMATION WRITING C EXTENSIONS SEE: http://wiki.scipy.org/Cookbook/C_Extensions/NumPy_arrays
***************************************************************************************************************/

//                  __  __                                                                               __   
//     ____  __  __/ /_/ /_  ____  ____     _      ___________ _____  ____  ___  _____   _________  ____/ /__ 
//    / __ \/ / / / __/ __ \/ __ \/ __ \   | | /| / / ___/ __ `/ __ \/ __ \/ _ \/ ___/  / ___/ __ \/ __  / _ \
//   / /_/ / /_/ / /_/ / / / /_/ / / / /   | |/ |/ / /  / /_/ / /_/ / /_/ /  __/ /     / /__/ /_/ / /_/ /  __/
//  / .___/\__, /\__/_/ /_/\____/_/ /_/    |__/|__/_/   \__,_/ .___/ .___/\___/_/      \___/\____/\__,_/\___/ 
// /_/    /____/                                            /_/   /_/                                         
//
//



/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
    int n;
    
    n=arrayin->dimensions[0];
    return (double *) arrayin->data;  /* pointer to arrayin data as double */
}
/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_doublevector(PyArrayObject *vec)  {
    if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
        PyErr_SetString(PyExc_ValueError,
            "In not_doublevector: array must be of type Float and 1 dimensional (n).");
        return 1;  }
    return 0;
}

/* Example function each value of array1 is multplied with multiplier and 
   devided by divisor the results are stored in array2. The return value is 
   arbitary chosen to be an integer of value 1. The operations on the array 
   are inplace, therefore no returning of the arrays is neccesary.
   Params:
            - array1       Array1 containing some values
            - array2       Array2 containing some values
            - multiplier   int value
            - divisor      double value
    Return:
            - return_value int value 1

*/

static PyObject* exampleCPPfunction(PyObject *self, PyObject *args){
    // Passed variables
    PyArrayObject *array1, *array2;
    int multiplier;
    double divisor;
    
    // Pointer for converted NumPy Arrays
    double *cArray1, *cArray2;

    // Extract variables and arrays from ARGS
    // O! belongs to one array O -> type; ! -> local Python variable
    // i belongs to an integer variable
    // d belongs to ab double variable
    if (!PyArg_ParseTuple(args, "O!O!id", &PyArray_Type, &array1, &PyArray_Type, &array2,
                                              &multiplier, &divisor)) return NULL;
    
    // Check wheter all array are of type double
    if (not_doublevector(array1)) return NULL;
    if (not_doublevector(array2)) return NULL;
    
    // Convert NumPy arrays to C-Arrays
    cArray1=pyvector_to_Carrayptrs(array1);
    cArray2=pyvector_to_Carrayptrs(array2);
    
    
    // Lengths of the arrays
    unsigned int array1Size = array1->dimensions[0];
    
    // The actual calculation part. NOTE it is assumed, 
    // that both arrays have the same dimension
    for(unsigned int i=0; i<array1Size; i++)
    {
        cArray2[i] = cArray1[i]*multiplier/divisor;
    }
    
    return Py_BuildValue("i", 1);
    
}


// method definitions for this module
// "example" => Function name in Python
// exampleCPPfunction => Function name within this moule
static PyMethodDef example_wrapper_methods[] = {
    {"example", exampleCPPfunction, METH_VARARGS,"Example function"},
    {NULL, NULL, 0, NULL}
};

// the required init function (gets called 
// once during the import statement in Python)
PyMODINIT_FUNC initexample_wrapper(void)
{   
    static PyObject* example_error;
    PyObject *m;
    m = Py_InitModule("example_wrapper", example_wrapper_methods);
    import_array();
    char Acception[] = "example_wrapper.AccessException";
    example_error = PyErr_NewException(Acception, NULL, NULL);
    Py_INCREF(example_error);
    PyModule_AddObject(m, "AccessException", example_error);
}
