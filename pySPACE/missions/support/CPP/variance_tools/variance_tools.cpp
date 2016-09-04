#include "variance_tools.h"

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

/* Function that filters the given data using the variance
   All operations are done in place.
   Params:
            - outData       Array containing the results of the operation
            - inData        Array containing the input timeseries
            - ringbuffer    Array with the n last samples needed for calculation
            - variables     Array containing the last variance and the last mean
            - width         Length of the variance filter (how many samples are used to calculate the variance)
            - index         Index for the current needed sample of the ringbuffer
    Return:
            - index         Returns the updated index for the ringbuffer

*/

static PyObject* variance_filter(PyObject *self, PyObject *args){
    // Passed variables
    PyArrayObject *outData, *inData, *ringbuffer, *variables;
    int width, index;
    
    // Pointer for converted NumPy Arrays
    double *cOutData, *cInData, *cRingbuffer, *cVariables;

    // Extract variables and arrays from ARGS
    if (!PyArg_ParseTuple(args, "O!O!O!O!ii", &PyArray_Type, &outData, &PyArray_Type, &inData,
                                              &PyArray_Type, &ringbuffer, &PyArray_Type, &variables,
                                              &width, &index)) return NULL;
    
    // Check wheter all array are of type double
    if (not_doublevector(outData)) return NULL;
    if (not_doublevector(inData)) return NULL;
    if (not_doublevector(ringbuffer)) return NULL;
    if (not_doublevector(variables)) return NULL;
    
    // Convert NumPy array to C-Arrays
    cOutData=pyvector_to_Carrayptrs(outData);
    cInData=pyvector_to_Carrayptrs(inData);
    cRingbuffer=pyvector_to_Carrayptrs(ringbuffer);
    cVariables=pyvector_to_Carrayptrs(variables);
    
    
    //Some local variables for speed up
    unsigned int inDataSize = inData->dimensions[0];
    double ww = width*width;
    double wm1 = width-1;
    double wp1 = width+1;

    double ringbufferValue=0.0;
    double inDataValue=0.0;
    double variable1 = 0.0;
    
    
    
    for(unsigned int i=0; i<inDataSize; i++)
    {
        //Speedup for array entries which are needed several times
        ringbufferValue = cRingbuffer[index];
        inDataValue = cInData[i];
        variable1 = cVariables[1];

        //Calculating the new variance
        cVariables[0] = cVariables[0] + (inDataValue - ringbufferValue) * ( ((wm1) * inDataValue) + ((wp1) * ringbufferValue) - (2.0*variable1));

        //Calculating the new mean value
        cVariables[1] = variable1 + (inDataValue-ringbufferValue);

        //Store the actual sample in the ringbuffer
        cRingbuffer[index] = inDataValue;

        //Increment the ringbuffer index
        index = (index < wm1 ? index+1:0);

        //Write the variance to the output array
        cOutData[i] = cVariables[0]/(ww);
    }
    
    return Py_BuildValue("i", index);
    
}


/* Function that filters the given data using the standard deviation
   All operations are done in place.
   Params:
            - outData       Array containing the results of the operation
            - inData        Array containing the input timeseries
            - ringbuffer    Array with the n last samples needed for calculation
            - variables     Array containing the last variance and the last mean
            - width         Length of the standard deviation filter (how many samples are used to calculate the standard deviation)
            - index         Index for the current needed sample of the ringbuffer
    Return:
            - index         Returns the updated index for the ringbuffer

*/

static PyObject* standard_filter(PyObject *self, PyObject *args){
    // Passed variables
    PyArrayObject *outData, *inData, *ringbuffer, *variables;
    int width, index;
    
    // Pointer for converted NumPy Arrays
    double *cOutData, *cInData, *cRingbuffer, *cVariables;

    // Extract variables and arrays from ARGS
    if (!PyArg_ParseTuple(args, "O!O!O!O!ii", &PyArray_Type, &outData, &PyArray_Type, &inData,
                                              &PyArray_Type, &ringbuffer, &PyArray_Type, &variables,
                                              &width, &index)) return NULL;
    
    // Check wheter all array are of type double
    if (not_doublevector(outData)) return NULL;
    if (not_doublevector(inData)) return NULL;
    if (not_doublevector(ringbuffer)) return NULL;
    if (not_doublevector(variables)) return NULL;
    
    // Convert NumPy array to C-Arrays
    cOutData=pyvector_to_Carrayptrs(outData);
    cInData=pyvector_to_Carrayptrs(inData);
    cRingbuffer=pyvector_to_Carrayptrs(ringbuffer);
    cVariables=pyvector_to_Carrayptrs(variables);
    
    
    //Some local variables for speed up
    unsigned int inDataSize = inData->dimensions[0];
    double ww = width*width;
    double wm1 = width-1;
    double wp1 = width+1;

    double ringbufferValue=0.0;
    double inDataValue=0.0;
    double variable1 = 0.0;
    
    
    
    for(unsigned int i=0; i<inDataSize; i++)
    {
        //Speedup for array entries which are needed several times
        ringbufferValue = cRingbuffer[index];
        inDataValue = cInData[i];
        variable1 = cVariables[1];

        //Calculating the new variance
        cVariables[0] = cVariables[0] + (inDataValue - ringbufferValue) * ( ((wm1) * inDataValue) + ((wp1) * ringbufferValue) - (2.0*variable1));

        //Calculating the new mean value
        cVariables[1] = variable1 + (inDataValue-ringbufferValue);

        //Store the actual sample in the ringbuffer
        cRingbuffer[index] = inDataValue;

        //Increment the ringbuffer index
        index = (index < wm1 ? index+1:0);

        //Write the variance to the output array
        cOutData[i] = sqrt(cVariables[0]/(ww));
    }
    
    return Py_BuildValue("i", index);
    
}



/* Function that calculates the energie of the signal using the Teager Kaiser Energie Operator(TKEO)
   This is a quadratic filter with the formula:

   .. math::

      x_{i-1}^2 - x_{i-2} \\cdot x_i

      The formula is taken from the following publication::

         Kaiser J. F. (1990)
         On a simple algorithm to calculate 'energy' of a signal.
         In Proceedings:
         International Conference on Acoustics, Speech, and Signal Processing (ICASSP-90)
         Pages 381-384
         (http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=115702)
   All operations are done in place.
   Params:
            - outData       Array containing the results of the operation
            - inData        Array containing the input timeseries
            - oldData       Array with the 2 last samples from previous window needed for calculation
    Return:
            - index         Returns 1

*/

static PyObject* tkeo_filter(PyObject *self, PyObject *args){
    // Passed variables
    PyArrayObject *outData, *inData, *oldData;
    
    // Pointer for converted NumPy Arrays
    double *cOutData, *cInData, *cOldData;

    // Extract variables and arrays from ARGS
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &outData, &PyArray_Type, &inData,
                                              &PyArray_Type, &oldData)) return NULL;
    
    // Check wheter all array are of type double
    if (not_doublevector(outData)) return NULL;
    if (not_doublevector(inData)) return NULL;
    if (not_doublevector(oldData)) return NULL;
    
    // Convert NumPy array to C-Arrays
    cOutData=pyvector_to_Carrayptrs(outData);
    cInData=pyvector_to_Carrayptrs(inData);
    cOldData=pyvector_to_Carrayptrs(oldData);
    
    
    //Some local variables for speed up
    unsigned int inDataSize = inData->dimensions[0];

    
    for(unsigned int i=0; i<inDataSize; i++)
    {
        switch(1){
            case 0:
            cOutData[i] = fabs(pow(cOldData[1],2.0) - (cOldData[0] * cInData[0]));
            break;
            
            case 1:
            cOutData[i] = fabs(pow(cInData[0],2.0) - (cOldData[1] * cInData[1]));
            break;
            
            default:
            cOutData[i] = fabs(pow(cInData[i-1],2.0) - (cInData[i-2] * cInData[i]));
            break;
        }
    }
    
    cOldData[0] = cInData[inDataSize-2];
    cOldData[1] = cInData[inDataSize-1];
    
    return Py_BuildValue("i", 1);
    
}



/* Function that calculates the standadtization of the given timeseries
   All operations are done in place.
   Params:
            - outData       Array containing the results of the operation
            - inData        Array containing the input timeseries
            - ringbuffer    Array with the n last samples needed for calculation
            - variables     Array containing the last variance and the last mean
            - width         Length of the variance filter (how many samples are used to calculate the variance)
            - index         Index for the current needed sample of the ringbuffer
    Return:
            - index         Returns the updated index for the ringbuffer

*/


static PyObject* standardization(PyObject *self, PyObject *args){
    // Passed variables
    PyArrayObject *outData, *inData, *ringbuffer, *variables;
    int width, index;
    
    // Pointer for converted NumPy Arrays
    double *cOutData, *cInData, *cRingbuffer, *cVariables;

    // Extract variables and arrays from ARGS
    if (!PyArg_ParseTuple(args, "O!O!O!O!ii", &PyArray_Type, &outData, &PyArray_Type, &inData,
                                              &PyArray_Type, &ringbuffer, &PyArray_Type, &variables,
                                              &width, &index)) return NULL;
    
    // Check wheter all array are of type double
    if (not_doublevector(outData)) return NULL;
    if (not_doublevector(inData)) return NULL;
    if (not_doublevector(ringbuffer)) return NULL;
    if (not_doublevector(variables)) return NULL;
    
    // Convert NumPy array to C-Arrays
    cOutData=pyvector_to_Carrayptrs(outData);
    cInData=pyvector_to_Carrayptrs(inData);
    cRingbuffer=pyvector_to_Carrayptrs(ringbuffer);
    cVariables=pyvector_to_Carrayptrs(variables);
    
    
    //Some local variables for speed up
    unsigned int inDataSize = inData->dimensions[0];
    double ww = width*width;
    double wm1 = width-1;
    double wp1 = width+1;

    double ringbufferValue=0.0;
    double inDataValue=0.0;
    double variable1 = 0.0;
    
    bool invalidValue = false;
    bool invalidValueOccured = false;
    double invalidValueBuffer = 0.0;
    
    for(unsigned int i=0; i<inDataSize; i++)
    {
        //Speedup for array entries which are needed several times
        ringbufferValue = cRingbuffer[index];
        inDataValue = cInData[i];
        variable1 = cVariables[1];

        //Calculating the new variance
        cVariables[0] = cVariables[0] + (inDataValue - ringbufferValue) * ( ((wm1) * inDataValue) + ((wp1) * ringbufferValue) - (2.0*variable1));

        if(cVariables[0] <= 0.0)
        {
        	invalidValue = true;
        	invalidValueOccured = true;
        	invalidValueBuffer = cVariables[0];
        	cVariables[0] = 1.0;
        }

        //Calculating the new mean value
        cVariables[1] = variable1 + (inDataValue-ringbufferValue);

        //Store the actual sample in the ringbuffer
        cRingbuffer[index] = inDataValue;

        //Increment the ringbuffer index
        index = (index < wm1 ? index+1:0);
                
        //Calculate the standadization
        cOutData[i] =( sqrt(cVariables[0]/(ww)) != 0.0 ? (inDataValue- (cVariables[1]/(width))) / sqrt(cVariables[0]/(ww)):0.0);

        if(invalidValue)
        {
        	invalidValue = false;
        	cVariables[0] = invalidValueBuffer;
        }   
    }
    
    if(invalidValueOccured)
    {
    	perror("OnlineStandardization:: Warning: Prevented division by zero during standardization\n");
    }
    
    return Py_BuildValue("i", index);
    
}

/* Function that calcualtes values of an adaptive threshold
   for an given time series. the Formula is given as:

   T(t) = mean_n(t) + p * std_n(t)
   ,where t is the timepoint, T the Threshold, mean_n the
   mean value of sample t and the n previous one, std_n
   the standard deviation of sample t and the n previous one,
   and p the sensitivity factor.

   Reference: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=06070974

   All operations are done in place.
   Params:
            - outData       Array containing the results of the operation
            - inData        Array containing the input timeseries
            - ringbuffer    Array with the n last samples needed for calculation
            - variables     Array containing the last variance and the last mean
            - width         Length of the variance filter (how many samples are used to calculate the variance)
            - index         Index for the current needed sample of the ringbuffer
            - p                 p-value of the adaptive threshold
   Return:
            - index         Returns the updated index for the ringbuffer
*/


static PyObject* adaptive_threshold(PyObject *self, PyObject *args){
    // Passed variables
    PyArrayObject *outData, *inData, *ringbuffer, *variables;
    int width, index, p;
    
    // Pointer for converted NumPy Arrays
    double *cOutData, *cInData, *cRingbuffer, *cVariables;

    // Extract variables and arrays from ARGS
    if (!PyArg_ParseTuple(args, "O!O!O!O!iii", &PyArray_Type, &outData, &PyArray_Type, &inData,
                                              &PyArray_Type, &ringbuffer, &PyArray_Type, &variables,
                                              &width, &index, &p)) return NULL;
    
    // Check wheter all array are of type double
    if (not_doublevector(outData)) return NULL;
    if (not_doublevector(inData)) return NULL;
    if (not_doublevector(ringbuffer)) return NULL;
    if (not_doublevector(variables)) return NULL;
    
    // Convert NumPy array to C-Arrays
    cOutData=pyvector_to_Carrayptrs(outData);
    cInData=pyvector_to_Carrayptrs(inData);
    cRingbuffer=pyvector_to_Carrayptrs(ringbuffer);
    cVariables=pyvector_to_Carrayptrs(variables);
    
    
    //Some local variables for speed up
    unsigned int inDataSize = inData->dimensions[0];
    double ww = width*width;
    double wm1 = width-1;
    double wp1 = width+1;

    double ringbufferValue=0.0;
    double inDataValue=0.0;
    double variable1 = 0.0;
    
    
    
    for(unsigned int i=0; i<inDataSize; i++)
    {
        //Speedup for array entries which are needed several times
        ringbufferValue = cRingbuffer[index];
        inDataValue = cInData[i];
        variable1 = cVariables[1];

        //Calculating the new variance
        cVariables[0] = cVariables[0] + (inDataValue - ringbufferValue) * ( ((wm1) * inDataValue) + ((wp1) * ringbufferValue) - (2.0*variable1));

        //Calculating the new mean value
        cVariables[1] = variable1 + (inDataValue-ringbufferValue);

        //Store the actual sample in the ringbuffer
        cRingbuffer[index] = inDataValue;

        //Increment the ringbuffer index
        index = (index < wm1 ? index+1:0);

        //Write the variance to the output array
        cOutData[i] = (double)(cVariables[1]/width) + ((double)p * sqrt(cVariables[0]/(ww)));
    }
    
    return Py_BuildValue("i", index);
    
}








// method definitions for this module
static PyMethodDef variance_toolsMethods[] = {
    {"filter", variance_filter, METH_VARARGS,"Variance filter"},
    {"filterSqrt", standard_filter, METH_VARARGS,"Standarddeviation filter"},
    {"standardization", standardization, METH_VARARGS,"Standartization"},
    {"adaptive_threshold", adaptive_threshold, METH_VARARGS,"Adaptive Threshold"},
    {"tkeo_filter", tkeo_filter, METH_VARARGS,"Teager Kaiser Energy Operator"},
    {NULL, NULL, 0, NULL}
};

// the required init function (gets called 
// once during the import statement in python)
PyMODINIT_FUNC initvariance_tools(void)
{   
    static PyObject* var_error;
    PyObject *m;
    m = Py_InitModule("variance_tools", variance_toolsMethods);
    import_array();
    char Acception[] = "variance_tools.AccessException";
    var_error = PyErr_NewException(Acception, NULL, NULL);
    Py_INCREF(var_error);
    PyModule_AddObject(m, "AccessException", var_error);
}

