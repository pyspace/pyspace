#include <boost/python.hpp>
#include <pyublas/numpy.hpp>

#include <math.h>
#include <stdio.h>
using namespace boost::python;

/* Function that filters the given data using the variance
   All operations are done in place.
   Params:
            - outData       Array containing the results of the operation
            - inData        Array containing the input timeseries
            - ringbuffer    Array with the n last samples needed for calculation
            - variables     Arra containing the last variance and the last mean
            - width         Length of the variance filter (how many samples are used to calculate the variance)
            - index         Index for the current needed sample of the ringbuffer
    Return:
            - index         Returns the updated index for the ringbuffer

*/
int variance_filter(pyublas::numpy_vector<double> outData, pyublas::numpy_vector<double> inData, pyublas::numpy_vector<double> ringbuffer, pyublas::numpy_vector<double> variables, int width, int index)
{
    //Some local variables for speed up
    unsigned int inDataSize = inData.size();
    double ww = width*width;
    double wm1 = width-1;
    double wp1 = width+1;

    double ringbufferValue=0.0;
    double inDataValue=0.0;
    double variable1 = 0.0;

    for(unsigned int i=0; i<inDataSize; i++)
    {
        //Speedup for array entries which are needed several times
        ringbufferValue = ringbuffer[index];
        inDataValue = inData[i];
        variable1 = variables[1];

        //Calculating the new variance
        variables[0] = variables[0] + (inDataValue - ringbufferValue) * ( ((wm1) * inDataValue) + ((wp1) * ringbufferValue) - (2.0*variable1));

        //Calculating the new mean value
        variables[1] = variable1 + (inDataValue-ringbufferValue);

        //Store the actual sample in the ringbuffer
        ringbuffer[index] = inDataValue;

        //Increment the ringbuffer index
        index = (index < wm1 ? index+1:0);

        //Write the variance to the output array
        outData[i] = variables[0]/(ww);
    }

    return index;
}

/* Function that filters the given data using the standard deviation
   All operations are done in place.
   Params:
            - outData       Array containing the results of the operation
            - inData        Array containing the input timeseries
            - ringbuffer    Array with the n last samples needed for calculation
            - variables     Arra containing the last variance and the last mean
            - width         Length of the standard deviation filter (how many samples are used to calculate the standard deviation)
            - index         Index for the current needed sample of the ringbuffer
    Return:
            - index         Returns the updated index for the ringbuffer

*/
int standard_filter(pyublas::numpy_vector<double> outData, pyublas::numpy_vector<double> inData, pyublas::numpy_vector<double> ringbuffer, pyublas::numpy_vector<double> variables, int width, int index)
{
    //Some local variables for speed up
    unsigned int inDataSize = inData.size();
    double ww = width*width;
    double wm1 = width-1;
    double wp1 = width+1;

    double ringbufferValue=0.0;
    double inDataValue=0.0;
    double variable1 = 0.0;

    for(unsigned int i=0; i<inDataSize; i++)
    {
        //Speedup for array entries which are needed several times
        ringbufferValue = ringbuffer[index];
        inDataValue = inData[i];
        variable1 = variables[1];

        //Calculating the new variance
        variables[0] = variables[0] + (inDataValue - ringbufferValue) * ( ((wm1) * inDataValue) + ((wp1) * ringbufferValue) - (2.0*variable1));

        //Calculating the new mean value
        variables[1] = variable1 + (inDataValue-ringbufferValue);

        //Store the actual sample in the ringbuffer
        ringbuffer[index] = inDataValue;

        //Increment the ringbuffer index
        index = (index < wm1 ? index+1:0);

        //Write the variance to the output array
        outData[i] = sqrt(variables[0]/(ww));
    }

    return index;
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
            - variables     Arra containing the last variance and the last mean
            - width         Length of the standard deviation filter (how many samples are used to calculate the standard deviation)
            - index         Index for the current needed sample of the ringbuffer
            - p             p-value of the adaptive threshold
    Return:
            - index         Returns the updated index for the ringbuffer

*/

int calc_threshold(pyublas::numpy_vector<double> outData,
                   pyublas::numpy_vector<double> inData,
                   pyublas::numpy_vector<double> ringbuffer,
                   pyublas::numpy_vector<double> variables,
                   int width,
                   int index,
                   double p,
                   pyublas::numpy_vector<int> rest_time,
                   pyublas::numpy_vector<int> movement_state,
                   pyublas::numpy_vector<double> last_onset,
                   int channel_index,
                   int min_rest_time,
                   int skip)
{
    //Some local variables for speed up
    unsigned int inDataSize = inData.size();
    double ww = width*width;
    double wm1 = width-1;
    double wp1 = width+1;

    double ringbufferValue=0.0;
    double inDataValue=0.0;
    double variable1 = 0.0;

    for(unsigned int i=0; i<inDataSize; i++)
    {
        //Speedup for array entries which are needed several times
        ringbufferValue = ringbuffer[index];
        inDataValue = inData[i];
        variable1 = variables[1];

        //Calculating the new variance
        variables[0] = variables[0] + (inDataValue - ringbufferValue) * ( ((wm1) * inDataValue) + ((wp1) * ringbufferValue) - (2.0*variable1));

        //Calculating the new mean value
        variables[1] = variable1 + (inDataValue-ringbufferValue);

        //Store the actual sample in the ringbuffer
        ringbuffer[index] = inDataValue;

        //Increment the ringbuffer index
        index = (index < wm1 ? index+1:0);

        if(skip != 1){
            if(movement_state[channel_index] == 1){
                if(inDataValue<=last_onset[channel_index]){
                    rest_time[channel_index] += 1;
                    if(rest_time[channel_index] > min_rest_time){
                        movement_state[channel_index] = 0;
                    }
                }else{
                    rest_time[channel_index] = 0;
                }
                outData[i] = -1.0;
            }else{
                if( (inDataValue - ((variables[1]/ww) + ( p*sqrt( variables[0]/(ww))))) >= 0.0 && rest_time[channel_index] > min_rest_time){
                    outData[i] = 1.0;
                    last_onset[channel_index] = inDataValue;
                    rest_time[channel_index] = 0;
                    movement_state[channel_index] = 1;
                }else{
                    outData[i] = -1.0;
                }
            }
        }else{
            outData[i] = -1.0;
        }
    }

    return index;
}



/* Function that calculates the standadtization of the given timeseries
   All operations are done in place.
   Params:
            - outData       Array containing the results of the operation
            - inData        Array containing the input timeseries
            - ringbuffer    Array with the n last samples needed for calculation
            - variables     Arra containing the last variance and the last mean
            - width         Length of the variance filter (how many samples are used to calculate the variance)
            - index         Index for the current needed sample of the ringbuffer
    Return:
            - index         Returns the updated index for the ringbuffer

*/

int standardization(pyublas::numpy_vector<double> outData, pyublas::numpy_vector<double> inData, pyublas::numpy_vector<double> ringbuffer, pyublas::numpy_vector<double> variables, int width, int index)
{
    //Some local variables for speed up
    unsigned int inDataSize = inData.size();
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
        ringbufferValue = ringbuffer[index];
        inDataValue = inData[i];
        variable1 = variables[1];

        //Calculating the new variance
        variables[0] = variables[0] + (inDataValue - ringbufferValue) * ( ((wm1) * inDataValue) + ((wp1) * ringbufferValue) - (2.0*variable1));

        if(variables[0] <= 0.0)
        {
        	invalidValue = true;
        	invalidValueOccured = true;
        	invalidValueBuffer = variables[0];
        	variables[0] = 1.0;
        }

        //Calculating the new mean value
        variables[1] = variable1 + (inDataValue-ringbufferValue);

        //Store the actual sample in the ringbuffer
        ringbuffer[index] = inDataValue;

        //Increment the ringbuffer index
        index = (index < wm1 ? index+1:0);

        //Calculate the standadization
        outData[i] =( sqrt(variables[0]/(ww)) != 0.0 ? (inDataValue- (variables[1]/(width))) / sqrt(variables[0]/(ww)):0.0);

        if(invalidValue)
        {
        	invalidValue = false;
        	variables[0] = invalidValueBuffer;
        }
    }

    if(invalidValueOccured)
    {
    	perror("OnlineStandardization:: Warning: Prevented division by zero during standardization\n");
    }

    return index;
}

// The gate to python
// Definition of the module name and its functions

BOOST_PYTHON_MODULE(variance_tools)
{
    def("filter", variance_filter);
    def("filterSqrt", standard_filter);
    def("calc_threshold", calc_threshold);
    def("standardization", standardization);
}
