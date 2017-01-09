""" Example module that can be used with the :mod:`~pySPACE.missions.operations.generic` operation.

This module shows how the mandelbrot set can be computed in a distributed 
fashion. For this, it is required that two module-level
functions are defined: "process" and "consolidate". The actual subpart of 
the computation is  performed by the function "process" (which is called once
for each process) and the function "consolidate" is called once at the end
of an operation and is responsible for merging the results of the
sub-computations into the the overall result.
    
:Author:  Jan Hendrik Metzen
:Created: 2012/11/30
"""

import os
import cPickle
import glob

from numpy import ogrid, zeros, conj

import numpy

def process(process_id, result_directory, config, parameter_setting):
    """ Perform computation for specific configuration and store results.
    
    This function is called once per process for the specific configuration.
    
    **Parameters**
        :process_id:
            Globally unique id of this process. Might be used e.g. for
            creating a file into which the results of this function are stored.
            
            (*obligatory*)
            
        :result_directory:
            Directory into which the results of the computation of this
            function are stored.
            
            (*obligatory*)
            
        :config:
            Configuration parameters. All parameters which parametrize the 
            actual computation MUST go into this dictionary.
            
            (*obligatory*)
            
        :parameter_setting:
            Dictionary containing the mapping from parameter name (must be 
            contained in configuration_template string) to parameter value.
            The specific parameter values define this particular computation.
            
            (*obligatory*)
    """
    # This function needs to be implemented! Replace the following lines 
    # by your own implementation!
    
    # Example showing the computation of the mandelbrot set where different
    # rectangular patches of the set are computed in different processes
    # Code is based on http://www.scipy.org/Tentative_NumPy_Tutorial/Mandelbrot_Set_Example
    def mandelbrot(xind, yind, xstep, ystep, h, w, maxit=50):
        """Returns an image of the Mandelbrot fractal of size (h,w)."""
        y,x = ogrid[(-1.4+yind*ystep):(-1.4+(yind+1)*ystep):(h*1j),
                    (-2.0+xind*xstep):(-2.0+(xind+1)*ystep):(w*1j)]
        c = x+y*1j
        z = c
        divtime = maxit + zeros(z.shape, dtype=numpy.int16)
        for i in xrange(maxit):
            z  = z**2 + c
            diverge = z*conj(z) > 2**2            # who is diverging
            div_now = diverge & (divtime==maxit)  # who is diverging now
            divtime[div_now] = i                  # note when
            z[diverge] = 2                        # avoid diverging too much

        return divtime
        
    # Compute subpart of the mandelbrot set
    mb_part = mandelbrot(config["xind"], config["yind"], config["xstep"], 
                         config["ystep"], h=config["resolution"],
                         w=config["resolution"])
        
    # Serialize the result
    f = open(result_directory + os.sep + "%s.pickle" % process_id, 'w')
    cPickle.dump((parameter_setting, mb_part), f, 
                 cPickle.HIGHEST_PROTOCOL)
    f.close()                               
    
def consolidate(result_directory, config_template):
    """ Consolidates results of single processes.
    
    This function is called once per operation.
    
    **Parameters**
        :result_directory:
            Directory into which the results of the computation of this
            function are stored.
            
            (*obligatory*)

        :config_template:
            Configuration parameters template which might include placeholders
            instead of real parameters.

            (*obligatory*)

    """
    # This function needs to be implemented! Replace the following lines 
    # by your own implementation!
    
    # Merging subslices of the mandelbrot set
    resolution = config_template["resolution"]
    
    subresults = glob.glob(result_directory + os.sep + "*.pickle")
    mb = zeros((len(subresults)**.5*resolution, len(subresults)**.5*resolution), 
                dtype=numpy.int16)
    
    for subresult in subresults:
        parameter_setting, mb_part = cPickle.load(open(subresult))
        os.remove(subresult)
        
        xind = parameter_setting["__XIND__"]
        yind = parameter_setting["__YIND__"]
        mb[xind*resolution:(xind+1)*resolution,
           yind*resolution:(yind+1)*resolution] = mb_part.T
    
    # Plot and store set as graphic
    from scipy import misc
    misc.imsave(result_directory + os.sep + "mb.pdf", mb)


# Note that we could run the script easily without using pySPACE; however, this
# would become impractical for higher resolutions
if __name__ == "__main__":
    result_directory = __import__("tempfile").mkdtemp()
    
    process_id = 0
    for xind in [0, 1, 2, 3, 4, 5, 6]:
        for yind in [0, 1, 2, 3, 4, 5, 6]:
            config = {"xind" : xind, "yind" : yind, "xstep" : 0.4, 
                      "ystep" : 0.4, "resolution" : 200}
            process(process_id, result_directory, config, 
                    {"__XIND__": xind, "__YIND__": yind})
            
            process_id += 1
            
    consolidate(result_directory, {"resolution": 200})
    
    print "Result is stored in directory: %s" % result_directory
    
