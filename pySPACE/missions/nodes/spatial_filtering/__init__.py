""" Erase and/or recombine channels of multivariate :class:`~pySPACE.resources.data_types.time_series.TimeSeries`

Spatial Filtering subsumes methods, that can be used to extract the most 
relevant information contained in a number of channels and create a number
of pseudo channels, that are (linear) combinations of the former channels.

Typically, spatial filtering requires a training phase in which a 
number of training examples is presented and a model is created 
that is later on used to concentrate the relevant information contained
in the signal in a small number of pseudo channels.
There are both unsupervised and supervised spatial filtering methods.
"""

