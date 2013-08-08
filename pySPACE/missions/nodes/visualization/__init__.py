""" Visualize the single different data samples or averages

All visualization nodes are zero-processing nodes, i.e. their execute method
returns exactly the data that it gets as parameter. However, when the data
is passed through the visualization node, it performs different kinds of 
analysis and creates some plots of the data. In principle, visualization nodes
can be plugged between two arbitrary other nodes; 
however some nodes expect that the data contains some meta-information like 
channel- or feature names.

Many of the nodes are trainable even though don't really learn a model (they
don't process the data anyway). The reason for that is that they require 
information about the class labels for creating the plots.
"""
