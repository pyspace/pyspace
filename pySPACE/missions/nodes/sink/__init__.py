""" Collect incoming signal types for further processing or to store in :mod:`datasets <pySPACE.resources.dataset_defs>`

These nodes can only be found at the end of a node chain.
For access to the collected data, the method *get_result_dataset*
has to be implemented.
The existence of this method is also used, to detect, if a node
is a sink node.
The access to the result data is done in the method
*process_current_split*, which is called by the benchmark node chain.
"""