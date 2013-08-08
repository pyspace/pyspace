""" Signal processing data types (time series, feature vector, prediction vector)

According to the conventions given by MDP,  all data types are two dimensional
numpy arrays.

The data is exchanged between the :mod:`~pySPACE.missions.nodes>`
via the
:mod:`~pySPACE.environments.chains.node_chain` module.
For loading there is need of a :mod:`~pySPACE.missions.nodes.source`,
for saving a :mod:`~pySPACE.missions.nodes.sink` node.
For loading and saving these build the interface to
:mod:`datasets <pySPACE.resources.dataset_defs>`,
where the data is collected and saved in certain datasets or loaded.



The BaseData type plays a special role, because it is a superclass
for every other data type. The main role of this module is to
equip each data type with the ability to store information beyond each node
(if this is intended), i.e. it circumvents the original philosophy of the 
MDP framework. This behaviour is invisible for the user as long as it is not
used.
"""