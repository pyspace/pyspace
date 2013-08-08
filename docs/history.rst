.. _history:

History of pySPACE
==================

pySPACE was founded in December 2012 as a unification and renewal of
its two predecessors BOR (``Benchmarking on Rails``) and aBRI
(``adaptive Brain Reading Interface``). These two frameworks
originated during the `VI-Bot project <http://robotik.dfki-bremen.de/de/forschung/projekte/vi-bot.html>`_
in 2008.
Still today these two foundations of pySPACE are clearly visible in its functionality.

Both, aBRI and BOR, were started by Jan Hendrik Metzen and Timo Duchrow.
The task of BOR was to provide parallelization of independent processes to
enable quick comparison of algorithms and parameters on large scales.
For this, interfaces to other frameworks such as WEKA and MMLF were implemented
(so called BOR operations) and BOR distributed the task when possible and intended by
the user. The parallelization part, the pySPACE environment and interfacing using operations in pySPACE 
therefore has its roots in the BOR framework.

The task of aBRI was to provide quick and modular signal processing and it was
purely used with EEG data (EEG=electroencephalogram). That's where its name came from:
the application. At the beginning aBRI was a wrapper around the MDP framework, but
soon many own node implementations were implemented and large parts of the MDP functionality
overwritten by own, better fitting functions. 
The main differences to MDP were that data in aBRI were two-dimensional
and aBRI was processing each single data instance
and did unlike MDP not include a whole dataset at once. aBRI had no own
execution part, input and output of the data were defined elsewhere. That is
why aBRI was inseparably connected with BOR from its very beginning. BOR took
over the execution part of aBRI for offline analysis of the data. Another execution
mode possible in aBRI was an online mode, where data could be directly streamed and
processed.
So, in pySPACE the signal processing capability, the node concept with 
its familiarity to MDP and the online mode (which is now pySPACE-live) comes from aBRI.

With the ending of the project VI-Bot and the beginning of the project
`IMMI <http://robotik.dfki-bremen.de/de/forschung/projekte/immi.html>`_ in May 2010,
more and more nodes and operations were added, classification algorithms introduced to
aBRI, and documentation and usability enhanced. Both frameworks were more and more connected
and structure and style were constantly changing (even diverging in some parts).
More functionality, like adaptivity, became a real part of the software in IMMI and
the backend to the cluster software LoadLeveler was implemented.

The idea of unifying both frameworks under what is now pySPACE came into play,
because we decided at some point to have a complete software with full execution
support which still has the property of being modular and usable from outside its 
own environment. We took the opportunity to create an easy and transparent structure 
for both, users and developers, and now can make full use of all features of both frameworks
and more.
In the process of creating pySPACE, both structures were merged and completely
rearranged. Furthermore, pySPACE is now fully independent of MDP which simplifies
future developments and integration of new components.

We hope, you like our child.