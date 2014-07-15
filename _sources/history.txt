.. _history:

History of pySPACE
==================

pySPACE was founded in December 2012 as a unification and renewal of
its two predecessors BOR (``Benchmarking on Rails``) and aBRI
(``adaptive Brain Reading Interface``). These two frameworks
originated during the `VI-Bot project <http://robotik.dfki-bremen.de/en/research/projects/vi-bot.html>`_
in 2008 at the `German Research Center for Artificial Intelligence (DFKI GmbH),
Robotics Innovation Center, Bremen (DFKI RIC) <http://robotik.dfki-bremen.de/en/>`_.
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

With the ending of the project `VI-Bot <http://robotik.dfki-bremen.de/en/research/projects/vi-bot.html>`_
and the beginning of the project
`IMMI <http://robotik.dfki-bremen.de/en/research/projects/immi.html>`_
at the DFKI RIC and the
`Robotics Group at the University of Bremen <http://www.informatik.uni-bremen.de/robotik/index_en.php>`_
in May 2010,
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

Publications
============

pySPACE and its predecessors have been used in several publications as listed
below. If you have your own publication using pySPACE please tell us,
such that we can complete it.

main pySPACE paper:

  | Mario Michael Krell, Sirko Straube, Anett Seeland, Hendrik Wöhrle, Johannes Teiwes, Jan Hendrik Metzen, Elsa Andrea Kirchner, Frank Kirchner (2013)
  | `pySPACE — A Signal Processing and Classification Environment in Python <http://www.frontiersin.org/Neuroinformatics/10.3389/fninf.2013.00040/abstract>`_
  | In Frontiers in Neuroinformatics 7(40), doi: 10.3389/fninf.2013.00040, in press

Journal Publications
--------------------

comparison of evaluation metrics for classification on data with imbalanced class ratio:

  | Sirko Straube, Mario Michael Krell (2014),
  | `How to evaluate an agent's behaviour to infrequent events? — Reliable performance estimation insensitive to class distribution <http://dx.doi.org/10.3389/fncom.2014.00043>`_,
  | In Frontiers in Computational Neuroscience 8(43), doi:10.3389/fncom.2014.00043

evaluations of Balanced Relative Margin Machine (BRMM) classifier:

  | Mario Michael Krell, David Feess, Sirko Straube (2014)
  | `Balanced Relative Margin Machine - The Missing Piece Between FDA and SVM Classification <http://dx.doi.org/10.1016/j.patrec.2013.09.018>`_
  | In Pattern Recognition Letters, Elsevier, doi: 10.1016/j.patrec.2013.09.018, in press

comparison of eye artifact removal methods:

  | Foad Ghaderi, Su Kyoung Kim, Elsa Andrea Kirchner (2014)
  | `Effects of eye artifact removal methods on single trial P300 detection, a comparative study <http://www.sciencedirect.com/science/article/pii/S0165027013003038>`_
  | In Journal of Neuroscience Methods 221(0): 41-47, doi: 10.1016/j.jneumeth.2013.08.025

applications for brain-computer interfaces:

  | Elsa Andrea Kirchner, Su Kyoung Kim, Sirko Straube, Anett Seeland, Hendrik Wöhrle, Mario Michael Krell, Marc Tabie, Manfred Fahle (2013)
  | `On the Applicability of Brain Reading for Self-Controlled, Predictive Human-Machine Interfaces in Robotics <http://dx.plos.org/10.1371/journal.pone.0081732>`_
  | In PLoS ONE 8(12): e81732, doi:10.1371/journal.pone.0081732

large scale evaluation and comparison of sensor selection algorithms:

  | David Feess, Mario Michael Krell\*, Jan Hendrik Metzen (2013)
  | `Comparison of Sensor Selection Mechanisms for an ERP-Based Brain-Computer Interface <http://dx.plos.org/10.1371/journal.pone.0067543>`_
  | In PLoS ONE 8(7): e67543, doi:10.1371/journal.pone.0067543

Conference Publications
-----------------------

application in movement prediction with :mod:`pySPACE live<pySPACE.run.launch_live>`:

  | Anett Seeland, Hendrik Wöhrle, Sirko Straube, Elsa Andrea Kirchner (2013)
  | `Online Movement Prediction in a Robotic Application Scenario`
  | In 6th International IEEE EMBS Conference on Neural Engineering (NER): 41-44

evaluation of online classifiers:

  | Hendrik Wöhrle, Johannes Teiwes, Mario Michael Krell, Elsa Andrea Kirchner, Frank Kirchner (2013)
  | `A Dataflow-Based Mobile Brain Reading System on Chip with Supervised Online Calibration`
  | In International Congress on Neurotechnology, Electronics and Informatics, (NEUROTECHNIX-2013), SciTePress Digital Library

comparison of different classification score postprocessing methods for movement prediction:

  | Sirko Straube, Anett Seeland, David Feess (2013)
  | `Striving for better and earlier movement prediction by postprocessing of classification scores`
  | In International Congress on Neurotechnology, Electronics and Informatics, (NEUROTECHNIX-2013), SciTePress Digital Library

application in brain-computer interface for exoskeleton control:

  | Elsa Andrea Kirchner, Jan Albiez, Anett Seeland, Mathias Jordan, Frank Kirchner (2013)
  | `Towards Assistive Robotics for Home Rehabilitation`
  | In Proceedings of the 6th International Conference on Biomedical Electronics and Devices (BIODEVICES-13), SciTePress, 168-177

error potential detection for brain-computer interface:

  | Su Kyoung Kim, Elsa Andrea Kirchner (2013)
  | `Classifier Transferability in the Detection of Error Related Potentials from Observation to Interaction`
  | In Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics

evaluation of adaptive periodic spatial filter (PiSF):

  | Foad Ghaderi, Sirko Straube (2013)
  | `An adaptive and efficient spatial filter for event-related potentials`
  | In Proceedings of European Signal Processing Conference (EUSIPCO)

evaluation of periodic spatio spectral filter:

  | Foad Ghaderi (2013)
  | `Joint spatial and spectral filter estimation for single trial detection of Event Related potentials <http://dx.doi.org/10.1109/MLSP.2013.6661938>`_
  | In IEEE International Workshop on Machine Learning for Signal Processing (MLSP)

first paper about the periodic spatial filter (PiSF)

  | Foad Ghaderi, Elsa Andrea Kirchner, 2013
  | `Periodic Spatial Filter for Single Trial Classification of Event Related Brain Activity <http://dx.doi.org/10.2316/P.2013.791-110>`_
  | In Proceedings of the 10th IASTED International Conference on Biomedical Engineering (BioMed-2013)

classification in compressed space:

  | Yohannes Kassahun, Hendrik Wöhrle, Alexander Fabisch, Marc Tabie (2012)
  | `Learning Parameters of Linear Models in Compressed Parameter Space <http://dx.doi.org/10.1007/978-3-642-33266-1_14>`_
  | In Artificial Neural Networks and Machine Learning – ICANN 2012, Lecture Notes in Computer Science: 108-115, doi: 978-3-642-33266-1_14

comparison of different downsampling methods and band pass filters for LRP:

  | Michele Folgheraiter, Elsa Andrea Kirchner, Anett Seeland, Su Kyoung Kim, Mathias Jordan, Hendrik Wöhrle, Bertold Bongardt, Steffen Schmidt, Jan Albiez, Frank Kirchner (2011)
  | `A multimodal brain-arm interface for operation of complex robotic systems and upper limb motor recovery`
  | In Proceedings of the 4th International Conference on Biomedical Electronics and Devices (BIODEVICES-11): 150-162

analysis of transferability of spatial filters:

  | Jan Hendrik Metzen, Su Kyoung Kim, Timo Duchrow, Elsa Andrea Kirchner, Frank Kirchner (2011)
  | `On Transferring Spatial Filters in a Brain Reading Scenario <http://dx.doi.org/10.1109/SSP.2011.5967825>`_
  | In Proceedings of the 2011 IEEE Workshop on Statistical Signal Processing: 797-800, doi: 10.1109/SSP.2011.5967825

:class:`threshold adaptation <pySPACE.missions.nodes.postprocessing.threshold_optimization.ThresholdOptimizationNode>`:

  | Jan Hendrik Metzen, Elsa Andrea Kirchner (2011)
  | `Rapid Adaptation of Brain Reading Interfaces based on Threshold Adjustment <http://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0CDkQFjAA&url=http%3A%2F%2Fwww.informatik.uni-bremen.de%2F~jhm%2Ffiles%2Fgfkl_2011_presentation.pdf&ei=7ROvUuntN8qStQbMyIDACA&usg=AFQjCNFITKvnEoH5lYorrnPEnwF_k7Bkbw&sig2=1SMphFKPh6ofdbFslFojBw&bvm=bv.57967247,d.Yms&cad=rja>`_
  | In Proceedings of the 2011 Conference of the German Classification Society (GfKl-2011): 138

ensemble classification for brain-computer interface:

  | Jan Hendrik Metzen, Su Kyoung Kim, Elsa Andrea Kirchner (2011)
  | `Minimizing Calibration Time for Brain Reading <http://dx.doi.org/10.1007/978-3-642-23123-0_37>`_
  | In *Pattern Recognition, Lecture Notes in Computer Science* 6835: 366-375, doi: 10.1007/978-3-642-23123-0_37

application in Brain Reading:

  | Elsa Andrea Kirchner, Hendrik Wöhrle, Constantin Bergatt, Su Kyoung Kim, Jan Hendrik Metzen, David Feess, Frank Kirchner (2010)
  | `Towards Operator Monitoring via Brain Reading - An EEG-based Approach for Space Applications <http://www.dfki.de/web/research/ric/publications/renameFileForDownload?filename=110722_Towards%20Operator%20Monitoring%20via%20Brain%20Reading%20-%20An%20EEG-based%20Approach_iSAIRAS_EKirchner.pdf&file_id=uploads_1087>`_
  | In Proceedings of the 10th International Symposium on Artificial Intelligence, Robotics and Automation in Space: 448-455

Other Publications
------------------

pySPACE has been presented at the NIPS2013 workshop *Machine Learning Open Source Software: Towards Open Workflows*.
