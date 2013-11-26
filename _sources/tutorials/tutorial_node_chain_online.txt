.. _tutorial_node_chain_online:

Process EEG data with the Live Environment
------------------------------------------------------

In this tutorial we will learn how to use the live-environment to
train the node chains on the data streamed from an external EEG server and
classify new instances based on the trained node chains.



Preparation
===========

    Before we start we have to look more closely to the content and the structure of
    the example parameter file for the live environment.
    
    .. literalinclude:: ../examples/specs/live_settings/example_param_file.yaml
        :language: yaml
        
    * data_files
        This files can be left empty for testing purposes.
    
    * potentials
        This field describes every flow which is executed during the live processing.
        In this example there is only one flow called *SineWave*. It uses a dataset which 
        is encoded in the raw-eeg format and is located in the default storage directory.
        Specific node chains for the processing of the data are referenced for the
        different processing steps (e.g. prewindowing, postprocessing).
    
    * eeg_server
        Here you can fill in the parameters for a online recording/processing session.
        The setup of the online streaming is described in the 
        tutorial :ref:`tutorial_work_with_the_eegserver`.
    
    * live_server
        ???
        
    * mars
        ???
        
    * flow_persistency_directory
        The directory, in which the pickled flows are stored.
    
    * prewindowed_data_directory
        The directory where the data after the prewindowing step is stored.
        
    * record 
        When using pySPACE-live the data which is processed also gets recorded in
        the raw-eeg format (.eeg/.vhdr/.vmrk) for later reference or analysis. The
        field *subject* is meant for any subject related identification. The
        *experiment* field is there to describe the experiment. Both fileds are
        used to create the filename of the resulting raw-data files. 
        The data is saved to the *storage* directory which is specified in 
        your spec-file.
        
    For purposes, other than testing the pySPACE-live environment with artificial 
    data, please feel free to adapt it to your need (e.g. fill in more potentials or
    node chains).
        
Execute the Test for pySPACE-live
==================================

    To make sure, everything is well defined and configured, we execute
    the unit-test for the pySPACE-live environment. 
    In order to do that navigate to the folder
    
    .. code-block:: bash

        cd $(pyspace)/pySPACE/tests/system_tests/live
        
    In there you will find *test_live.py*
    Execute the test by invoking the following command:
    
    .. code-block:: bash

        python test_live.py --param example_param_file.yaml --conf <your-config>.yaml
        
    After that you will see the output of the different steps which
    are preformed in pySPACE-live.
    
    
    

    

