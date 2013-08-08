""" Execute MMLF experiments

This module contains an operation and a process for MMLF. One MMLF
process consists of a executing one run of a predefined number of
episodes of a MMLF world. An MMLF operation consists of one or several runs of
one or several MMLF worlds.

.. note:: The results of the runs are not stored in the standard operation
    result directories but in the MMLF ReadWrite Area. Only the outputs of the
    MMLF to stdout are stored in log files stored in the operation result
    directories.

.. todo:: documentation link to MMLF-framework

:Author: Jan-Hendrik Metzen

"""

import sys
import os
import time
import csv
import xml.etree.ElementTree as ElementTree
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    import processing
else:
    import multiprocessing as processing

import pySPACE
from pySPACE.missions.operations.base import Operation, Process


class MmlfOperation(Operation):
    """ Operation class for executing MMLF experiments.

    An MMLF operation consists of one or several runs of
    one or several MMLF worlds.
    """
    def __init__(self, processes, operation_spec, result_directory,
                 world_name, number_processes, create_process=None):
        super(MmlfOperation, self).__init__(processes, operation_spec,
                                            result_directory)

        self.world_name = world_name
        self.number_processes = number_processes
        self.create_process = create_process

    @classmethod
    def create(cls, operation_spec, result_directory, debug=False, input_paths=[]):
        """
        A factory method that creates an MMLF operation based on the
        information given in the operation specification operation_spec
        """
        assert(operation_spec["type"] == "mmlf")

        # The generic world configuration YAML file
        world_conf = """
worldPackage : %s
environment:
%s
agent:
%s
monitor:
    policyLogFrequency : 100000
%s
"""
        # Create directory for the experiment
        world_name = operation_spec['world_name']
        world_path = "%s/config/%s" % (result_directory, world_name)
        if not os.path.exists(world_path):
            os.makedirs(world_path)

        # Compute all possible parameter combinations
        # Determine all parameter combinations that should be tested
        parameter_settings = cls._get_parameter_space(operation_spec)

        # If the operation spec defines parameters for a generalized domain:
        if "generalized_domain" in operation_spec:
            # We have to test each parameter setting in each instantiation of
            # the generalized domain. This can be achieved by computing the
            # crossproduct of parameter settings and domain settings
            augmented_parameter_settings = []
            for parameter_setting in parameter_settings:
                for domain_parameter_setting in operation_spec["generalized_domain"]:
                    instantiation = dict(parameter_setting)
                    instantiation.update(domain_parameter_setting)
                    augmented_parameter_settings.append(instantiation)
            parameter_settings = augmented_parameter_settings

        # Create and remember all worlds for the given  parameter_settings
        world_pathes = []

        # for all parameter setting
        for parameter_setting in parameter_settings:
            # Add 4 blanks to all lines in templates
            environment_template = operation_spec['environment_template']
            environment_template = \
                    "\n".join("    " + line
                                for line in environment_template.split("\n"))
            agent_template = operation_spec['agent_template']
            agent_template = \
                    "\n".join("    " + line
                                for line in agent_template.split("\n"))
            monitor_conf = \
                operation_spec['monitor_conf'] if 'monitor_conf' in operation_spec else ""
            monitor_conf = \
                    "\n".join("    " + line
                                for line in monitor_conf.split("\n"))

            # Instantiate the templates
            environment_conf = environment_template
            agent_conf = agent_template
            for parameter, value in parameter_setting.iteritems():
                environment_conf = environment_conf.replace(parameter, str(value))
                agent_conf = agent_conf.replace(parameter, str(value))

            def get_parameter_str(parameter_name):
                return "".join(subpart[:4]
                                    for subpart in  parameter_name.split("_"))
            configuration_str = "{" + "}{".join(["%s:%s" % (get_parameter_str(parameter),
                                                            str(value)[:6])
                                        for parameter, value in parameter_setting.iteritems()]) + "}"
            configuration_str = configuration_str.replace('_','')

            world_file_name = "world_%s.yaml" %  configuration_str

            open(os.path.join(world_path, world_file_name),
                 'w').write(world_conf % (world_name, environment_conf,
                                          agent_conf, monitor_conf))
            world_pathes.append(os.path.join(world_path, world_file_name))

        number_processes = len(world_pathes)*int(operation_spec["runs"])

        if debug is True:
            # To better debug creation of processes we don't limit the queue
            # and create all processes before executing them
            processes = processing.Queue()
            cls._createProcesses(processes, world_pathes, operation_spec,
                                 result_directory)
            return cls( processes, operation_spec, result_directory,
                                            world_name, number_processes)
        else:
            # Create all processes by calling a recursive helper method
            # in another thread so that already created processes can be
            # executed although creation of processes is not finished yet.
            # Therefor a queue is used which size is limited to guarantee that
            # not to much objects are created (since this costs memory).
            # However, the actual number of 100 is arbitrary and might be
            # changed according to the system at hand.
            processes = processing.Queue(100)
            create_process = processing.Process(target =
                cls._createProcesses, args=( processes, world_pathes,
                                operation_spec, result_directory))
            create_process.start()
            # create and return the operation object
            return cls( processes, operation_spec, result_directory, 
                        world_name, number_processes, create_process)

    @classmethod
    def _createProcesses(cls, processes, world_pathes, operation_spec,
                         result_directory):
        """Function that creates the mmlf process. """
        # For each created world, perform operation_spec["runs"] runs
        for world_conf_path in world_pathes:
            for run_number in range(operation_spec["runs"]):
                process = MMLFProcess(mmlf_path=operation_spec["mmlf_path"],
                                      world_conf_path=world_conf_path,
                                      learning_episodes=operation_spec["learning_episodes"],
                                      test_episodes=operation_spec["test_episodes"],
                                      result_directory=result_directory)
                processes.put(process)
        # give executing process the sign that creation is now finished
        processes.put(False)

    def consolidate(self):
        """ Consolidates the results of the MMLF operation."""
        return  # Consolidation into a csv-file currently not supported.
        # Consolidate the results in an csv table
        output_file = open(os.path.join(self.result_directory,
                                        'results.csv'), 'w')
        results_writer = csv.writer(output_file)

        # Determine directory in which the operation's results are stored
        log_dir = os.path.join(self.result_directory, 'logs', self.world_name)

        # For all configurations tested
        keys = None
        for conf_dir in [os.path.join(log_dir, name) for name in os.listdir(log_dir)
                           if os.path.isdir(os.path.join(log_dir, name))]:
            # Read agent configuration dict
            agent_conf = ElementTree.parse(os.path.join(conf_dir, 'agent.xml'))
            conf_dict = dict(agent_conf.getroot().find('configDict').items())

            # Read environment configuration dict
            env_conf = ElementTree.parse(os.path.join(conf_dir, 'env.xml'))
            conf_dict.update(dict(env_conf.getroot().find('configDict').items()))

            # Evaluate the values of all dict entries
            for key, value in conf_dict.iteritems():
                conf_dict[key] = eval(value)

            # Flatten (potentially) nested dict
            expanded_conf_dict = MmlfOperation._flatten_dict(conf_dict)

            if keys is None:
                keys = expanded_conf_dict.keys()
                # TODO: Currently hard coded
                keys.append('reward')
                keys.append('accumulated_reward')
                keys.append('offline_reward')
                keys.append('offline_accumulated_reward')
                results_writer.writerow(keys)

            # For all runs conducted with this setup
            for run_dir in [os.path.join(conf_dir, name) for name in os.listdir(conf_dir)
                                if os.path.isdir(os.path.join(conf_dir, name))]:
                # Read in the accumulated reward per episode file
                reward_file = \
                    open(os.path.join(run_dir, 'environment_logs', 'reward'),
                         'r')
                rewards = [eval(row) for row in reward_file.readlines()]
                # Chop off overlong sequences
                rewards = rewards[:self.operation_spec["learning_episodes"]]

                expanded_conf_dict['reward'] = rewards
                expanded_conf_dict['accumulated_reward'] = sum(rewards)

                # Read in the accumulated reward per episode file of the
                # offline test
                offline_reward_file = \
                    open(os.path.join(run_dir, 'environment_logs',
                                      'offlineReward'),'r')
                offline_rewards = [eval(row) for row in offline_reward_file.readlines()]
                # Chop off overlong sequences
                offline_rewards = offline_rewards[:self.operation_spec["test_episodes"]]

                expanded_conf_dict['offline_reward'] = offline_rewards
                expanded_conf_dict['offline_accumulated_reward'] = sum(offline_rewards)

                results_writer.writerow([expanded_conf_dict[key] for key in keys])

        output_file.close()

    @classmethod
    def _flatten_dict(cls, input_dict):
        """ Flatten the (potentially) nested dict *input_dict* """
        flattened_dict = dict()
        for key, value in input_dict.iteritems():
            if type(value) == dict:
                value = cls._flatten_dict(value)
                for subkey, subvalue in value.iteritems():
                    flattened_dict[key + '_' + subkey] = subvalue
            else:
                flattened_dict[key] = value
        return flattened_dict


class MMLFProcess(Process):
    """ Process for executing an MMLF run

    One MMLF process consists of a executing one run of a predefined number of
    episodes of a MMLF world.

    The following **parameters** are mandatory:

      :mmlf_path: The path to the MMLF code
      :world_conf_path:   The relative path of the world configuration file
                          within the specified MMLF rw area
      :learning_episodes:   The number of episodes the MMLF agent is allowed
                            to train (i.e. optimize its policy)
      :test_episodes:   The number of episodes the learned policy is tested.
                        This can be set to 1 for deterministic environments.
      :result_directory:   The directory where the stdout of the MMLF run is
                           stored
    """

    def __init__(self, mmlf_path, world_conf_path, learning_episodes,
                 test_episodes, result_directory):

        super(MMLFProcess, self).__init__()

        self.result_directory = result_directory

        self.mmlf_path = mmlf_path
        self.world_conf_path = world_conf_path
        self.learning_episodes = learning_episodes
        self.test_episodes = test_episodes

        self.handler_class = None

    def __call__(self):
        """ Executes this process on the respective modality """
        # Restore configuration
        pySPACE.configuration = self.configuration

        ############## Prepare benchmarking ##############
        super(MMLFProcess, self).pre_benchmarking()

        # Redirect stdout to a log file
        process_log_file_name = "%s/%s_%s" % (self.result_directory,
                                              time.strftime("%Y%m%d_%H_%M_%S"),
                                              os.getpid())
        log_file = open(process_log_file_name, 'w')
        sys.stdout = log_file
        sys.stderr = log_file
        # Insert path to MMLF to python search path and import it
        sys.path.insert(0, self.mmlf_path)
        mmlf = __import__("mmlf")

        # On distributed file systems, it might happen that our configuration
        # file is not yet available on the machine where this process is
        # executed
        while not os.path.exists(self.world_conf_path):
            mmlf.log.warn("Config file %s not yet available on local machine"
                                                        % self.world_conf_path)
            time.sleep(1)

        mmlf.initializeRWArea(rwPath=self.result_directory)

        # Create and start world
        world = mmlf.loadWorldFromConfigFile(self.world_conf_path, useGUI=False)
        try:
            world.run(self.learning_episodes)
        except Exception, e:
            mmlf.log.error("An exception of type %s occurred: %s"
                                                    % (e.__class__.__name__, e))
            print __import__("traceback").print_exc()
        # Stop the world
        world.stop()

        # Restore stdout
        sys.stdout.flush()
        sys.stdout = sys.__stdout__

        ############## Clean up after benchmarking ##############
        super(MMLFProcess, self).post_benchmarking()
