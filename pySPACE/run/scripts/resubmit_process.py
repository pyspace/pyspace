"""
script for re-submitting pickled processes when using loadleveler backend

input: absolute path to file ll_call.cmd in result directory
       list of process indices

example call::

    python resubmit_process.py /gpfs01/comman/home/seeland/collections/operation_results/20111201_19_03_42/ll_call.cmd "range(1472,1541)"

:Author: Anett Seeland
"""
import sys
import os

if __name__ == "__main__":
    ll_call_file = sys.argv[1]
    process_list = eval(sys.argv[2])
    template = ""
    
    # read-in cmd-file
    f=open(ll_call_file,'r')
    old_file = f.readlines()
    f.close()
    # parse arguments line
    for line in old_file:
        if line.startswith("# @ arguments"):
            tmp = line.split("/")[-1] # should be something like process_1559.pickle 192.168.12.21 25343
            end_index = tmp.index(".pickle")
            line = line.replace(tmp[:end_index],"process_%d")
        template += line
    # re-submit processes
    for number in process_list:
        f = open(ll_call_file,'w')
        content = template % number
        f.write(content)
        f.close()
        os.system("llsubmit -q %s" % ll_call_file)
        
    