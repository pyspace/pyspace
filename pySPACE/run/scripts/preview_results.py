""" Allows to visualize the results of a running operation.

This script collects all partial results file that have already been generated 
by a running operation, merges them, and visualizes them in the 
performance results analysis GUI.

Invoke script with the pathname of the running operation
E.g.
$ python preview_results.py /mnt/cluster_home/collections/operation_results/20110908_10_43_52
"""

if __name__ == '__main__':
    import sys
    import os
    import shutil
    import tempfile
    import fnmatch
    
    from PyQt4 import QtGui
    
    # Copy the csv-files to a temporary directory such that there is no risk of 
    # interfering with the running operation
    input_dir = sys.argv[1]
    temp_dir = tempfile.mkdtemp()
    for filename in fnmatch.filter(os.listdir(os.path.abspath(input_dir)),
                                   "*.csv"):
        shutil.copy(input_dir + os.sep + filename, temp_dir)

    file_path = os.path.dirname(os.path.abspath(__file__))
    pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
    if not pyspace_path in sys.path:
        sys.path.append(pyspace_path)


    # Import csv-analysis and merge csv files
    from pySPACE.resources.dataset_defs.performance_result import PerformanceResultSummary
    PerformanceResultSummary.merge_performance_results(temp_dir)
    
    # Invoke results analysis gui
    from pySPACE.run.gui.performance_results_analysis import PerformanceResultsAnalysisMainWindow
    
    app = QtGui.QApplication(sys.argv)
    performance_results_analysis = \
            PerformanceResultsAnalysisMainWindow(temp_dir + os.sep + "results.csv")
    performance_results_analysis.show()
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    sys.exit(app.exec_())