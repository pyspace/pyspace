""" Merge `result*.csv` files when operation failed to finish.

Evoke script with the pathname where the csv-files are stored.
E.g.

::
    
    python csv-merge.py /Users/seeland/collections/20100812_11_18_58

Optional change *delete_files* to False, when you don't want to remove them.
"""

def main():
    import sys, os
    file_path = os.path.dirname(os.path.abspath(__file__))
    pyspace_path = file_path[:file_path.rfind('pySPACE')-1]
    if not pyspace_path in sys.path:
        sys.path.append(pyspace_path)

    from pySPACE.resources.dataset_defs.performance_result import PerformanceResultSummary as PerformanceResultSummary
    
    input_dir = sys.argv[1]
    PerformanceResultSummary.merge_performance_results(input_dir)
    
    
if __name__ == '__main__':
    main()
