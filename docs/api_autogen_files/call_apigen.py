#!/usr/bin/env python
""" Script to auto-generate our API docs """
# stdlib imports for file name handling
import os

# local import of (largely modified) file generator
from apigen import ApiDocWriter

import sys
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,os.pardir)))
#*****************************************************************************
if __name__ == '__main__':
    package = os.path.join('pySPACE')
    outdir = os.path.join('api','generated')
    docwriter = ApiDocWriter(package)
    docwriter.package_skip_patterns = [r'\.docs$']
    docwriter.module_skip_patterns = []
    docwriter.write_api_docs(outdir)
#    docwriter.write_index(outdir+os.sep, 'gen', relative_to='api')
    print('%d module files written' % len(docwriter.written_modules))
