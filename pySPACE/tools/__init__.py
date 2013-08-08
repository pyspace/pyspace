""" Several communication, file, memory, and performance tools used in pySPACE

This includes communication tools, file, memory, performance handling,
functions for the console output,...

The most important component of tools for nodes is the
:mod:`~pySPACE.tools.memoize_generator`,
which is responsible for the optimization of function calls, by using the memory.
In general :mod:`~pySPACE.tools.filsystem` is often used for handling files and directories.
:mod:`~pySPACE.tools.csv_analysis` is quite useful for handling
csv files, especially if the processing broke or
you want to merge or reduce existing files
Last but not least, :mod:`~pySPACE.tools.gprof2dot` can be used to generate plots
for profiling calls of :mod:`~pySPACE.run.launch`.
"""