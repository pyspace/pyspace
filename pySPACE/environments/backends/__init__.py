""" Handle the execution of processes

If you are testing the software or you want to locate a bug,
the :mod:`serial backend<pySPACE.environments.backends.serial>` is your choice.
If you want to be fast and use each of the CPU kernels on your computer,
use the :mod:`multicore backend<pySPACE.environments.backends.multicore>`.
For High Performance Computing or other distributed computing variants
you will mostly need to implement new backends.

The backend is chosen in the :func:`~pySPACE.environments.backends.base.create_backend` method and
normally the user chooses it as the first parameter in the command line execution.
"""