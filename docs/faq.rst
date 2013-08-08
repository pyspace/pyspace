.. _faq:

Frequently Asked Questions and pySPACE Troubleshooting
======================================================

Most errors in pySPACE come from errors in your configuration files or data.
Sometimes programming errors or lack of documentation is the problem.
In the following, we would like to give some advice for handling errors,
when you can not see from the error message or the method documentation,
what went wrong.

Logging and backend choice
--------------------------

Since pySPACE is under development, it is unfortunately not working perfect
currently, though we strive to improve. Normally you should get a meaningful
error message telling you what to do.
To get the best information, we recommend to use the
default serial backend in case of
problems and to increase the logging level. This is done in your
:ref:`configuration file<conf>`.

Finish the run.launch.py process after crash
--------------------------------------------

We are currently searching for the reason
but possibly due to multiprocessing or multi-threading usage
python is sometimes not forwarding crashes to the main process
but only the error message.
So ``ctrl+c`` want work. You have to kill the process by finding out its id.
Another possibility is to us ``ctrl+z`` to stop it and afterwards the command
``kill -9 %1`` and press ``enter`` twice.
This command forces the stopped process with number *1* to be finished.
If your stopped process has another number, you have to change that.

Process seems to be finished successfully but it does not
---------------------------------------------------------

This can have two reasons.

When running very large operations with final classifications,
pySPACE is reducing the output at the end to mainly have only
two result tabular the config files, an example folder and
a zip file which compresses the other results.
The construction of the tabular and the zip file can take a while.
Here more waiting is the best solution.

The more probable reason for this issue is, that there is a very subtle random
error in the final finishing of the software after code execution,
when every component is just shut down.
We guess the logging is the reason and we are planning to rewrite it soon.
Until now, the only solution we know is to finish the process
as already described for the crashed process.

Operation using LibSVM stops processing after a while
-----------------------------------------------------

For some system configurations the LibSVM package
is not working together with the multiprocessing package of python
and subprocesses are not finished in the right way.
So you should use a different classifier or avoid using the multiprocessing
package.
This can be done by changing the backend and/or switching of parallelization
in a meta node which calls the classifier.

Documentation search tool is not showing the expected results
-------------------------------------------------------------

This search tool from sphinx is not making a full text search but
created a reduced search index beforehand.
For future we may try to integrate other search tools.

Further questions?
------------------

If you run into any troubles feel free to contact the developers.
Furthermore, we appreciate any help and bug reports, even if it just tells
us, that documentation can be misunderstood or is incomplete or
certain errors should be caught or throw better error messages.