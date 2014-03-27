.. _license:

License
=======

pySPACE ist distributed under GPL::

    Copyright 2008-2013 DFKI RIC and University of Bremen

    pySPACE is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    pySPACE is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with pySPACE.  If not, see <http://www.gnu.org/licenses/>.

.. _cite:

Citation
========

When using our software, we request you to cite the following paper::

    @article{krell2013:pySPACE,
    author = {Krell, Mario Michael and Straube, Sirko and Seeland, Anett and W{\"o}hrle, Hendrik and Teiwes, Johannes and Metzen, Jan Hendrik and Kirchner, Elsa Andrea and Kirchner, Frank},
    journal = {Frontiers in Neuroinformatics},
    title = {{pySPACE} — a signal processing and classification environment in {Python}},
    year = {2013},
    volume = {7},
    number = {40},
    url = {http://www.frontiersin.org/neuroinformatics/10.3389/fninf.2013.00040/abstract},
    doi = {10.3389/fninf.2013.00040},
    ISSN = {1662-5196},
    }

    Krell, M. M., Straube, S., Seeland, Anett,  Wöhrle, H., Teiwes, J.,
    Metzen, J. H., Kirchner, E. A., Kirchner, F. (2013).
    pySPACE — a signal processing and classification environment in Python,
    Frontiers in Neuroinformatics 7(40), doi: 10.3389/fninf.2013.00040.

Furthermore, when using special algorithms of pySPACE we request you to also
cite the corresponding publications to give credits to the scientists
who invented them.
This does not count for standard algorithms but for algorithms
with explicit naming of the corresponding citation.


External Licenses
=================

A few parts of this framework come from external sources.
Every external source should be declared separately in the corresponding files.
In the following there are some examples of external sources.
All external code components should be marked as such and contain their
license or reference.
If not, please inform the developers such they can change this.

MDP
---

pySPACE partially uses and changes code of the MDP framework (version 3.1).
This is done in :mod:`~pySPACE.missions.nodes.base_node` (signal_node.py)
and
:mod:`~pySPACE.environments.chains.node_chain` (mdp flow)
and
:mod:`~pySPACE.missions.nodes.scikits_nodes` (scikits_nodes.py)

MDP is distributed under the following BSD license::

    This file is part of Modular toolkit for Data Processing (MDP).
    All the code in this package is distributed under the following conditions:

    Copyright (c) 2003-2012, MDP Developers <mdp-toolkit-devel@lists.sourceforge.net>

    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the Modular toolkit for Data Processing (MDP)
          nor the names of its contributors may be used to endorse or promote
          products derived from this software without specific prior written
          permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Zito, T., Wilbert, N., Wiskott, L., Berkes, P. (2009).
Modular toolkit for Data Processing (MDP): a Python data processing frame work,
Front. Neuroinform. (2008) 2:8. doi:10.3389/neuro.11.008.2008.

Sphinx
------

To automatically generate the documentation api files, we manipulated
Sphinx scripts, to fit our needs, which can be found in
``docs.api_autogen_files``.

Sphinx is distributed under the following BSD license::

    Copyright (c) 2007-2011 by the Sphinx team (see AUTHORS file).
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Relative Margin Machine
-----------------------

The matlab code for the relative margin machine in
``pySPACE.missions.nodes.classification.svm_variants.rmm.m`` is provided under
the following BSD license::

    Copyright (c) 2008, Pannagadatta Shivaswamy and Tony Jebara, Columbia University
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    The views and conclusions contained in the software and documentation are those
    of the authors and should not be interpreted as representing official policies,
    either expressed or implied, of the FreeBSD Project.

The code is partially wrapped in
:class:`~pySPACE.missions.nodes.classification.svm_variants.brmm.RMMClassifierMatlabNode`.

Other external sources
-----------------------------------------

Several :mod:`pySPACE.tools` are external code copies.

The module :mod:`~pySPACE.tools.gprof2dot` is under LGPL::

    Copyright 2008-2009 Jose Fonseca

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

      Copyright (c) 2002-2009 -- ProphICy Semiconductor, Inc.
                       All rights reserved.

The module :mod:`~pySPACE.tools.memory_profiling`
is under the following BSD license::

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

    - Neither the name of ProphICy Semiconductor, Inc. nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
    OF THE POSSIBILITY OF SUCH DAMAGE.

The module :mod:`~pySPACE.tools.progressbar` is under LGPL license::

    progressbar  - Text progressbar library for python.
    Copyright (c) 2005 Nilton Volpato

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

The modules :mod:`~pySPACE.tools.logging_stream_colorer` and
:mod:`~pySPACE.tools.socket_logger` were mainly
taken as code snippets from web sites.