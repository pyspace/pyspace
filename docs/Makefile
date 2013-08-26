# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -q
SPHINXBUILD   = sphinx-build
PAPER         =
SPHINXAPI     = python api_autogen_files/apidoc.py #sphinx-apidoc
APIGEN        = python api_autogen_files/call_apigen.py

UNAME :=  $(shell uname)

ifeq ($(UNAME), Darwin)
	SPHINXBUILD  =  sphinx-build-2.6
	SPHINXAPI    = python api_autogen_files/apidoc.py #sphinx-apidoc-2.6
endif

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d .build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
PACKAGEGEN      = $(SPHINXAPI) -T -M -q -d 3 -P -f -o api/generated ../ docs library abri_dp adappt #output_path,module_path, exclude_paths
ADAPPT          = $(SPHINXAPI) -T -M -q -d 3 -P -f -o api/adappt ../adappt
MODULEGEN       = $(APIGEN)

.PHONY: help clean html web pickle htmlhelp latex changes linkcheck

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  html_complete to generate html and api files"
	@echo "  pickle    to make pickle files"
	@echo "  json      to make JSON files"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview over all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"

clean:
	-rm -rf .build/*
	-rm -rf api/generated/*
	-rm -rf api/tests/*

html:
	@echo "cleaning ..."
	-rm -rf .build/*
	-rm -rf api/generated/*
	-rm -rf api/tests/*
	@echo "generating API documentation ..."
	mkdir -p .build/html .build/doctrees
	$(MODULEGEN)
	$(PACKAGEGEN)
#	$(ADAPPT)
	@echo
	@echo "API files have been created automatically."
	@echo
	@echo "Building the documentation ... This may take a while."
	@echo
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) .build/html
	@echo
	@echo "Build finished. The HTML pages are in .build/html."
	@echo
	@echo "If you got errors or warnings from scikits or not referring to any module or file, just ignore them. "
	@echo

html_short:
	mkdir -p .build/html .build/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) .build/html
	@echo
	@echo "Build finished. The HTML pages are in .build/html."

pickle:
	mkdir -p .build/pickle .build/doctrees
	$(MODULEGEN)
	$(PACKAGEGEN)
	@echo
	@echo "API files created automatically."
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) .build/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

web: pickle

json:
	mkdir -p .build/json .build/doctrees
	$(MODULEGEN)
	$(PACKAGEGEN)
	@echo
	@echo "API files created automatically."
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) .build/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	mkdir -p .build/htmlhelp .build/doctrees
	$(MODULEGEN)
	$(PACKAGEGEN)
	@echo
	@echo "API files created automatically."
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) .build/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
		".hhp project file in .build/htmlhelp."

latex:
	$(MODULEGEN)
	$(PACKAGEGEN)
	@echo
	@echo "API files created automatically."
	mkdir -p .build/latex .build/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) .build/latex
	@echo
	@echo "Build finished; the LaTeX files are in .build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
		"run these through (pdf)latex."

changes:
	$(MODULEGEN)
	$(PACKAGEGEN)
	@echo
	@echo "API files created automatically."
	mkdir -p .build/changes .build/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) .build/changes
	@echo
	@echo "The overview file is in .build/changes."

linkcheck:
	$(MODULEGEN)
	$(PACKAGEGEN)
	@echo
	@echo "API files created automatically."
	mkdir -p .build/linkcheck .build/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) .build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
		"or in .build/linkcheck/output.txt."
