# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile figures

# Regenerate the hand-drawn TikZ figures in _static/ from their sources.
figures:
	cd figures && pdflatex -interaction=nonstopmode -halt-on-error annulus_coordinate.tex
	cd figures && dvisvgm --pdf --no-fonts \
		--output=../_static/annulus_coordinate.svg annulus_coordinate.pdf
	cd figures && rm -f *.aux *.log *.pdf

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
