# Disable all built in rules
.SUFFIXES:
MAKEFLAGS += --no-builtin-rules

# Allow bash syntax
SHELL := /bin/bash

install ::
	pip install -e .
	pip install -e .[docs]
	pip install -e .[test]
	pip install pre-commit build bump-my-version
	pre-commit install

doc-dev ::
	sphinx-autobuild doc doc/_build --watch turbigen

doc ::
	sphinx-build -W doc doc/_build

sdist ::
	python -m build --sdist .

test ::
	pytest

compile ::
	f2py -m compiled -c turbigen/compiled.f90
	mv compiled*.so turbigen


verify-sdist ::
	mkdir -p test-sdist
	rm -rv test-sdist/*
	tar -xvf $(TARBALL) --directory=test-sdist
	python -m virtualenv test-sdist/venv
	source test-sdist/venv/bin/activate \
		&& cd $(wildcard ./test-sdist/turbigen*/) \
		&& python -m pip install -e ./[test] \
		&& pytest --rootdir=. -x


TARBALL := $(shell mkdir -p dist && find dist -name '*.tar.gz' | sort | tail -1)

FIG_PY := $(wildcard fig/*.py)
FIG_PDF := $(FIG_PY:%.py=%.pdf)

fig : $(FIG_PDF)

fig/%.pdf : fig/%.py
	python $<

clean ::
	rm -rf fig/*.pdf
	rm -rf doc/_build
