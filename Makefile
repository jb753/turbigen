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
	python -m numpy.f2py -m embsolve --opt='-O3 -fcheck=array-temp -ffast-math -fmax-errors=1' -c turbigen/embsolve.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1
	mv embsolve*.so turbigen


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

lint ::
	pre-commit run -a

clean ::
	rm -rf fig/*.pdf
	rm -rf doc/_build
