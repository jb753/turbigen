# Disable all built in rules
.SUFFIXES:
MAKEFLAGS += --no-builtin-rules

# Allow bash syntax
SHELL := /bin/bash

install ::
	pip install -e .[docs,test,emb]
	pip install pre-commit build bump-my-version
	pre-commit install

doc-dev ::
	sphinx-autobuild doc doc/_build --watch=turbigen --watch=doc

doc ::
	sphinx-build -W doc doc/_build

sdist ::
	python -m build --sdist .

test ::
	pytest

compile-slow ::
	python -m numpy.f2py -m embsolvec --opt='-O3 -fcheck=array-temp,bounds -ffast-math -fmax-errors=1' -c turbigen/solvers/embsolve-src/embsolve.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1
	mv embsolve*.so turbigen/solvers

compile-openmp ::
	python -m numpy.f2py -m embsolvec --opt='-O3 -ffast-math -fmax-errors=1' -c turbigen/solvers/embsolve-src/embsolve.f90 --f90flags='-fopenmp' -lgomp
	mv embsolve*.so turbigen/solvers

compile ::
	python -m numpy.f2py -m embsolvec  --opt='-O3  -ffast-math -fmax-errors=1' -c turbigen/solvers/embsolve-src/*.f90
	mv embsolve*.so turbigen/solvers

compile-double ::
	python -m numpy.f2py -m embsolvec --opt='-fdefault-real-8 -O3  -ffast-math -fmax-errors=1' -c turbigen/solvers/embsolve-src/embsolve.f90
	mv embsolve*.so turbigen/solvers



compile-intel ::
	python -m numpy.f2py -m embsolvec --fcompiler=intelem --opt='-O3 -xHost -align array64byte -fast -fmax-errors=1' -c turbigen/solvers/embsolve-src/embsolve.f90
	mv embsolve*.so turbigen/solvers








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
