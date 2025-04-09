# Disable all built in rules
.SUFFIXES:
MAKEFLAGS += --no-builtin-rules

# Allow bash syntax
SHELL := /bin/bash

make setup ::
	source setup.sh

make check ::
	source setup.sh && turbigen examples/cascade_test.yaml  && echo "Check passed"


make reinstall ::
	rm -rf venv
	source setup.sh

doc-dev ::
	sphinx-autobuild doc doc/_build --watch=src --watch=doc

doc ::
	sphinx-build -W doc doc/_build

sdist ::
	python -m build --sdist .

compile-slow ::
	python -m numpy.f2py -m embsolvec --opt='-O3 -fcheck=array-temp,bounds -ffast-math -fmax-errors=1' -c turbigen/solvers/embsolve-src/embsolve.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1
	mv embsolve*.so turbigen/solvers

compile-openmp ::
	python -m numpy.f2py -m embsolvec --opt='-O3 -ffast-math -fmax-errors=1' -c turbigen/solvers/embsolve-src/embsolve.f90 --f90flags='-fopenmp' -lgomp
	mv embsolve*.so turbigen/solvers

compile ::
	python3 -m numpy.f2py -m embsolvec  --opt='-O3  -ffast-math -fmax-errors=1' -c turbigen/solvers/embsolve-src/*.f90
	mv embsolve*.so turbigen/solvers

compile-double ::
	python -m numpy.f2py -m embsolvec --opt='-fdefault-real-8 -O3  -ffast-math -fmax-errors=1' -c turbigen/solvers/embsolve-src/embsolve.f90
	mv embsolve*.so turbigen/solvers

compile-intel ::
	python -m numpy.f2py -m embsolvec --fcompiler=intelem --opt='-O3 -xHost -align array64byte -fast -fmax-errors=1' -c turbigen/solvers/embsolve-src/embsolve.f90
	mv embsolve*.so turbigen/solvers

generate-examples ::
	python doc/generate_examples.py
