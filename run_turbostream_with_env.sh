#!/bin/bash
#
# A slim wrapper around run_turbostream.py that CLEAR existing environment and
# sources a version of the official env without MPI
#
env -i bash -c "source ./bashrc_module_ts3610_a100 && run_turbostream.py $@"
