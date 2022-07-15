#!/bin/bash
#
# A slim wrapper around run_turbostream.py that CLEAR existing environment and
# sources a version of the official env without MPI
#
# Save SSH agent
env -i bash -c "source ./bashrc_module_ts3610_a100 && SSH_AUTH_SOCK=$SSH_AUTH_SOCK run_turbostream.py $@"
