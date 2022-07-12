#!/bin/bash
#
# A slim wrapper around run_turbostream.py that sources the correct environment
#
source /usr/local/software/turbostream/ts3610_a100/bashrc_module_ts3610_a100
run_turbostream.py "$@"
