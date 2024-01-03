#!/bin/bash
# noquit DIR
# Exit 1 if 'quit()' is found in any Python files under DIR
! grep -nH '^ *quit()' "$1"
