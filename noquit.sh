#!/bin/bash
# noquit DIR
# Exit 1 if any 'quit()' or print statements are found in any files under DIR
! grep -nH '^ *quit()' "$1"
! grep -nH '^ *print(' "$1"
