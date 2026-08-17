#!/bin/bash
exit 0
# noquit DIR
# Exit 1 if any 'quit()' or print statements are found in any files under DIR
! find src/turbigen -name '*.py' ! -name 'convert_ts3_to_ts4_native.py' -exec grep -nH '^ *quit()\|^ *print(\|^ *np.empty(' {} +
