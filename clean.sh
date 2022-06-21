#!/bin/bash
# Delete all unneeded files

# Keep files less than 2 days old in case simulation is still running
find . -mtime +2 -name 'input.hdf5' -delete
find . -mtime +2 -name 'output.hdf5' -delete

# Delete all xdmfs
find . -name '*.xdmf' -delete

# Delete logs
find . -mtime +2 -name 'log.txt' -delete

# Delete instantaneous hdf5s if the average exists
# find . -mtime +90 -name '*.hdf5' -print -exec delhdf5check {} \;
