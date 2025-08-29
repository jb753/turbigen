#!/usr/bin/env bash
# Autogrid server wrapper for running as systemd unit
set -euo pipefail

WORKDIR="/mnt/data/jb753/meshing/"
LOGFILE="$WORKDIR/daemon.log"

# Add Numeca IGG to PATH if it exists
IGG_PATH="/usr/numeca/bin"
if [ -d "$IGG_PATH" ]; then
    export PATH="$IGG_PATH:$PATH"
else
    echo "Error: Numeca IGG path $IGG_PATH does not exist."
    exit 1
fi

# Activate the venv
VENV_PATH="/home/jb753/python/turbigen-dev/.venv"
source "$VENV_PATH/bin/activate"

# Run the server, teeing output to log
turbigen-autogrid-server "$WORKDIR" --workers=8  2>&1 | tee -a "$LOGFILE"
