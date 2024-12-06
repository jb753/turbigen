#!/bin/bash
# Usage: source setup.sh
#
# Set up the Linux environment ready to run turbigen

# If we are running on the Cambridge HPC then prepare modules
# if [[ "$(hostname)" =~ "login-" ]] || [[ "$(hostname)" =~ "gpu-" ]]; then
  # # Load modules
  # . /etc/profile.d/modules.sh
  # module purge
  # module load python-3.9.6-gcc-5.4.0-sbr552h
  # module load metis-5.1.0-gcc-5.4.0-rcmbph3
  # # Load correct libraries on login/compute nodes
  # if [[ "$(hostname)" =~ "login-e" ]]; then
  #   module load rhel7/default-gpu
  # else
  #   module load rhel8/default-amp
  # fi
# fi
# (otherwise, running locally, will have to rely on system python)

# Make a python virtualenv
TURBIGEN_ROOT=$( realpath $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ))
export PATH="$TURBIGEN_ROOT:$PATH"
ENV_DIR="$TURBIGEN_ROOT/venv"
ENV_BIN="$ENV_DIR"/bin/activate
if [ -f "$ENV_BIN" ]; then
  echo 'Python environment already initialised'
  source "$ENV_BIN"
else
  echo 'Making new Python environment'
  # Initialse the venv with a copy of the Python interpreter
  python3 -m venv --copies "$ENV_DIR"
  # Make the venv portable
  # https://aarongorka.com/blog/portable-virtualenv/
  sed -i '43s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' "$ENV_DIR"/bin/activate
  sed -i '1s/.*/#!\/usr\/bin\/env python/' "$ENV_DIR"/bin/pip*
  source "$ENV_DIR"/bin/activate
  # Get more recent version of pip
  python3 -m pip install --upgrade pip
  # Installl this directory as an editabe package
  python3 -m pip install -e "$TURBIGEN_ROOT"
  # Fix the hardcoded path in shebang on installed scripts
  sed -i '1s/.*python$/#!\/usr\/bin\/env python/' "$ENV_DIR"/bin/*
fi
