#!/bin/bash
# Usage: source setup.sh
#
# Set up the Linux environment ready to run turbigen

# If we are running on the Cambridge HPC then prepare modules
if [[ $(hostname) =~ "login-" ]] || [[ $(hostname) =~ "gpu-q" ]]; then

    # Load modules
    . /etc/profile.d/modules.sh
    module purge
    if [ $(which pyenv) ]; then
        echo "Found pyenv, using currently activated python version."
    else
        module load python-3.9.6-gcc-5.4.0-sbr552h
    fi
    module load metis-5.1.0-gcc-5.4.0-rcmbph3
    # Load correct libraries on login/compute nodes
    if [[ $(hostname) =~ "login-e" ]]; then
        module load rhel7/default-gpu
    else
        module load rhel8/default-amp
    fi

fi
# (otherwise, running locally, will have to rely on system python)

# Make a python virtualenv
ENV_DIR="env-turbigen"
if [ -f .init_finished ]; then
    echo 'venv already initialised'
    source "$ENV_DIR"/bin/activate
else
    echo 'Making new venv'
    python3 -m virtualenv "$ENV_DIR"
    source "$ENV_DIR"/bin/activate
    # Installl this directory as a package
    python3 -m pip install --upgrade pip
    python3 -m pip install -e .
fi

export PYTHONDONTWRITEBYTECODE=0

touch .init_finished
