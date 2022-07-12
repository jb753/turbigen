# Set up the Linux environment ready to run turbigen
#
# Notes:
#

# HPC modules
. /etc/profile.d/modules.sh 
module purge 
for mod in \
    rhel8/default-amp \
    python-3.9.6-gcc-5.4.0-sbr552h  #\py-mpi4py-3.0.0-gcc-5.4.0-ik4dghw  #pvy42hk      
do
    module load "$mod"> /dev/null
done


# Python virtualenv
ENV_DIR="env-turbigen"
if [ ! -d "$ENV_DIR" ]; then
    python3 -m virtualenv "$ENV_DIR"
fi
source "$ENV_DIR"/bin/activate

# # Install pyfr into this venv if we have not got it
if ! pip show openmdao &> /dev/null ; then
    pip install numpy
    pip install scipy
    pip install compflow
    pip install mpi4py
    pip install petsc
    pip install petsc4py
    pip install openmdao
fi

# # Point pyfr to my home-built CGNS library
# export PYFR_LIBRARY_PATH="$HOME"/builds/CGNS/lib/lib
