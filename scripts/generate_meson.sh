#!/bin/bash
# This script generates a meson.build file for ember
# There are some gotchas...

# Check we are in projec root dir
if [ ! -d "src" ]; then
    echo "This script must be run from the project root directory."
    exit 1
fi

# Remove old wrapper files
rm -f src/ember/*-f2pywrappers*

# Run f2py in a temporary directory to generate wrappers
WORKDIR="tmp_build"
# T
python3 -m numpy.f2py -m emberc  --opt='-O3  -ffast-math -fmax-errors=1' -c src/ember/*.f90 --build-dir $WORKDIR

# Move the generated meson.build file to root
mv $WORKDIR/meson.build meson.build

# Move the generated wrappers to the src/ember directory
mv $WORKDIR/embercmodule.c $WORKDIR/emberc-f2pywrappers* src/ember/

# Replace the hardcode python installation path
sed -i "s|py = import('python').find_installation('''.*''', pure: false)|py = import('python').find_installation(pure: false)|" meson.build

# Insert relative path to the src/ember directory
sed -i "/py\.extension_module.*emberc/,/fortranobject_c/ s|'''\\([^']*\\)'''|'src/ember/\1'|g" meson.build

# Also install the pure python turbigen module
echo "install_subdir('src/turbigen', install_dir: py.get_install_dir())" >> meson.build

# Clean up
rm -rf $WORKDIR
