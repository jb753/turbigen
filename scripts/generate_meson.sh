#!/bin/bash
# This script generates a meson.build file for ember
# There are some gotchas...
# Usage:

# Check we are in projec root dir
if [ ! -d "src" ]; then
    echo "This script must be run from the project root directory."
    exit 1
fi

# Uninstall existing turbigen
uv pip uninstall turbigen

# Install required build dependencies
uv pip install meson-python numpy ninja

# Remove old wrapper files
rm -f src/ember/*wrapper* src/ember/embercmodule.c

# Run f2py in a temporary directory to generate wrappers
WORKDIR="tmp_build"
python3 -m numpy.f2py --backend=meson -m emberc  --opt='-fmax-errors=1 -Werror' -c src/ember/*.f90 --build-dir $WORKDIR

# Move the generated meson.build file to root
mv $WORKDIR/meson.build meson.build

# Move the generated wrappers to the src/ember directory
mv $WORKDIR/embercmodule.c $WORKDIR/emberc-f2pywrappers* src/ember/

# Replace the hardcoded python installation path
sed -i "s|py = import('python').find_installation('''.*''', pure: false)|py = import('python').find_installation(pure: false)|" meson.build

# Modify paths to fully point to the src/ember directory
# Not inside the build directory
sed -i "/py\.extension_module.*emberc/,/fortranobject_c/ s|'''\\([^']*\\)'''|'src/ember/\1'|g" meson.build

# Insert compiler flags for the Fortran compiler
sed -i "/fc =/i\add_global_arguments(['-Werror', '-O3','-funroll-loops','-march=native','-fno-math-errno', '-fno-trapping-math','-ftree-vectorize'], language : 'fortran')" meson.build

# Convert print include directories to relative paths
# This is needed because
# 1) Doing the pip install -e . with build isolation enabled sets the
#    numpy include directories to absolute paths in /tmp
#    which are not available when rebuilding editable installs
# 2) Running pip install -e . --no-build-isolation causes an error
#    when the numpy include is in an absolute path to a .venv inside
#    the project root directory, like uv makes
# 3) The workaround is to convert the absolute paths for numpy includes
#    to relative paths from the project root, as if the numpy code is ours
sed -i 's/print(\(.*\))/print(os.path.relpath(\1))/' meson.build
sed -i 's/os.chdir("..");//' meson.build

# Also install the pure python turbigen module
echo "install_subdir('src/turbigen', install_dir: py.get_install_dir())" >> meson.build

# Clean up
rm -r $WORKDIR
rm -f emberc.cpython-*.so

# Reinstall turbigen
uv pip install . #--no-cache
