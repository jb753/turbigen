project('emberc',
        ['c', 'fortran'],
        version : '0.1',
        meson_version: '>= 1.1.0',
        default_options : [
                            'warning_level=1',
                            'buildtype=release'
                          ])
add_global_arguments(['-Werror', '-O3','-funroll-loops','-march=native','-fno-math-errno', '-fno-trapping-math','-ftree-vectorize'], language : 'fortran')
fc = meson.get_compiler('fortran')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os;  import numpy; print(os.path.relpath(numpy.get_include()))'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os;  import numpy.f2py; print(os.path.relpath(numpy.f2py.get_include()))'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

inc_np = include_directories(incdir_numpy, incdir_f2py)
# gh-25000
quadmath_dep = fc.find_library('quadmath', required: false)




py.extension_module('emberc',
                     [
                     'src/ember/distribute.f90',
                     'src/ember/ember.f90',
                     'src/ember/fluxes.f90',
                     'src/ember/indexing.f90',
                     'src/ember/matmul.f90',
                     'src/ember/multigrid.f90',
                     'src/ember/operators.f90',
                     'src/ember/smooth.f90',
                     'src/ember/viscous.f90',
                     'src/ember/embercmodule.c',
                     'src/ember/emberc-f2pywrappers2.f90',
                     'src/ember/emberc-f2pywrappers.f',
                     fortranobject_c
                     ],
                     include_directories: [
                     inc_np,

                     ],
                     dependencies : [
                     py_dep,
                     quadmath_dep,



                     ],

                     install : true)
install_subdir('src/turbigen', install_dir: py.get_install_dir())
