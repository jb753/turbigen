Changelog
=========

v2.2.0
^^^^^^

* Allow partial configuration files, exiting at first blank required key
* Faster compiler flags for Fortran
* Turbostream solvers with new base class
* Generate meanline documentation from the docstrings
* Catch post-processing errors and continue
* Submit jobs to a local queue, implement worker script
* Check for nans in ember
* Allow invocation with `python -m turbigen`
* Implement design space sampling
* Bug fix for repeating stage iterator if spanwise points unequal
* General tidying up

v2.1.0
^^^^^^

* Add generic job configuration, and implement SLURM submission
* Add description and README to PyPI
* Add changelog, license, command-line options to docs
* Clean up some dead code and unused imports

v2.0.0
^^^^^^

* Comprehensive internal refactoring of the code for better maintainability
* Finish the built-in CFD solver
* Plugin system now uses abstract base classes and reads all modules from a directory, rather than importing a single file by path
* AutoGrid server supports parallel workers, more settings, better robustnees checks
* Automate running examples to include results in docs
* Numerous bug fixes, improved error messages, etc.
* Use meson to compile the Fortran during installation, upload built linux
  wheels to PyPI -- allows pip install

v1.10.1
^^^^^^^

* Bugfix for AutoGrid server not reading the delete switch
* Bugfix for scalar mean-line iterate variables

v1.10.0
^^^^^^^

* Use rotation-corrected static pressure to find stagnation points
* Add mixed-flow mean-line with mixed and pure radial examples
* Script to read polynomial coefficients and evaluate a previously-fitted design space
* Allow unbladed first row in H-mesh
* Add a syntax check for TS4 body force template
* Generalise to multi-stage machines
* Wrap the incidence angle with respect to camber angle
* Bugfix for TS4 throttle config
* Add -m command-line switch to debug the mean-line
* Take CFD inlet pitch angle from the mesh, not mean-line design
* Implement command-line options for AutoGrid server
* Bugfix to allow incidence correction on only one section
* Reduce LE recamber when incidence is close to tolerance


v1.9.1
^^^^^^

* Add retry for AutoGrid connections
* Use unstructured spanwise cuts to calculate incidence
* More robust tip H-meshing
* Incidence correction bugfix for unbladed cases
* Only reduce proportional part of PID controller gain on restart
* Clean up the hypercube running code
* Turbostream 3 SA helicity correction
* Better Turbostream 4 application variable defaults
* Use turbulent viscosity from old solution on restart
* Fix error in docs iterate config


v1.9.0
^^^^^^

* Add post-processing functions for unstructured cuts at constant span fraction
* Put rotational speeds in AutoGrid geomTurbo file when OH meshing
* Support TS3 cooling patches
* Method to allow trimming of grid blocks in pre-processing
* Compatability fixes for numpy 2 and pin dependency versions
* Add TS4 halo_implementation setting
* Fix bug with zero wall distance on non-matching patches
* Check patch indices are consistent with block size before writing TS3 input


v1.8.0
^^^^^^

* Fit blade sections to coordinates read from a file
* Overhaul post processing with separate functions for all plots
* Rewrite LE incidence calculation to work generally for axial and radial
* Allow off-design operation with rpm_adjust and mass_adjust settings
* Setting for inviscid boudary condition on zero-radius rod
* When running installed turbomachine, reuse installed initial guess
* Allow install function to return inverse design variables to mean line
* Add rounded trailing edge and fillet options to autogrid meshing
* Check for membership of the turbostream group before running TS3
* Error if we cannot locate the specified TS4 throttle tag
* Allow arbitrary setting of perfect gas internal energy datum
* Handle errors in cluster jobs and hold the node for debugging
* Allow unbladed rows in OH meshes
* Close off the tip of unshrouded rotor blades in STL export
* Make the coordinate check optional for debugging
* Fix bounds error on annulus interpolation due to floating point error


v1.7.0
^^^^^^

* Fix bug with noisy TS4 console logging
* Allow labels on TS4 point probes with different file names
* Implement unsteady boundary conditions into pre-processing framework
* Add tutorial to documentation

v1.6.1
^^^^^^

* Fix bug where zero-valued configuration options are not written out

v1.6.0
^^^^^^

* General grid refinement by subdivision of cells
* Allow halting iterations by creating a stopit file in working directory
* Check for two-phase flow at the end of the calculation
* Incidence correction for splitters
* Improve robustness and simplify AutoGrid meshing script
* Use improved clustering functions for H-meshing
* Plot pressure distributions
* Implement loading mean-line, annulus, and installation modules from file
* Allow arbitrary external monitoring scripts in TS4 simulations (e.g. to change body force)
* Configuration option for maximum H-mesh free stream skew
* Fix AutoGrid patch matching bug
* Fix bug with TS4 cfl_ramp_en not set


v1.5.1
^^^^^^

* Minor corrections for open release

v1.5.0
^^^^^^


* General tidying up of the code
* Incidence correction only when mass flow is on target
* Rework configuration and command-line options
* Improve documentation
* Automatic numbering of working directories

v1.4.0
^^^^^^

* Implement sweep by changing meridional locations of LE/TE
* Add splitter capability
* Allow preconditioning in TS4
* Yet more AutoGrid meshing options
* Record Exceptions in the turbigen log file, in addition to STDERR.

v1.3.4
^^^^^^

* More robust unstructured cutting by Marching Cubes algorithm.
* Allow prescribing body force in TS 4.2.82
* Find stagnation point by sign change of surface velocity
* Implement NaN check for TS4
* Write out a design space fit to json for web interface
* More AutoGrid options including untwist outlet

v1.3.3
^^^^^^

* Update radial turbine to set stator LE diameter ratio

v1.3.2
^^^^^^

* Allow custom TS3->TS4 conversion pipelines
* True Taylor camberline (quartic in chi, not tan chi)
* Generalise incidence correction to radial inflows and outflows

v1.3.1
^^^^^^

* Add rotor-only fan mean line

v1.3.0
^^^^^^

* Improved H-mesh tip-gap grid
* Add installation effects module
* Add write coordinates solver
* Allow running in parallel

v1.2.0
^^^^^^

* Implement polynomial design-space fitting.
* Add option to run a hypercube of designs.
* Improve characteristic running.
* Fix bugs with mixing of supersonic flows and area signage.
* Fix bug with setting shroud rpm.
* Generalise to select a type of thickness distribution.
* Clean up the log file outputs.
* Skew H-mesh in flow direction outside of blade rows.
* Added unstructured cutting for post-processing the mixed-out flow.
* Added throttling options to target mass flow for TS3 and TS4.
* Added radial turbine mean-line design functions and example.
* Internal rewrite of data structures to be CFD-solver agnostic.
* ... plus other miscellaneous enhancements and tidying.

v1.1.0
^^^^^^

* Added H-meshing option, with pinched tips.
* `Config` object for programmatic creation and validation of input files.
* Automated post processing to get a `MeanLine` object from mixed-out CFD cuts.
* Use inlet velocity as reference for compressor circulation coefficient.
* Mixed-out averaging generalised for any meridional cut (not just constant axial coordinate).
* Iteration to correct for incidence, deviation, and mean-line guesses.
* Options to set blade number directly or Lieblein diffusion factor.
* Running characteristics for compressor designs.
* Generate real gas tables for TS4 on demand.
* Let TS3 grid object use arbitrary equation of state for post-processing.
* Post-processing TS4 simulations by reading the flow field into a structured TS3 grid.
* Config file options to submit a job to the SLURM queue.

v1.0.0
^^^^^^

* First Whittle Laboratory internal release.
