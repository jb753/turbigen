Changelog
=========

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
