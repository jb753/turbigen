# turbigen

The turbigen library is a tool for generating three-dimensional axial turbine
geometries based on a set of input aerodynamic parameters.

Also included are meshing and preprocessing scripts for generating and running
cases using Turbostream.

## Basic usage

0. Clone this repository,
   ```
   git clone -b hmesh git clone -b hmesh https://github.com/jb753/turbigen.git
   ```

1. Set up the Turbostream environment,
   ```
   source /usr/local/software/turbostream/ts3610/bashrc_module_ts3610
   ```

2. To run a case with the default parameters use,
   ```
   python run_default.py
   ```

3. Input files will have been prepared in a numbered directory `run/0000`, and
   a job is now waiting in the queue to run Turbostream. Eventually, when the
   calculation is finished, the results will be processed automatically. The
   input parameters for this case are stored in `params.json`, and metadata
   calculated from the solution in `meta.json`.

4. To run a case with amended parameters, edit the parameters dictionary. For
   example, a sweep in flow coefficient would look like,
   ```python
   import submit
   params = submit.read_params('default_params.json')
   base_run_dir = 'phi_sweep'
   for phii in [0.4, 0.6, 0.8, 1.0, 1.2]:
       params['mean-line']['phi'] = phii
       submit.run(params, base_run_dir)
   ```
   This will generate five cases under the `phi_sweep` directory.

## Parameters listing

The default parameters are representative of a large industrial gas turbine.
Shown below is a list of the defaults; all may be changed. The hub-to-tip ratio
is high to approximate a linear cascade.

```
default_params.json
{
    "mean-line": {
        "phi": 0.8,  # Flow coefficient [-]
        "psi": 1.6,  # Stage loading coefficient [-]
        "Lam": 0.5,  # Degree of reaction [-]
        "Al1": 0.0,  # Inlet yaw angle [deg]
        "Ma": 0.65,  # Vane exit Mach number [-]
        "eta": 0.95,  # Estimated polytropic efficiency [-]
        "ga": 1.33  # Ratio of specific heats [-]
        },
    "bcond": {
        "To1": 1600.0,  # Inlet stagnation temperature [K]
        "Po1": 16e5,  # Inlet stagnation pressure [Pa]
        "rgas": 287.14,  # Specific gas constant [J/kgK]
        "Omega": 314.159  # Rotor shaft speed [rad/s]
        },
    "3d": {
        "htr": 0.995,  # Hub-to-tip radius ratio [-]
        "Re": 4.0e6,  # Axial chord based Reynolds number [-]
        "Co": 0.65  # Circulation coefficient [-] (sets pitch-to-chord)
        },
    "mesh": {
        "dx_c" : [2.0, 1.0, 3.0],  # Axial spacings to inlet, between rows, and to outlet [-]
        "dev" : [0.0, 0.0]  # Guesses for deviation of stator, rotor [deg]
    }
}
```

## Post processing

At the end of the computation, `write_dbslice.py` reads in the time-averaged
solution. Then, any quantity of interest may be calculated and saved to
`meta.json`.

After making a change to the `write_dbslice.py` script, one can re-process all
data using this shell command,
```
find -name "output_1_avg.hdf5" -exec python write_dbslice.py {} \; 
```

James Brind
Jul 2022
