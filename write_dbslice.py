"""This file contains functions for reading TS probe data."""
import numpy as np
import sys
from ts import ts_tstream_reader, ts_tstream_patch_kind

Pref = 1e5
Tref = 300.0

# Choose which variables to write out
varnames = ["x", "rt", "ds", "pfluc"]


def read_dat(fname, shape):
    """Load flow data from a .dat file"""

    # Get raw data
    raw = np.genfromtxt(fname, skip_header=1, delimiter=" ")

    # Reshape to correct size
    nvar = 8
    shp = np.append(shape, -1)
    shp = np.append(shp, nvar)
    raw = np.reshape(raw, shp, order="F")

    # Split the columns into named variables
    Dat = {}
    varnames = ["x", "r", "rt", "ro", "rovx", "rovr", "rorvt", "roe"]
    for i, vi in enumerate(varnames):
        Dat[vi] = raw[:, :, :, :, i]

    return Dat


def secondary(d, rpm, cp, ga):
    """Calculate other variables from the primary vars stored in dat files."""
    # Velocities
    d["vx"] = d["rovx"] / d["ro"]
    d["vr"] = d["rovr"] / d["ro"]
    d["vt"] = d["rorvt"] / d["ro"] / d["r"]
    d["U"] = d["r"] * rpm / 60.0 * np.pi * 2.0
    d["vtrel"] = d["vt"] - d["U"]
    d["v"] = np.sqrt(d["vx"] ** 2.0 + d["vr"] ** 2.0 + d["vt"] ** 2.0)
    d["vrel"] = np.sqrt(d["vx"] ** 2.0 + d["vr"] ** 2.0 + d["vtrel"] ** 2.0)

    # Total energy for temperature
    E = d["roe"] / d["ro"]
    cv = cp / ga
    d["tstat"] = (E - 0.5 * d["v"] ** 2.0) / cv

    # Pressure from idea gas law
    rgas = cp - cv
    d["pstat"] = d["ro"] * rgas * d["tstat"]

    # Entropy change wrt reference
    d["ds"] = cp * np.log(d["tstat"] / Tref) - rgas * np.log(d["pstat"] / Pref)

    # Pressure fluc wrt time mean
    d["pfluc"] = d["pstat"] - np.mean(d["pstat"], 3, keepdims=True)

    # Angular velocity
    d["omega"] = rpm / 60.0 * 2.0 * np.pi

    # Blade speed
    d["U"] = d["omega"] * d["r"]

    # Save the parameters
    d["rpm"] = rpm
    d["cp"] = cp
    d["ga"] = ga

    return d


if __name__ == "__main__":

    output_hdf5 = sys.argv[1]

    # Load the grid
    tsr = ts_tstream_reader.TstreamReader()
    g = tsr.read(output_hdf5)

    # Here we extract some parameters from the TS grid to use later

    rpm = np.array(
        [g.get_bv("rpm", bid) for bid in g.get_block_ids()]
    )  # RPM in rotor row
    Omega = rpm / 60.0 * np.pi * 2.0
    cp = g.get_av("cp")  # Specific heat capacity at const p
    ga = g.get_av("ga")  # Specific heat ratio
    rgas = cp * (1.0 - 1.0 / ga)

    # Determine number of probes
    ncycle = g.get_av("ncycle")  # Number of cycles
    nstep_cycle = g.get_av("nstep_cycle")  # Time steps per cycle
    nstep_save_probe = g.get_av("nstep_save_probe")
    nstep_save_start_probe = g.get_av("nstep_save_start_probe")
    nstep_total = ncycle * nstep_cycle
    nstep_probe = nstep_total - nstep_save_start_probe

    # Iterate over all probe patches
    vars_all = []
    for bid in g.get_block_ids():

        rpm_now = g.get_bv("rpm", bid)

        for pid in g.get_patch_ids(bid):
            patch = g.get_patch(bid, pid)
            if patch.kind == ts_tstream_patch_kind.probe:

                print("Reading probe bid=%d pid=%d" % (bid, pid))

                di = patch.ien - patch.ist
                dj = patch.jen - patch.jst
                dk = patch.ken - patch.kst
                fname_now = output_hdf5.replace(
                    ".hdf5", "_probe_%d_%d.dat" % (bid, pid)
                )

                dat_now = read_dat(fname_now, (di, dj, dk))
                dat_now = secondary(dat_now, rpm_now, cp, ga)

                # Remove variables we are not interested in
                for k in list(dat_now.keys()):
                    if not k in varnames:
                        dat_now.pop(k)

                # Wangle the dimensions and add to list
                vars_all.append(
                    np.stack(
                        [
                            np.squeeze(
                                dat_now[ki].reshape((-1, dat_now[ki].shape[-1]))
                            ).transpose()
                            for ki in dat_now
                        ],
                        axis=-1,
                    )
                )

# Join the grid points from all probes together
var_out = np.concatenate(vars_all, axis=1)
print(var_out.shape)

# Write out
np.save("dbslice", var_out)
np.savez_compressed("dbslice", data=var_out)
