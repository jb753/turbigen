"""This file contains functions for reading TS probe data."""
import numpy as np
import sys, os, json
from ts import ts_tstream_reader, ts_tstream_patch_kind, ts_tstream_cut

Pref = 1e5
Tref = 300.0

# Choose which variables to write out
varnames = ["x", "rt", "eff_lost", "pfluc"]


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

    print("POST-PROCESSING %s\n" % output_hdf5)

    # Load the grid
    tsr = ts_tstream_reader.TstreamReader()
    g = tsr.read(output_hdf5)

    # Gas properties
    cp = g.get_av("cp")  # Specific heat capacity at const p
    ga = g.get_av("ga")  # Specific heat ratio
    rgas = cp * (1.0 - 1.0 / ga)

    # Numbers of grid points
    blk = [g.get_block(bidn) for bidn in g.get_block_ids()]
    ni = [blki.ni for blki in blk]
    nj = [blki.nj for blki in blk]
    nk = [blki.nk for blki in blk]

    # Take cuts at inlet/outlet planes of each row
    stator_inlet = ts_tstream_cut.TstreamStructuredCut()
    stator_inlet.read_from_grid(
        g,
        Pref,
        Tref,
        0,
        ist=0,
        ien=1,  # First streamwise
        jst=0,
        jen=nj[0],  # All radial
        kst=0,
        ken=nk[0],  # All pitchwise
    )
    stator_outlet = ts_tstream_cut.TstreamStructuredCut()
    stator_outlet.read_from_grid(
        g,
        Pref,
        Tref,
        0,
        ist=ni[0] - 2,
        ien=ni[0] - 1,  # Last streamwise
        jst=0,
        jen=nj[0],  # All radial
        kst=0,
        ken=nk[0],  # All pitchwise
    )

    rotor_inlet = ts_tstream_cut.TstreamStructuredCut()
    rotor_inlet.read_from_grid(
        g,
        Pref,
        Tref,
        g.get_block_ids()[-1],
        ist=1,
        ien=2,  # First streamwise
        jst=0,
        jen=nj[1],  # All radial
        kst=0,
        ken=nk[1],  # All pitchwise
    )
    rotor_outlet = ts_tstream_cut.TstreamStructuredCut()
    rotor_outlet.read_from_grid(
        g,
        Pref,
        Tref,
        g.get_block_ids()[-1],
        ist=ni[1] - 2,
        ien=ni[1] - 1,  # Last streamwise
        jst=0,
        jen=nj[1],  # All radial
        kst=0,
        ken=nk[1],  # All pitchwise
    )

    # Pull out area-average flow varibles from the cuts
    cuts = [stator_inlet, stator_outlet, rotor_inlet, rotor_outlet]
    Po, P, To, T, Vx, Vt, Vt_rel = [
        np.array([ci.mass_avg_1d(var_name)[1] for ci in cuts])
        for var_name in ["pstag", "pstat", "tstag", "tstat", "vx", "vt", "vt_rel"]
    ]

    # Calculate entropy change with respect to inlet
    ds = cp * np.log(T / Tref) - rgas * np.log(P / Pref)
    ds_ref = ds[0]
    ds = ds - ds_ref

    # Calculate metadata
    meta = {}

    # Polytropic efficiency
    meta["eff_poly"] = ga / (ga - 1.0) * np.log(To[3] / To[0]) / np.log(Po[3] / Po[0])
    meta["eff_isen"] = (To[3] / To[0] - 1.0) / (
        (Po[3] / Po[0]) ** ((ga - 1.0) / ga) - 1.0
    )

    # Flow angles
    meta["alpha"] = np.degrees(np.arctan2(Vt, Vx))
    meta["alpha_rel"] = np.degrees(np.arctan2(Vt_rel, Vx))

    # Lost effy from
    # eta = wx/(wx+Tds) = 1/(1+Tds/wx) approx 1-Tds/wx using Taylor expansion
    meta["eff_lost_avg"] = -T[3] * ds / cp / (To[3] - To[0])

    # Determine number of probes
    ncycle = g.get_av("ncycle")  # Number of cycles
    nstep_cycle = g.get_av("nstep_cycle")  # Time steps per cycle
    nstep_save_probe = g.get_av("nstep_save_probe")
    nstep_save_start_probe = g.get_av("nstep_save_start_probe")
    nstep_total = ncycle * nstep_cycle
    nstep_probe = nstep_total - nstep_save_start_probe

    # Iterate over all probe patches
    vars_all = []
    dijk_all = []
    eff_lost_unst = []
    for bid in g.get_block_ids():

        rpm_now = g.get_bv("rpm", bid)

        for pid in g.get_patch_ids(bid):
            patch = g.get_patch(bid, pid)

            if patch.kind == ts_tstream_patch_kind.probe:

                print("Reading probe bid=%d pid=%d" % (bid, pid))

                di = patch.ien - patch.ist
                dj = patch.jen - patch.jst
                dk = patch.ken - patch.kst

                dijk_all.append((di, dj, dk))

                fname_now = output_hdf5.replace(
                    ".hdf5", "_probe_%d_%d.dat" % (bid, pid)
                )

                dat_now = read_dat(fname_now, (di, dj, dk))
                dat_now = secondary(dat_now, rpm_now, cp, ga)

                dat_now["eff_lost"] = (
                    -T[3] * (dat_now["ds"] - ds_ref) / cp / (To[3] - To[0])
                )

                # Unsteady lost efficiency at exit if this is a rotor passage
                if not rpm_now == 0.0:
                    eff_lost_unst.append(
                        np.mean(dat_now["eff_lost"][-2, ...], axis=(0, 1))
                    )

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

# Take mean of each rotor passage effy
meta["eff_lost_unst"] = np.mean(eff_lost_unst, 0)

# Determine number of stators and rotors
rpms = np.array([g.get_bv("rpm", bidi) for bidi in g.get_block_ids()])
nstator = np.sum(rpms == 0.0)
nrotor = np.sum(rpms != 0.0)

# Join the grid points from all probes together
var_out = np.concatenate(vars_all, axis=1)

basedir = os.path.dirname(output_hdf5)
np.savez_compressed(
    os.path.join(basedir, "dbslice"),
    data=var_out,
    sizes=dijk_all,
    nsr=(nstator, nrotor),
)

# Save the metadata (lists because numpy arrays not seralisable)
for k in meta:
    try:
        meta[k][0]
        meta[k] = list(meta[k])
    except IndexError:
        pass

with open(os.path.join(basedir, "meta.json"), "w") as f:
    json.dump(meta, f)
