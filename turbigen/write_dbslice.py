"""This file contains functions for reading TS probe data."""
import numpy as np
import compflow
import sys, os, json
from ts import ts_tstream_reader, ts_tstream_patch_kind, ts_tstream_cut

Pref = 1e5
Tref = 300.0

# Choose which variables to write out
varnames = ["x", "rt", "eff_lost", "pfluc"]


def save_meta(meta, basedir):

    # Save the metadata (lists because numpy arrays not seralisable)
    for k in meta:
        try:
            meta[k][0]
            meta[k] = list(meta[k][:72])
        except IndexError:
            pass

    with open(os.path.join(basedir, "meta.json"), "w") as f:
        json.dump(meta, f)


def node_to_cell(cut, prop_name):
    return np.mean(
        np.stack(
            (
                getattr(cut, prop_name)[:-1, :-1],
                getattr(cut, prop_name)[1:, 1:],
                getattr(cut, prop_name)[:-1, 1:],
                getattr(cut, prop_name)[1:, :-1],
            )
        ),
        axis=0,
    )


def cell_vec(c):
    return c[1:, 1:] - c[:-1, :-1], c[:-1, 1:] - c[1:, :-1]


def cell_area(cut):
    dx1, dx2 = cell_vec(cut.x)
    dr1, dr2 = cell_vec(cut.r)
    drt1, drt2 = cell_vec(cut.rt)

    Ax = 0.5 * (dr1 * drt2 - dr2 * drt1)
    Ar = 0.5 * (dx2 * drt1 - dx1 * drt2)

    return Ax, Ar


def mix_out(cuts):

    cp = cuts[0].cp
    ga = cuts[0].ga
    rgas = cp * (ga - 1.0) / ga
    cv = cp / ga

    props = [
        "ro",
        "rovx",
        "rovr",
        "rorvt",
        "roe",
        "pstat",
        "vx",
        "r",
        "vr",
        "vt",
        "tstag",
    ]
    fluxes = ["mass", "xmom", "rmom", "tmom", "energy"]

    # Preallocate totals
    total = {f: 0.0 for f in fluxes}
    total["Ax"] = 0.0
    total["Ar"] = 0.0

    # Loop over cuts
    for cut in cuts:

        # Cell centered primary properties
        cell = {prop: node_to_cell(cut, prop) for prop in props}

        # Cell areas
        Ax, Ar = cell_area(cut)

        # Fluxes of the non-uniform flow
        flux_x = {
            "mass": cell["rovx"],
            "xmom": cell["rovx"] * cell["vx"] + cell["pstat"],
            "rmom": cell["rovx"] * cell["vr"],
            "tmom": cell["rovx"] * cell["r"] * cell["vt"],
            "energy": cell["rovx"] * cell["tstag"],
        }
        flux_r = {
            "mass": cell["rovr"],
            "xmom": cell["rovr"] * cell["vx"],
            "rmom": cell["rovr"] * cell["vr"] + cell["pstat"],
            "tmom": cell["rovr"] * cell["r"] * cell["vt"],
            "energy": cell["rovr"] * cell["tstag"],
        }

        # Multiply by area and accumulate totals
        for f in fluxes:
            total[f] += np.sum(flux_x[f] * Ax) + np.sum(flux_r[f] * Ar)

        # Accumulate areas
        total["Ax"] += np.sum(Ax)
        total["Ar"] += np.sum(Ar)

    # Now we solve for the state of mixed out flow assuming constant area

    # Mix out at the mean radius
    rmid = np.mean((cut.r.min(), cut.r.max()))

    # Guess for density
    mix = {"ro": np.mean(cell["ro"])}

    # Iterate on density
    for i in range(10):

        # Conservation of mass to get mixed out axial velocity
        mix["vx"] = total["mass"] / mix["ro"] / total["Ax"]

        # Conservation of axial momentum to get mixed out static pressure
        mix["pstat"] = (
            total["xmom"] - mix["ro"] * mix["vx"] ** 2.0 * total["Ax"]
        ) / total["Ax"]

        # Conservation of tangential momentum to get mixed out tangential velocity
        mix["vt"] = total["tmom"] / mix["ro"] / mix["vx"] / total["Ax"] / rmid

        # Destruction of radial momentum
        mix["vr"] = 0.0

        # Total temperature from first law of thermodynamics
        mix["tstag"] = total["energy"] / total["mass"]

        # Velocity magnitude
        mix["vabs"] = np.sqrt(
            mix["vx"] ** 2.0 + mix["vr"] ** 2.0 + mix["vt"] ** 2.0
        )

        # Lookup compressible flow relation
        V_cpTo = mix["vabs"] / np.sqrt(cp * mix["tstag"])
        Ma = np.sqrt(V_cpTo ** 2.0 / (ga - 1.0) / (1.0 - 0.5 * V_cpTo ** 2.0))
        To_T = 1.0 + 0.5 * (ga - 1.0) * Ma ** 2.0

        # Get static T
        mix["tstat"] = mix["tstag"] / To_T

        # Record mixed out flow condition in primary flow variables
        mix["ro"] = mix["pstat"] / (rgas * mix["tstat"])

    # Max a new cut with the mixed out flow
    cut_out = ts_tstream_cut.TstreamStructuredCut()

    cut_out.pref = cut.pref
    cut_out.tref = cut.tref
    cut_out.ni = 1
    cut_out.nj = 1
    cut_out.nk = 1
    cut_out.ga = ga
    cut_out.cp = cp
    cut_out.ifgas = 0
    cut_out.write_egen = 0

    cut_out.rpm = cuts[0].rpm
    cut_out.x = np.mean(cut.x)
    cut_out.r = rmid
    cut_out.rt = np.mean(cut.rt)
    cut_out.ro = mix["ro"]
    cut_out.rovx = mix["ro"] * mix["vx"]
    cut_out.rovr = mix["ro"] * mix["vr"]
    cut_out.rorvt = mix["ro"] * mix["vt"] * rmid
    cut_out.roe = mix["ro"] * (cv * mix["tstat"] + 0.5 * mix["vabs"] ** 2.0)

    cut_out.tstat = mix["tstat"]
    cut_out.tstag = mix["tstag"]
    cut_out.pstat = mix["pstat"]

    cut_out.vx = mix["vx"]
    cut_out.vr = mix["vr"]
    cut_out.vt = mix["vt"]
    cut_out.vabs = mix["vabs"]
    cut_out.U = rmid * cut_out.rpm / 60.0 * 2.0 * np.pi
    cut_out.vt_rel = mix["vt"] - cut_out.U

    cut_out.vabs_rel = np.sqrt(
        cut_out.vx ** 2.0 + cut_out.vr ** 2.0 + cut_out.vt_rel ** 2.0
    )
    cut_out.mach_rel = cut_out.vabs_rel / np.sqrt(ga * rgas * cut_out.tstat)

    cut_out.mach = cut_out.vabs / np.sqrt(ga * rgas * cut_out.tstat)
    cut_out.pstag = compflow.Po_P_from_Ma(cut_out.mach, ga) * cut_out.pstat
    cut_out.pstag_rel = (
        compflow.Po_P_from_Ma(cut_out.mach_rel, ga) * cut_out.pstat
    )

    return cut_out


def average_cuts(cuts, var_name):
    mass, prop = zip(*[ci.mass_avg_1d(var_name) for ci in cuts])
    mass = np.array(mass)
    prop = np.array(prop)
    return np.sum(mass * prop) / np.sum(mass)


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
    d["t"] = d["rt"] / d["r"]

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
    basedir = os.path.dirname(output_hdf5)
    run_name = os.path.split(os.path.abspath(basedir))[-1]
    print("POST-PROCESSING %s\n" % output_hdf5)

    opts = sys.argv[2:]
    meta_only = "--meta-only" in opts

    # Load the grid
    tsr = ts_tstream_reader.TstreamReader()
    g = tsr.read(output_hdf5)

    # Gas properties
    cp = g.get_av("cp")  # Specific heat capacity at const p
    ga = g.get_av("ga")  # Specific heat ratio
    rgas = cp * (1.0 - 1.0 / ga)

    # Numbers of grid points
    bid_all = np.array(g.get_block_ids())
    blk = [g.get_block(bidn) for bidn in bid_all]
    ni = [blki.ni for blki in blk]
    nj = [blki.nj for blki in blk]
    nk = [blki.nk for blki in blk]
    rpm = np.array([g.get_bv("rpm", bid) for bid in bid_all])

    # Stator/rotor blocks
    bid_stator = bid_all[rpm == 0.0]
    bid_rotor = bid_all[rpm != 0.0]

    # Take cuts at inlet/outlet planes of each row
    stator_inlet = []
    stator_outlet = []
    for bid in bid_stator:
        stator_inlet.append(ts_tstream_cut.TstreamStructuredCut())
        stator_inlet[-1].read_from_grid(
            g,
            Pref,
            Tref,
            bid,
            ist=0,
            ien=1,  # First streamwise
            jst=0,
            jen=nj[0],  # All radial
            kst=0,
            ken=nk[0],  # All pitchwise
        )
        stator_outlet.append(ts_tstream_cut.TstreamStructuredCut())
        stator_outlet[-1].read_from_grid(
            g,
            Pref,
            Tref,
            bid,
            ist=ni[0] - 2,
            ien=ni[0] - 1,  # Last streamwise
            jst=0,
            jen=nj[0],  # All radial
            kst=0,
            ken=nk[0],  # All pitchwise
        )

    rotor_inlet = []
    rotor_outlet = []
    for bid in bid_rotor:
        rotor_inlet.append(ts_tstream_cut.TstreamStructuredCut())
        rotor_inlet[-1].read_from_grid(
            g,
            Pref,
            Tref,
            bid,
            ist=1,
            ien=2,  # First streamwise
            jst=0,
            jen=nj[1],  # All radial
            kst=0,
            ken=nk[1],  # All pitchwise
        )
        rotor_outlet.append(ts_tstream_cut.TstreamStructuredCut())
        rotor_outlet[-1].read_from_grid(
            g,
            Pref,
            Tref,
            bid,
            ist=ni[1] - 2,
            ien=ni[1] - 1,  # Last streamwise
            jst=0,
            jen=nj[1],  # All radial
            kst=0,
            ken=nk[1],  # All pitchwise
        )

    # Pull out mass-average flow varibles from the cuts
    cuts = [
        mix_out(ci)
        for ci in [stator_inlet, stator_outlet, rotor_inlet, rotor_outlet]
    ]
    var_names = [
        "pstag",
        "pstat",
        "tstag",
        "tstat",
        "vx",
        "vt",
        "vt_rel",
        "pstag_rel",
    ]
    Po, P, To, T, Vx, Vt, Vt_rel, Po_rel = [
        np.array([getattr(ci, var_name) for ci in cuts])
        for var_name in var_names
    ]

    # Calculate entropy change with respect to inlet
    ds = cp * np.log(T / Tref) - rgas * np.log(P / Pref)
    ds_ref = ds[0]
    ds = ds - ds_ref

    # Calculate metadata
    meta = {}

    # Polytropic efficiency
    meta["eff_poly"] = (
        ga / (ga - 1.0) * np.log(To[3] / To[0]) / np.log(Po[3] / Po[0])
    )
    meta["eff_isen"] = (To[3] / To[0] - 1.0) / (
        (Po[3] / Po[0]) ** ((ga - 1.0) / ga) - 1.0
    )

    # Flow angles
    meta["alpha"] = np.degrees(np.arctan2(Vt, Vx))
    meta["alpha_rel"] = np.degrees(np.arctan2(Vt_rel, Vx))

    # Lost effy from
    # eta = wx/(wx+Tds) = 1/(1+Tds/wx) approx 1-Tds/wx using Taylor expansion
    # meta["eff_isen_lost"] = T[3] * ds / cp / (To[0] - To[3])
    meta["eff_isen_lost"] = 1.0 - 1.0 / (1.0 + T[3] * ds / cp / (To[0] - To[3]))

    meta["runid"] = run_name

    if meta_only:
        save_meta(meta, basedir)
        quit()

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
    mdot_unst = []
    blockage_unst = []
    Cp = []
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
                ).replace("_avg", "")

                dat_now = read_dat(fname_now, (di, dj, dk))
                dat_now = secondary(dat_now, rpm_now, cp, ga)

                dat_now["eff_lost"] = (
                    -T[3] * (dat_now["ds"] - ds_ref) / cp / (To[3] - To[0])
                )

                # Unsteady lost efficiency at exit if this is a rotor passage
                if not rpm_now == 0.0:

                    # Indexes for LE and TE
                    pitch = np.diff(dat_now["t"][:, 0, (0, -1), 0], 1, 1)
                    tol = 1e-5
                    ile = (
                        np.where(np.abs(pitch / pitch[0] - 1.0) > tol)[0][0] - 1
                    )
                    ite = (
                        np.where(np.abs(pitch / pitch[-1] - 1.0) > tol)[0][-1]
                        + 1
                    )

                    # Pressure coefficient
                    Cp_now = (
                        dat_now["pstat"][ile : (ite + 1), 0, (0, -1), :]
                        - Po_rel[2]
                    ) / (Po_rel[2] - P[2])
                    Cp.append(Cp_now)

                    # Mass average lost effy
                    rt_now = dat_now["rt"][-2, 0, :, :]
                    eff_lost_now = dat_now["eff_lost"][-2, 0, :, :]
                    rovx_now = dat_now["rovx"][-2, 0, :, :]
                    mdot_now = np.trapz(rovx_now, rt_now, axis=0)
                    eff_av_now = (
                        np.trapz(rovx_now * eff_lost_now, rt_now, axis=0)
                        / mdot_now
                    )

                    # blockage

                    # Get the rovx distribution at leading edge
                    rovx_le = dat_now["rovx"][ile, 0, :, :]
                    rt_le = dat_now["rt"][ile, 0, :, :]

                    # Mass flow is integral rovx drt
                    mdot_le = np.trapz(rovx_le, rt_le, axis=0)

                    # Area is integral drt
                    A_le = np.trapz(np.ones_like(rt_le), rt_le, axis=0)

                    # Reference mass flux is total mass over total area
                    # Or the mass-averaged rovx
                    rovx_ref = mdot_le / A_le

                    # Blockage is integral 1 - rovx/rover_ref drt
                    blockage_now = np.trapz(
                        1.0 - rovx_now / rovx_ref, rt_now, axis=0
                    )

                    mdot_unst.append(mdot_now)
                    eff_lost_unst.append(eff_av_now)
                    blockage_unst.append(blockage_now)

                    # # For one rotor passage only, look at incidence
                    # # Mass average inlet tangential momentum
                    # rorvt_now = dat_now["rorvt"][-2, 0, :, :]
                    # rorvt_av_now = np.trapz(rorvt_now*rovx_now, rt_now, axis=0)/mdot_now

                # Remove variables we are not interested in
                for k in list(dat_now.keys()):
                    if not k in varnames:
                        dat_now.pop(k)
                    else:
                        # Truncate to one cycle
                        dat_now[k] = dat_now[k][..., :72]

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

Cp = np.array(Cp)
eff_lost_unst = np.array(eff_lost_unst)
mdot_unst = np.array(mdot_unst)
blockage_unst = np.array(blockage_unst)

# Take mass av of each rotor passage effy
meta["eff_lost_unst"] = (
    np.sum(eff_lost_unst * mdot_unst, 0) / np.sum(mdot_unst, 0)
)[:]
meta["mdot_unst"] = np.sum(mdot_unst, 0)[:]
meta["blockage_unst"] = np.mean(blockage_unst, 0)[:]

x_c = dat_now["x"][ile : (ite + 1), 0, 0, 0]
x_c = (x_c - x_c[0]) / np.ptp(x_c)

# Save with unsteady metadata
save_meta(meta, basedir)

# Determine number of stators and rotors
rpms = np.array([g.get_bv("rpm", bidi) for bidi in g.get_block_ids()])
nstator = np.sum(rpms == 0.0)
nrotor = np.sum(rpms != 0.0)

# Join the grid points from all probes together
var_out = np.concatenate(vars_all, axis=1)

np.savez_compressed(
    os.path.join(basedir, "dbslice"),
    data=var_out,
    sizes=dijk_all,
    nsr=(nstator, nrotor),
    Cp=Cp,
    x_c=x_c,
)
