"""Post process a steady Turbostream solution"""
import numpy as np
import compflow_native as compflow
import sys, os, json
from ts import ts_tstream_reader, ts_tstream_cut

Pref = 1e5
Tref = 300.0

# Choose which variables to write out
varnames = ["x", "rt", "eff_lost", "pfluc"]


class suppress_print:
    """A context manager that temporarily sets STDOUT to /dev/null."""

    def __enter__(self):
        self.orig_out = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.orig_out


def node_to_face(cut, prop_name):
    """For a (n,m) matrix of some property, average over the four corners of
    each face to produce an (n-1,m-1) matrix of face-centered properties."""
    return np.mean(
        np.stack(
            (
                getattr(cut, prop_name)[:-1, :-1].astype(float),
                getattr(cut, prop_name)[1:, 1:].astype(float),
                getattr(cut, prop_name)[:-1, 1:].astype(float),
                getattr(cut, prop_name)[1:, :-1].astype(float),
            )
        ),
        axis=0,
    )


def face_length_vec(c):
    """For a matrix of coordinates, get face length matrices along each dim."""
    return c[1:, 1:] - c[:-1, :-1], c[:-1, 1:] - c[1:, :-1]


def face_area(cut):
    """Calculate x and r areas for all cells in a cut."""
    (dx1, dx2), (dr1, dr2), (drt1, drt2) = [
        face_length_vec(c) for c in (cut.x, cut.r, cut.rt)
    ]

    Ax = 0.5 * (dr1 * drt2 - dr2 * drt1)
    Ar = 0.5 * (dx2 * drt1 - dx1 * drt2)

    return Ax, Ar


def mix_out(cuts):
    """Take a number of structured cuts and mix out the flow at constant A."""

    # If only one cut is input, make a trivial array
    try:
        cuts[0]
    except AttributeError:
        cuts = [
            cuts,
        ]

    # Gas properties
    cp = cuts[0].cp
    ga = cuts[0].ga
    rgas = cp * (ga - 1.0) / ga
    cv = cp / ga

    # Identify the flow properties and fluxes we will need for the calculation
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
        cell = {prop: node_to_face(cut, prop) for prop in props}

        # Cell areas
        Ax, Ar = face_area(cut)

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
    for i in range(20):

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

    cut_out.yaw = np.degrees(np.arctan2(cut_out.vt, cut_out.vx))
    cut_out.yaw_rel = np.degrees(np.arctan2(cut_out.vt_rel, cut_out.vx))

    cut_out.entropy = cp * np.log(cut_out.tstat / cut_out.tref) - rgas * np.log(
        cut_out.pstat / cut_out.pref
    )

    return cut_out


def cut_by_indices(g, bid, ijk_sten):
    """Structured cut from grid, allowing for end indices with -1.

    Parameters
    ----------
    g :
        The Turbostream grid object.
    bid : int
        Block id in which to take cut.
    ijk_sten: (3, 2) array, int
        First col ijk start indices, second col ijk end indices.

    """

    # Assemble end indices nijk for desired block
    blk = g.get_block(bid)
    nijk = np.tile([blk.ni, blk.nj, blk.nk], (2, 1)).T

    # Correct the end indices
    ijk_sten = np.array(ijk_sten)
    ijk_sten[ijk_sten < 0] = nijk[ijk_sten < 0] + ijk_sten[ijk_sten < 0]

    ijk_sten[:, 1] += 1

    cut = ts_tstream_cut.TstreamStructuredCut()
    cut.read_from_grid(g, Pref, Tref, bid, *ijk_sten.flat)

    return cut


def cut_rows_mixed(g):
    """Mixed-out cuts at row inlet and exit"""
    ind_all = [0, -1]
    di = 2
    ind_in = [di, di]
    ind_out = [-1 - di, -1 - di]
    cuts = [
        cut_by_indices(g, 0, [ind_in, ind_all, ind_all]),
        cut_by_indices(g, 0, [ind_out, ind_all, ind_all]),
        cut_by_indices(g, 1, [ind_in, ind_all, ind_all]),
        cut_by_indices(g, 1, [ind_out, ind_all, ind_all]),
    ]
    return [mix_out(c) for c in cuts]


def _integrate_length(chi):
    """Integrate quadratic camber line length given angles."""
    xhat = np.linspace(0.0, 1.0)
    tanchi_lim = np.tan(np.radians(chi))
    tanchi = np.diff(tanchi_lim) * xhat + tanchi_lim[0]
    return np.trapz(np.sqrt(1.0 + tanchi ** 2.0), xhat)


def find_chord(g, bid):
    """Determine axial chord of a row."""
    x, r, rt = [
        np.swapaxes(g.get_bp(vi, bid), 0, -1)[:, 2, (0, -1)]
        for vi in ["x", "r", "rt"]
    ]
    dt = np.diff(rt / r, 1, axis=1).flat
    pitch = dt[0]
    is_blade = dt / pitch < 0.995
    ile = np.argmax(is_blade)
    is_blade_2 = dt / pitch < 0.995
    is_blade_2[: (ile + 1)] = True
    ite = np.argmax(~is_blade_2)
    cx = x[ile:ite, 0].ptp()
    return cx, ile, ite, pitch


def extract_surf(g, bid):
    cx, ile, ite, _ = find_chord(g, bid)
    C = cut_by_indices(g, bid, [[ile, ite], [2, 2], [0, -1]])
    P = np.moveaxis(C.pstat, 0, -1)[:, (0, -1)].astype(float)
    x = np.moveaxis(C.x, 0, -1)[:, (0, -1)]
    rt = np.moveaxis(C.rt, 0, -1)[:, (0, -1)]
    surf = np.cumsum(
        np.sqrt(np.diff(x, 1, 0) ** 2.0 + np.diff(rt, 1, 0) ** 2.0), axis=0
    )
    surf = np.insert(surf, 0, np.zeros((1, 2)), axis=0).astype(float)
    return surf, P


def circ_coeff(g, bid, Po1, P2):
    surf, P = extract_surf(g, bid)
    Cp = (Po1 - P) / (Po1 - P2)
    Cp[Cp < 0.0] = 0.0
    # Normalise distance
    surfn = surf / surf[(-1,), :]
    side = [
        np.trapz(np.sqrt(Cpi), surfi, axis=0)
        for Cpi, surfi in zip(Cp.T, surfn.T)
    ]
    if g.get_bv("rpm", bid):
        return side[0] - side[1], surf[-1, 0]
    else:
        return side[1] - side[0], surf[-1, 1]


# This file is called as a script from the SLURM job script
if __name__ == "__main__":

    # Get file paths
    output_hdf5 = sys.argv[1]
    if not os.path.isfile(output_hdf5):
        raise IOError("%s not found." % output_hdf5)

    basedir = os.path.dirname(output_hdf5)
    run_name = os.path.split(os.path.abspath(basedir))[-1]

    # Load the flow solution, supressing noisy printing
    tsr = ts_tstream_reader.TstreamReader()
    with suppress_print():
        g = tsr.read(output_hdf5)

    # Gas properties
    cp = g.get_av("cp")  # Specific heat capacity at const p
    ga = g.get_av("ga")  # Specific heat ratio
    rgas = cp * (1.0 - 1.0 / ga)
    rpm = g.get_bv("rpm", 1)
    omega = rpm / 60.0 * 2.0 * np.pi

    # 1D mixed-out average cuts for stator/rotor inlet/outlet
    cut_all = cut_rows_mixed(g)
    sta_in, sta_out, rot_in, rot_out = cut_all

    # Calculate stage loading coefficient
    U = omega * rot_in.r
    Psi = cp * (sta_in.tstag - rot_out.tstag) / U ** 2.0

    # Polytropic efficiency
    eff_poly = (
        ga
        / (ga - 1.0)
        * np.log(rot_out.tstag / sta_in.tstag)
        / np.log(rot_out.pstag / sta_in.pstag)
    )
    eff_isen = (rot_out.tstag / sta_in.tstag - 1.0) / (
        (rot_out.pstag / sta_in.pstag) ** ((ga - 1.0) / ga) - 1.0
    )

    # Reaction
    Lam = (rot_out.tstat - rot_in.tstat) / (rot_out.tstat - sta_in.tstat)

    # Flow angles
    Al = [ci.yaw for ci in cut_all]
    Al_rel = [ci.yaw_rel for ci in cut_all]

    # Viscosity
    if g.get_av("viscosity_law"):
        muref = g.get_av("viscosity")
        Tref = 288.0
        expon = 0.62
        T2 = sta_out.tstat
        mu2 = muref * (T2 / Tref) ** expon
    else:
        mu2 = g.get_av("viscosity")

    # Reynolds num
    ro2 = sta_out.ro
    V2 = sta_out.vabs
    cx = find_chord(g, 0)[0]
    Re_cx = ro2 * V2 * cx / mu2
    ell_cx = _integrate_length(Al[:2])
    Re_ell = Re_cx * ell_cx

    # Circulation coefficient
    Cov, Sov = circ_coeff(g, 0, sta_in.pstag, sta_out.pstat)
    Cob, Sob = circ_coeff(g, 1, rot_in.pstag_rel, rot_out.pstat)
    Re_So = Re_cx * Sov / cx
    Co = [Cov, Cob]

    # Pitch to chord
    pitch_rt = np.array([find_chord(g, bid)[3] for bid in [0, 1]])
    cx_all = np.array([find_chord(g, bid)[0] for bid in [0, 1]])
    s_cx = (pitch_rt / cx_all).tolist()

    # Loss coefficients
    Ypv = (sta_in.pstag - sta_out.pstag) / (sta_in.pstag - sta_out.pstat)
    Ypb = (rot_in.pstag_rel - rot_out.pstag_rel) / (
        rot_in.pstag_rel - rot_out.pstat
    )
    Yp = [Ypv, Ypb]

    # Axial velocity ratio
    zeta = (sta_in.vx / sta_out.vx, rot_out.vx / sta_out.vx)

    # Convergence residual
    resid_str = os.popen("grep 'TOTAL DAVG' log.txt | tail -10 | cut -d ' ' -f3").read()
    resid = np.array([float(ri) for ri in resid_str.splitlines()]).mean()

    # Save metadata in dict
    meta = {
        "Al": Al,
        "Alrel": Al_rel,
        "psi": Psi,
        "eta": eff_poly,
        "eta_lost": 1.0 - eff_poly,
        "eta_isen": eff_isen,
        "runid": run_name,
        "Ma2": sta_out.mach,
        "phi": sta_out.vx / U,
        "Lam": Lam,
        "Re": Re_ell,
        "Re_cx": Re_cx,
        "Re_So": Re_So,
        "Co": Co,
        "resid": resid,
        "s_cx": s_cx,
        "Yp": Yp,
        "zeta": zeta,
    }

    with open(os.path.join(basedir, "meta.json"), "w") as f:
        json.dump(meta, f)
