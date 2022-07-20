"""Functions for exporting a stage design to Turbostream."""
import numpy as np
from ts import (
    ts_tstream_type,
    ts_tstream_grid,
    ts_tstream_patch_kind,
    ts_tstream_default,
    ts_tstream_load_balance,
    ts_tstream_reader,
    ts_autogrid_reader,
)
import compflow
from . import design, ohmesh
import sys, os
import subprocess
from scipy.spatial import KDTree

muref = 1.8e-5


def get_free_gpu_id():
    """Parse output of nvidia-smi to find a vacant GPU."""
    output = subprocess.check_output(
        "nvidia-smi pmon -c 1 | tail -4 | tr -s ' ' | cut -d ' ' -f2-3",
        shell=True,
    )
    for l in output.splitlines():
        gpu_id, pid = l.split()
        if pid == "-":
            return int(gpu_id)


def _make_patch(kind, bid, i, j, k, nxbid=0, nxpid=0, dirs=None):
    # Periodic patches
    p = ts_tstream_type.TstreamPatch()

    p.kind = getattr(ts_tstream_patch_kind, kind)

    p.bid = bid

    p.ist, p.ien = i
    p.jst, p.jen = j
    p.kst, p.ken = k

    p.nxbid = nxbid
    p.nxpid = nxpid

    if dirs is not None:
        p.idir, p.jdir, p.kdir = dirs
    else:
        p.idir, p.jdir, p.kdir = (0, 1, 2)

    p.nface = 0
    p.nt = 1

    return p


def apply_inlet(g, bid, pid, Poin, Toin):

    patch = g.get_patch(bid, pid)
    nk = patch.ken-patch.kst
    nj = patch.jen-patch.jst

    yaw = np.zeros((nk, nj), np.float32)
    pitch = np.zeros((nk, nj), np.float32)
    pstag = np.zeros((nk, nj), np.float32)
    tstag = np.zeros((nk, nj), np.float32)

    pstag += Poin
    tstag += Toin
    yaw += 0.0
    pitch += 0.0

    g.set_pp("yaw", ts_tstream_type.float, bid, pid, yaw)
    g.set_pp("pitch", ts_tstream_type.float, bid, pid, pitch)
    g.set_pp("pstag", ts_tstream_type.float, bid, pid, pstag)
    g.set_pp("tstag", ts_tstream_type.float, bid, pid, tstag)
    g.set_pv("rfin", ts_tstream_type.float, bid, pid, 0.5)
    g.set_pv("sfinlet", ts_tstream_type.float, bid, pid, 0.1)

def find_spanwise_patches(g):

    all_pairs = []
    for bid in g.get_block_ids():
        patch_pair = []
        for pid in g.get_patch_ids(bid):
            patch = g.get_patch(bid, pid)
            print(patch.jst, patch.jen)
            if (patch.jst + 1) == patch.jen:
                print('beans')
                patch_pair.append(patch)
        if len(patch_pair):
            all_pairs.append(patch_pair)

    return all_pairs

# def slipwall_to_periodic(g):
#     """Set slipwalls to periodic for poisson calc."""
#     for P in find_spanwise_patches(g):
#         P[0].kind = ts_tstream_patch_kind.periodic
#         P[1].kind = ts_tstream_patch_kind.periodic
#         P[0].nxpid = P[1].pid
#         P[0].nxbid = P[1].bid
#         P[1].nxpid = P[0].pid
#         P[1].nxbid = P[0].bid
#         P[0].idir = 0
#         P[1].idir = 0
#         P[0].jdir = -1
#         P[1].jdir = -1
#         P[0].kdir = 2
#         P[1].kdir = 2


def periodic_to_slipwall(g):
    """Set spanwise periodic to slipwall for main calc."""
    # Hard code everything for now
    for P in find_spanwise_patches(g):
        print(P)
        P[0].kind = ts_tstream_patch_kind.slipwall
        P[1].kind = ts_tstream_patch_kind.slipwall
        P[0].nxpid = P[1].pid
        P[0].nxbid = P[1].bid
        P[1].nxpid = P[0].pid
        P[1].nxbid = P[0].bid
        P[0].idir = 0
        P[1].idir = 0
        P[0].jdir = -1
        P[1].jdir = -1
        P[0].kdir = 2
        P[1].kdir = 2



def add_to_grid(g, xin, rin, rtin, ilte):
    """From mesh coordinates, add a block with patches to TS grid object"""
    ni, nj, nk = np.shape(rtin)
    rt = rtin + 0.0
    r = np.repeat(rin[:, :, None], nk, axis=2)
    if xin.ndim == 3:
        x = xin + 0.0
    else:
        x = np.tile(xin[:, None, None], (1, nj, nk))

    # Permute the coordinates into C-style ordering
    # Turbostream is very fussy about this
    xp = np.zeros((nk, nj, ni), np.float32)
    rp = np.zeros((nk, nj, ni), np.float32)
    rtp = np.zeros((nk, nj, ni), np.float32)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                xp[k, j, i] = x[i, j, k]
                rp[k, j, i] = r[i, j, k]
                rtp[k, j, i] = rt[i, j, k]

    # Generate new block
    bid = g.get_nb()
    b = ts_tstream_type.TstreamBlock()
    b.bid = bid
    b.ni, b.nj, b.nk = ni, nj, nk
    b.np = 0
    b.procid = 0
    b.threadid = 0

    # Add to grid and set coordinates as block properties
    g.add_block(b)
    for vname, vval in zip(["x", "r", "rt"], [xp, rp, rtp]):
        g.set_bp(vname, ts_tstream_type.float, bid, vval)

    # Leading and trailing edges
    ile, ite = ilte

    # Add periodic patches first

    # Upstream of LE
    periodic_up_1 = _make_patch(
        kind="periodic",
        bid=bid,
        i=(0, ile + 1),
        j=(0, nj),
        k=(0, 1),
        dirs=(0, 1, 6),
        nxbid=bid,
        nxpid=1,
    )
    periodic_up_2 = _make_patch(
        kind="periodic",
        bid=bid,
        i=(0, ile + 1),
        j=(0, nj),
        k=(nk - 1, nk),
        dirs=(0, 1, 6),
        nxbid=bid,
        nxpid=0,
    )
    periodic_up_1.pid = g.add_patch(bid, periodic_up_1)
    periodic_up_2.pid = g.add_patch(bid, periodic_up_2)

    # Downstream of TE
    periodic_dn_1 = _make_patch(
        kind="periodic",
        bid=bid,
        i=(ite, ni),
        j=(0, nj),
        k=(0, 1),
        dirs=(0, 1, 6),
        nxbid=bid,
        nxpid=3,
    )
    periodic_dn_2 = _make_patch(
        kind="periodic",
        bid=bid,
        i=(ite, ni),
        j=(0, nj),
        k=(nk - 1, nk),
        dirs=(0, 1, 6),
        nxbid=bid,
        nxpid=2,
    )
    periodic_dn_1.pid = g.add_patch(bid, periodic_dn_1)
    periodic_dn_2.pid = g.add_patch(bid, periodic_dn_2)

    # Add slip walls if this is a cascade
    htr = r[0, 1, 0] / r[0, 0, 0]
    if htr > 0.95:
        slip_j0 = _make_patch(
            kind="slipwall", bid=bid, i=(0, ni), j=(0, 1), k=(0, nk)
        )
        slip_nj = _make_patch(
            kind="slipwall", bid=bid, i=(0, ni), j=(nj - 1, nj), k=(0, nk)
        )
        slip_j0.pid = g.add_patch(bid, slip_j0)
        slip_nj.pid = g.add_patch(bid, slip_nj)

    # Default avs
    for name in ts_tstream_default.av:
        val = ts_tstream_default.av[name]
        if type(val) == type(1):
            g.set_av(name, ts_tstream_type.int, val)
        else:
            g.set_av(name, ts_tstream_type.float, val)

    # Default bvs
    for name in ts_tstream_default.bv:
        for bid in g.get_block_ids():
            val = ts_tstream_default.bv[name]
            if type(val) == type(1):
                g.set_bv(name, ts_tstream_type.int, bid, val)
            else:
                g.set_bv(name, ts_tstream_type.float, bid, val)


def guess_block(g, bid, x, Po, To, Ma, Al, ga, rgas):
    b = g.get_block(bid)
    ni, nj, nk = (b.ni, b.nj, b.nk)
    xb = g.get_bp("x", bid)
    rb = g.get_bp("r", bid)
    cp = rgas / (ga - 1.0) * ga
    cv = cp / ga

    # Interpolate guess to block coords
    Pob = np.interp(xb, x, Po)
    Tob = np.interp(xb, x, To)
    Mab = np.interp(xb, x, Ma)
    Alb = np.interp(xb, x, Al)

    # Get velocities
    Vb = compflow.V_cpTo_from_Ma(Mab, ga) * np.sqrt(cp * Tob)
    Vxb = Vb * np.cos(np.radians(Alb))
    Vtb = Vb * np.sin(np.radians(Alb))
    Vrb = np.zeros_like(Vb)

    # Static pressure and temperature
    Pb = Pob / compflow.Po_P_from_Ma(Mab, ga)
    Tb = Tob / compflow.To_T_from_Ma(Mab, ga)

    # Density
    rob = Pb / rgas / Tb

    # Energy
    eb = cv * Tb + (Vb ** 2.0) / 2

    # Primary vars
    rovxb = rob * Vxb
    rovrb = rob * Vrb
    rorvtb = rob * rb * Vtb
    roeb = rob * eb

    # Permute the coordinates into C-style ordering
    robp = np.zeros((nk, nj, ni), np.float32)
    rovxbp = np.zeros((nk, nj, ni), np.float32)
    rovrbp = np.zeros((nk, nj, ni), np.float32)
    rorvtbp = np.zeros((nk, nj, ni), np.float32)
    roebp = np.zeros((nk, nj, ni), np.float32)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                robp[k, j, i] = rob[k, j, i]
                rovxbp[k, j, i] = rovxb[k, j, i]
                rovrbp[k, j, i] = rovrb[k, j, i]
                rorvtbp[k, j, i] = rorvtb[k, j, i]
                roebp[k, j, i] = roeb[k, j, i]

    # Apply to grid
    g.set_bp("ro", ts_tstream_type.float, bid, robp)
    g.set_bp("rovx", ts_tstream_type.float, bid, rovxbp)
    g.set_bp("rovr", ts_tstream_type.float, bid, rovrbp)
    g.set_bp("rorvt", ts_tstream_type.float, bid, rorvtbp)
    g.set_bp("roe", ts_tstream_type.float, bid, roebp)

def cut_by_patch(g, bid, pid):
    """Structured cut a patch from grid."""

    P = g.get_patch(bid, pid)
    x = g.get_bp("x",bid)[P.kst:P.ken, P.jst:P.jen, P.ist:P.ien]
    r = g.get_bp("r",bid)[P.kst:P.ken, P.jst:P.jen, P.ist:P.ien]
    rt = g.get_bp("rt",bid)[P.kst:P.ken, P.jst:P.jen, P.ist:P.ien]

    return x, r, rt

def find_omesh(g):
    oblocks = []
    for b in g.get_block_ids():
        for p in g.get_patch_ids(b):
            P = g.get_patch(b,p)
            if (P.kind == ts_tstream_patch_kind.periodic) and (P.nxbid==b):
                x1, r1, rt1 = cut_by_patch(g, b, p)
                x2, r2, rt2 = cut_by_patch(g, P.nxbid, P.nxpid)
                if np.isclose(np.mean(rt1), np.mean(rt2)):
                    oblocks.append(b)
                    break

    return oblocks

def cut_by_blade(g, row_ind):
    bid_o = find_omesh(g)[row_ind]
    x = g.get_bp("x",bid_o)[0, :, :].T
    r = g.get_bp("r",bid_o)[0, :, :].T
    rt = g.get_bp("rt",bid_o)[0, :, :].T

    return x, r, rt

def wall_distance(g):
    """Set wall distance field using KDTree."""

    print('getting coords')

    stator_bids = [i for i in range(7)]
    rotor_bids = [i for i in range(7,len(g.get_block_ids()))]

    row_bids = [stator_bids, rotor_bids]

    for irow in range(2):

        x, r, rt = cut_by_blade(g,irow)

        t = rt/r

        xrt = np.column_stack([x.reshape(-1),r.reshape(-1),t.reshape(-1)])
        print(xrt.shape)

        # get pitch for this row
        bid_o = find_omesh(g)[irow]
        nb = g.get_bv("nblade",bid_o)
        pitch = 2.*np.pi/float(nb)

        rmax = r.max()
        dmax = pitch*rmax*0.05

        # duplicate blades
        xrt_1 = xrt+0.
        xrt_1[:,2] += pitch
        xrt_2 = xrt+0.
        xrt_2[:,2] -= pitch

        xrt = np.concatenate((xrt,xrt_1,xrt_2))
        print(xrt.shape)

        # Convert to polar
        x = xrt[:,0]
        y = xrt[:,1]*np.sin(xrt[:,2])
        z = xrt[:,1]*np.cos(xrt[:,2])
        xyz = np.column_stack((x,y,z))

        print('x',xyz[:,0].min(), xyz[:,0].max())
        print('y',xyz[:,1].min(), xyz[:,1].max())
        print('z',xyz[:,2].min(), xyz[:,2].max())

        # Make kdtree
        print('making kdtree, row %d'% irow)
        kdt = KDTree(xyz,leafsize=5)


        for bid in row_bids[irow]:
            print('setting phi, bid %d' % bid)

            x_b = g.get_bp("x",bid)
            r_b = g.get_bp("r",bid)
            rt_b = g.get_bp("rt",bid)
            t_b = rt_b/r_b
            y_b = r_b*np.sin(t_b)
            z_b = r_b*np.cos(t_b)
            xyz_b = np.column_stack((x_b.reshape(-1),y_b.reshape(-1),z_b.reshape(-1)))
            wdist = kdt.query(xyz_b, distance_upper_bound=dmax)[0].reshape(x_b.shape).astype(np.float32)
            wdist[np.isinf(wdist)] = dmax
            print(wdist.min(),wdist.max())
            g.set_bp("phi", ts_tstream_type.float, bid, wdist)

    return g


def set_variables(g, mu=None):
    """Set application and block variables on a TS grid object."""

    for bid in g.get_block_ids():
        g.set_bv("fmgrid", ts_tstream_type.float, bid, 0.)
        g.set_bv("poisson_fmgrid", ts_tstream_type.float, bid, 0.00)
        g.set_bv("xllim_free", ts_tstream_type.float, bid, 0.0)
        g.set_bv("free_turb", ts_tstream_type.float, bid, 0.05)
        g.set_bv("fsturb", ts_tstream_type.float, bid, 1.0)

    g.set_av("restart", ts_tstream_type.int, 1)
    g.set_av("ilos", ts_tstream_type.int, 1)
    g.set_av("nlos", ts_tstream_type.int, 5)
    g.set_av("nstep", ts_tstream_type.int, 50000)
    g.set_av("nstep_save_start", ts_tstream_type.int, 45000)
    g.set_av("nchange", ts_tstream_type.int, 5000)
    g.set_av("sfin", ts_tstream_type.float, 0.5)
    g.set_av("facsecin", ts_tstream_type.float, 0.005)
    g.set_av("cfl", ts_tstream_type.float, 0.4)
    g.set_av("poisson_cfl", ts_tstream_type.float, 0.7)
    g.set_av("fac_stmix", ts_tstream_type.float, 0.0)
    g.set_av("rfmix", ts_tstream_type.float, 0.01)
    g.set_av("write_yplus", ts_tstream_type.int, 1)
    g.set_av("nlos", ts_tstream_type.int, 5)
    g.set_av("rfvis", ts_tstream_type.float, 0.2)

    g.set_av("smooth_scale_dts_option", ts_tstream_type.int, 0)

    g.set_av("poisson_restart", ts_tstream_type.int, 0)
    g.set_av("poisson_nsmooth", ts_tstream_type.int, 10)
    g.set_av("poisson_sfin", ts_tstream_type.float, 0.02)
    g.set_av("poisson_nstep", ts_tstream_type.float, 5000)


    if mu:
        # Set a constant viscosity if one is specified
        g.set_av("viscosity", ts_tstream_type.float, mu)
        g.set_av("viscosity_law", ts_tstream_type.int, 0)
    else:
        # Otherwise, use power law
        g.set_av("viscosity", ts_tstream_type.float, muref)
        g.set_av("viscosity_law", ts_tstream_type.int, 1)


def set_rotation(g, bids, rpm, spin_j):
    for bid in bids:
        g.set_bv("rpm", ts_tstream_type.float, bid, rpm)
        g.set_bv("rpmi1", ts_tstream_type.float, bid, rpm)
        g.set_bv("rpmi2", ts_tstream_type.float, bid, rpm)
        if spin_j:
            g.set_bv("rpmj1", ts_tstream_type.float, bid, rpm)
            g.set_bv("rpmj2", ts_tstream_type.float, bid, rpm)
        g.set_bv("rpmk1", ts_tstream_type.float, bid, rpm)
        g.set_bv("rpmk2", ts_tstream_type.float, bid, rpm)


def set_xllim(g, frac):
    # Mixing length limit
    for bid in g.get_block_ids():
        nb = g.get_bv("nblade", bid)
        rnow = g.get_bp("r", bid)
        rm = np.mean([rnow.max(), rnow.min()])
        pitch = 2.0 * np.pi * rm / float(nb)
        g.set_bv("xllim", ts_tstream_type.float, bid, frac * pitch)


def find_patches(g, kind):
    bid = []
    pid = []
    for b in g.get_block_ids():
        for p in g.get_patch_ids(b):

            if g.get_patch(b,p).kind == kind:
                bid.append(b)
                pid.append(p)

    if len(bid)==1:
        bid, pid = bid[0], pid[0]

    return bid, pid


def prep_ag(g, x, ilte, nb, stg, Po1, To1, Omega, rgas, guess_file, ilos, nchange):

    #
    # First, set variables
    #

    # Default avs
    for name in ts_tstream_default.av:
        val = ts_tstream_default.av[name]
        if type(val) == type(1):
            g.set_av(name, ts_tstream_type.int, val)
        else:
            g.set_av(name, ts_tstream_type.float, val)

    # Default bvs
    for name in ts_tstream_default.bv:
        for bid in g.get_block_ids():
            val = ts_tstream_default.bv[name]
            if type(val) == type(1):
                g.set_bv(name, ts_tstream_type.int, bid, val)
            else:
                g.set_bv(name, ts_tstream_type.float, bid, val)

    # Apply application/block variables
    set_variables(g)
    g.set_av("dampin", ts_tstream_type.float, 10. if ilos == 1 else 25.)
    g.set_av("ilos", ts_tstream_type.float, ilos)

    g.set_av("nchange", ts_tstream_type.int, nchange)

    cp = rgas * stg.ga / (stg.ga - 1.0)
    g.set_av("ga", ts_tstream_type.float, stg.ga)
    g.set_av("cp", ts_tstream_type.float, cp)

    # g.set_av("nstep", ts_tstream_type.int, 10e3 if ilos == 1 else 60e3)
    # g.set_av("nstep_save_start", ts_tstream_type.int, 5e3 if ilos == 1 else 55e3)

    # No multigrid at mixing planes
    bid_mix, _  = find_patches(g, ts_tstream_patch_kind.mixing)
    for b in bid_mix:
        g.set_bv("fmgrid", ts_tstream_type.float, b, 0.)

    # 
    # Boundary conditions
    #

    # Inlet
    bid_in, pid_in = find_patches(g, ts_tstream_patch_kind.inlet)
    apply_inlet(g, bid_in, pid_in, Po1, To1)

    # Outlet
    bid_out, pid_out = find_patches(g, ts_tstream_patch_kind.outlet)
    P3 = Po1 * stg.P3_Po1
    g.set_pv("throttle_type", ts_tstream_type.int, bid_out, pid_out, 0)
    g.set_pv("ipout", ts_tstream_type.int, bid_out, pid_out, 3)
    g.set_pv("pout", ts_tstream_type.float, bid_out, pid_out, P3)

    # Kill turbulence in outlet for better convergence
    g.set_bv("fac_st0", ts_tstream_type.float, bid_out, 0.)

    stator_bids = [i for i in range(7)]
    rotor_bids = [i for i in range(7,len(g.get_block_ids()))]

    for bid in stator_bids:
        g.set_bv("fblade", ts_tstream_type.float, bid, float(nb[0]))
        g.set_bv("nblade", ts_tstream_type.int, bid, nb[0])
    for bid in rotor_bids:
        g.set_bv("fblade", ts_tstream_type.float, bid, float(nb[1]))
        g.set_bv("nblade", ts_tstream_type.int, bid, nb[1])

    # Rotation
    # Swap rotation if compressor
    if stg.psi < 0.:
        stator_bids, rotor_bids = rotor_bids, stator_bids
    rpm_rotor = Omega / 2.0 / np.pi * 60.0
    set_rotation( g, stator_bids, 0., spin_j=False,)
    set_rotation( g, rotor_bids, rpm_rotor, spin_j=True,)

    # Mixing length limit
    set_xllim(g, 0.03)

    # Add spanwise periodics if this is a cascade (but possibly convert them to
    # slipwalls later)
    r = g.get_bp("r",0)
    htr = r.min() / r.max()
    if htr > 0.95:
        for bid in g.get_block_ids():
            blk = g.get_block(bid)

            span_j0 = _make_patch(
                kind="slipwall", bid=bid, i=(0, blk.ni), j=(0, 1), k=(0, blk.nk)
            )
            span_nj = _make_patch(
                kind="slipwall", bid=bid, i=(0, blk.ni), j=(blk.nj - 1, blk.nj), k=(0, blk.nk)
            )


            span_j0.pid = g.add_patch(bid, span_j0)
            span_nj.pid = g.add_patch(bid, span_nj)

            print('added slipwall')

    # Load initial guess from file if provided
    if guess_file:

        # Use existing Poisson field
        g.set_av("poisson_nstep", ts_tstream_type.int, 0)
        g.set_av("poisson_restart", ts_tstream_type.int, 1)

        tsr = ts_tstream_reader.TstreamReader()
        gg = tsr.read(guess_file)

        # Copy varaibles
        for var in ["ro", "rovx", "rovr", "rorvt", "roe", "trans_dyn_vis", "phi"]:
            for bid in g.get_block_ids():
                g.set_bp(var, ts_tstream_type.float, bid, gg.get_bp(var, bid))

        # Throw out spurious negative wall distances
        for bid in g.get_block_ids():
            phi = g.get_bp('phi',bid)
            if np.any(phi[:]<0.):
                phi[phi<0.] = 0.
                g.set_bp('phi', ts_tstream_type.float, bid, phi)

    else:

        # Apply an initial guess by linear interpolation in  x

        xg = np.concatenate( [ x[0][ [ 0, ] + ilte[0], ], x[1][ ilte[1] + [ -1, ], ] ])
        Pog = np.repeat(stg.Po_Po1 * Po1, 2)
        Tog = np.repeat(stg.To_To1 * To1, 2)
        Mag = np.repeat(stg.Ma, 2)
        Alg = np.repeat(stg.Al, 2)

        for bid in g.get_block_ids():
            guess_block(g, bid, xg, Pog, Tog, Mag, Alg, stg.ga, rgas)

    ts_tstream_load_balance.load_balance(g, 1, 1.0)

    return g



def make_grid(
    stg, x, r, rt, ilte, Po1, To1, Omega, rgas, guess_file, dampin, ilos
):

    # Make grid, add the blocks
    g = ts_tstream_grid.TstreamGrid()
    for args in zip(x, r, rt, ilte):
        add_to_grid(g, *args)

    bid_vane = 0
    bid_blade = 1

    # calc nb
    t = [rti / ri[..., None] for rti, ri in zip(rt, r)]
    nb = [
        np.asscalar(np.round(2.0 * np.pi / np.diff(ti[0, 0, (0, -1)], 1)))
        for ti in t
    ]
    nb_int = [int(nbi) for nbi in nb]

    ni, nj, nk = zip(*[rti.shape for rti in rt])

    # Inlet
    inlet = _make_patch(
        kind="inlet",
        bid=bid_vane,
        i=(0, 1),
        j=(0, nj[0]),
        k=(0, nk[0]),
        dirs=(6, 1, 2),
    )
    inlet.pid = g.add_patch(bid_vane, inlet)
    apply_inlet(g, bid_vane, inlet.pid, Po1, To1, nk[0], nj[0])

    # Outlet
    outlet = _make_patch(
        kind="outlet",
        bid=bid_blade,
        i=(ni[1] - 1, ni[1]),
        j=(0, nj[1]),
        k=(0, nk[1]),
    )

    outlet.pid = g.add_patch(bid_blade, outlet)
    g.set_pv("throttle_type", ts_tstream_type.int, bid_blade, outlet.pid, 0)
    g.set_pv("ipout", ts_tstream_type.int, bid_blade, outlet.pid, 3)
    P3 = Po1 * stg.P3_Po1
    g.set_pv("pout", ts_tstream_type.float, bid_blade, outlet.pid, P3)

    # Mixing upstream
    mix_up = _make_patch(
        kind="mixing",
        bid=bid_vane,
        i=(ni[0] - 1, ni[0]),
        j=(0, nj[0]),
        k=(0, nk[0]),
        nxbid=bid_blade,
        nxpid=outlet.pid + 1,
        dirs=(6, 1, 2),
    )
    mix_up.pid = g.add_patch(bid_vane, mix_up)

    # Mixing downstream
    mix_dn = _make_patch(
        kind="mixing",
        bid=bid_blade,
        i=(0, 1),
        j=(0, nj[1]),
        k=(0, nk[1]),
        nxbid=bid_vane,
        nxpid=outlet.pid + 1,
        dirs=(6, 1, 2),
    )
    mix_dn.pid = g.add_patch(bid_blade, mix_dn)

    # Apply application/block variables
    set_variables(g)
    g.set_av("dampin", ts_tstream_type.float, dampin)
    g.set_av("ilos", ts_tstream_type.float, ilos)

    # Rotation
    rpm_rotor = Omega / 2.0 / np.pi * 60.0
    for bid, rpmi in zip(g.get_block_ids(), [0, rpm_rotor]):
        set_rotation(
            g,
            [
                bid,
            ],
            rpmi,
            spin_j=False,
        )

    # Numbers of blades
    for bid, nbi, nb_inti in zip(g.get_block_ids(), nb, nb_int):
        g.set_bv("fblade", ts_tstream_type.float, bid, nbi)
        g.set_bv("nblade", ts_tstream_type.int, bid, nb_inti)

    set_xllim(g, 0.03)

    # Initial guess
    if guess_file:
        tsr = ts_tstream_reader.TstreamReader()
        gg = tsr.read(guess_file)

        # Copy varaibles
        for var in ["ro", "rovx", "rovr", "rorvt", "roe", "trans_dyn_vis", "phi"]:
            for bid in g.get_block_ids():
                g.set_bp(var, ts_tstream_type.float, bid, gg.get_bp(var, bid))

        # With a good initial guess, do not need nchange
        g.set_av("nchange", ts_tstream_type.int, 0)

        # Throw out spurious negative wall distances
        for bid in g.get_block_ids():
            phi = g.get_bp('phi',bid)
            if np.any(phi[:]<0.):
                g.set_av("nchange", ts_tstream_type.int, 5000)
                phi[phi<0.] = 0.
                g.set_bp('phi', ts_tstream_type.float, bid, phi)

        # Use the existing Poisson field
        g.set_av("poisson_nstep", ts_tstream_type.int, 0)
        g.set_av("poisson_restart", ts_tstream_type.int, 1)

        # Make spanwise periodics slipwalls
        # periodic_to_slipwall(g)

    else:
        xg = np.concatenate(
            [
                x[0][
                    [
                        0,
                    ]
                    + ilte[0],
                    0,
                    0,
                ],
                x[1][
                    ilte[1]
                    + [
                        -1,
                    ],
                    0,
                    0,
                ],
            ]
        )
        Pog = np.repeat(stg.Po_Po1 * Po1, 2)
        Tog = np.repeat(stg.To_To1 * To1, 2)
        Mag = np.repeat(stg.Ma, 2)
        Alg = np.repeat(stg.Al, 2)

        for bid in g.get_block_ids():
            guess_block(g, bid, xg, Pog, Tog, Mag, Alg, stg.ga, rgas)

        # Make slipwalls spanwise periodic for Poisson
        # g.set_av("nstep", ts_tstream_type.int, 0)
        # slipwall_to_periodic(g)

    cp = rgas * stg.ga / (stg.ga - 1.0)
    g.set_av("ga", ts_tstream_type.float, stg.ga)
    g.set_av("cp", ts_tstream_type.float, cp)

    ts_tstream_load_balance.load_balance(g, 1, 1.0)

    # g.write_hdf5(fname)
    return g


class suppress_print:
    """A context manager that temporarily sets STDOUT to /dev/null."""

    def __enter__(self):
        self.orig_out = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.orig_out


def write_grid_from_params(params, fname=None):
    """Generate a Turbostream input file from a dictionary of parameters."""

    # Set geometry using dimensional bcond and 3D design parameter
    Dstg = design.scale_geometry(params.stg, **params.dimensional)

    # Generate mesh using geometry and the meshing parameters

    # Relax the inscribed radius limit ever so slightly if we are writing out a
    # file. This is ok because we check constraints without writing out first.
    pcopy = params.copy()
    if fname:
        pcopy.min_Rins *= 0.99

    # Hmesh
    # mesh = hmesh.stage_grid(Dstg, **pcopy.mesh)

    # OH mesh

    # Make a g and bcs
    x, ilte, nb = ohmesh.make_g_bcs(Dstg, **pcopy.mesh)

    with suppress_print():

        # Load them
        agr = ts_autogrid_reader.AutogridReader()
        g = agr.read('mesh.bcs','mesh.g')
        g = prep_ag(g, x, ilte, nb, params.stg, **params.cfd_input_file)
        if fname:
            g.write_hdf5(fname)


        # # Make the TS grid
        # g = make_grid(stg, *mesh, **params.cfd_input_file)

#         # Write out input file if specified
#         # (if not specified, this function acts as a constraint check)
#         if fname:
#             g.write_hdf5(fname)
