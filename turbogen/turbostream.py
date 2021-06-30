"""Functions for exporting a stage design to Turbostream."""

def make_patch(kind, bid, i, j, k, nxbid=0, nxpid=0, dirs=None):
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
        p.idir, p.jdir, p.kdir = (0,1,2)

    return p

def apply_inlet(g, bid, pid, Poin, Toin, nk, nj):

    yaw   = np.zeros(( nk,  nj), np.float32)
    pitch = np.zeros(( nk,  nj), np.float32)
    pstag = np.zeros(( nk,  nj), np.float32)
    tstag = np.zeros(( nk,  nj), np.float32)

    pstag += Poin
    tstag += Toin
    yaw += 0.0
    pitch += 0.0

    g.set_pp("yaw", ts_tstream_type.float, bid, pid, yaw)
    g.set_pp("pitch", ts_tstream_type.float, bid, pid, pitch)
    g.set_pp("pstag", ts_tstream_type.float, bid, pid, pstag)
    g.set_pp("tstag", ts_tstream_type.float, bid, pid, tstag)
    g.set_pv("rfin", ts_tstream_type.float, bid, pid, 0.5)
    g.set_pv("sfinlet", ts_tstream_type.float, bid, pid, 1.0)

def add_to_grid(g, xin, rin, rtin, ind, bid):
    """From mesh coordinates, add a block with patches to TS grid object"""
    ni, nj, nk = np.shape(rtin)
    rt = rtin+0.
    r = np.repeat(rin[:,:,None],nk,axis=2)
    x = np.tile(xin[:,None,None],(1,nj,nk))

    # Permute the coordinates into C-style ordering
    # Turbostream is very fussy about this
    xp = np.zeros((nk, nj, ni), np.float32)
    rp = np.zeros((nk, nj, ni), np.float32)
    rtp = np.zeros((nk, nj, ni), np.float32)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                xp[k,j,i] = x[i,j,k]
                rp[k,j,i] = r[i,j,k]
                rtp[k,j,i] = rt[i,j,k]

    # Generate new block
    b = ts_tstream_type.TstreamBlock()
    b.bid = bid
    b.ni, b.nj, b.nk = ni, nj, nk
    b.np = 0
    b.procid = 0
    b.threadid = 0

    # Add to grid with coordinates
    g.add_block(b)
    for vname, vval in zip(['x','r','rt'],[xp,rp,rtp]):
        g.set_bp(vname, ts_tstream_type.float, bid, vval)

    # Leading and trailing edges
    ile, ite = ind

    # Add periodic patches first

    # Upstream of LE
    periodic_up_1 = make_patch(
        kind='periodic', bid=bid, i=(0,ile+1), j=(0,nj), k=(0,1),
        dirs=(0,1,6), nxbid=bid, nxpid=1,
        )
    periodic_up_2 = make_patch(
        kind='periodic', bid=bid, i=(0,ile+1), j=(0,nj), k=(nk-1,nk),
        dirs=(0,1,6), nxbid=bid, nxpid=0,
        )
    periodic_up_1.pid = g.add_patch(bid, periodic_up_1)
    periodic_up_2.pid = g.add_patch(bid, periodic_up_2)

    # Downstream of TE
    periodic_dn_1 = make_patch(
        kind='periodic', bid=bid, i=(ite,ni), j=(0,nj), k=(0,1),
        dirs=(0,1,6), nxbid=bid, nxpid=3,
        )
    periodic_dn_2 = make_patch(
        kind='periodic', bid=bid, i=(ite,ni), j=(0,nj), k=(nk-1,nk),
        dirs=(0,1,6), nxbid=bid, nxpid=2,
        )
    periodic_dn_1.pid = g.add_patch(bid, periodic_dn_1)
    periodic_dn_2.pid = g.add_patch(bid, periodic_dn_2)

    # Slip wall
    slip_j0 = make_patch( kind='slipwall', bid=bid,
            i=(0,ni), j=(0,1), k=(0,nk) )
    slip_nj = make_patch( kind='slipwall', bid=bid,
            i=(0,ni), j=(nj-1,nj), k=(0,nk) )
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
    xb = g.get_bp('x', bid)
    rb = g.get_bp('r', bid)
    cp = rgas / (ga-1.)*ga
    cv = cp/ga

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
    eb = cv*Tb + (Vb**2.)/2

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
                robp[k,j,i] = rob[k,j,i]
                rovxbp[k,j,i] = rovxb[k,j,i]
                rovrbp[k,j,i] = rovrb[k,j,i]
                rorvtbp[k,j,i] = rorvtb[k,j,i]
                roebp[k,j,i] = roeb[k,j,i]

    # Apply to grid
    g.set_bp("ro", ts_tstream_type.float, bid, robp)
    g.set_bp("rovx", ts_tstream_type.float, bid, rovxbp)
    g.set_bp("rovr", ts_tstream_type.float, bid, rovrbp)
    g.set_bp("rorvt", ts_tstream_type.float, bid, rorvtbp)
    g.set_bp("roe", ts_tstream_type.float, bid, roebp)

def set_variables(g):
    for bid in g.get_block_ids():
        g.set_bv("fmgrid", ts_tstream_type.float, bid, 0.2)
        g.set_bv("poisson_fmgrid", ts_tstream_type.float, bid, 0.05)
        g.set_bv("xllim_free", ts_tstream_type.float, bid, 0.0)
        g.set_bv("free_turb", ts_tstream_type.float, bid, 0.0)

    g.set_av("restart", ts_tstream_type.int, 1)
    g.set_av("poisson_restart", ts_tstream_type.int, 0)
    g.set_av("poisson_nstep", ts_tstream_type.int, 5000)
    g.set_av("ilos", ts_tstream_type.int, 1)
    g.set_av("nlos", ts_tstream_type.int, 5)
    g.set_av("nstep", ts_tstream_type.int, 25000)
    # g.set_av("nstep", ts_tstream_type.int, 20)
    g.set_av("nchange", ts_tstream_type.int, 5000)
    g.set_av("dampin", ts_tstream_type.float, 25.0)
    g.set_av("sfin", ts_tstream_type.float, 0.5)
    g.set_av("facsecin", ts_tstream_type.float, 0.005)
    g.set_av("cfl", ts_tstream_type.float, 0.4)
    g.set_av("poisson_cfl", ts_tstream_type.float, 0.5)
    g.set_av("fac_stmix", ts_tstream_type.float, 0.0)
    g.set_av("rfmix", ts_tstream_type.float, 0.01)
    g.set_av("viscosity", ts_tstream_type.float, muref)
    g.set_av("viscosity_law", ts_tstream_type.int, 1)
    g.set_av("write_yplus", ts_tstream_type.int, 1)

def set_rotation(g,bids,rpm,spin_j):
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
        nb = g.get_bv('nblade',bid)
        rnow = g.get_bp('r',bid)
        rm = np.mean([rnow.max(),rnow.min()])
        pitch = 2. * np.pi * rm  / float(nb)
        g.set_bv("xllim", ts_tstream_type.float, bid, frac*pitch)
def stuff():
    # Below here is code for h-meshing

    jp = -1
    f,a = plt.subplots()
    a.plot(vane_x,vane_rt[:,jp,:],'k-')
    a.plot(blade_x,blade_rt[:,jp,:],'r-')
    a.axis('equal')

    # Make grid, add the blocks
    bid_vane = 0
    bid_blade = 1
    g = ts_tstream_grid.TstreamGrid()
    add_to_grid(g, vane_x, vane_r, vane_rt, vane_i, bid_vane)
    add_to_grid(g, blade_x, blade_r, blade_rt, blade_i, bid_blade)

    # Inlet
    ni, nj, nk = np.shape(vane_rt)
    inlet = make_patch(
        kind='inlet', bid=bid_vane, i=(0,1), j=(0,nj), k=(0,nk),
        )
    inlet.pid = g.add_patch(bid_vane, inlet)
    apply_inlet(g, bid_vane, inlet.pid, Poin, Toin, nk, nj)

    # Outlet
    ni, nj, nk = np.shape(blade_rt)
    outlet = make_patch( kind='outlet', bid=bid_blade,
            i=(ni-1,ni), j=(0,nj), k=(0,nk) )
    outlet.pid = g.add_patch(bid_blade, outlet)
    g.set_pv("throttle_type", ts_tstream_type.int, bid_blade, outlet.pid, 0)
    g.set_pv("ipout", ts_tstream_type.int, bid_blade, outlet.pid, 3)
    g.set_pv("pout", ts_tstream_type.float, bid_blade, outlet.pid, Pout)

    # Mixing upstream
    ni, nj, nk = np.shape(vane_rt)
    mix_up = make_patch(
        kind='mixing', bid=bid_vane, i=(ni-1,ni), j=(0,nj), k=(0,nk),
        nxbid=bid_blade, nxpid=outlet.pid+1, dirs=(6,1,2))
    mix_up.pid = g.add_patch(bid_vane, mix_up)

    # Mixing downstream
    ni, nj, nk = np.shape(blade_rt)
    mix_dn = make_patch(
        kind='mixing', bid=bid_blade, i=(0,1), j=(0,nj), k=(0,nk),
        nxbid=bid_vane, nxpid=outlet.pid+1, dirs=(6,1,2))
    mix_dn.pid = g.add_patch(bid_blade, mix_dn)

    # Apply application/block variables
    set_variables(g)

    set_rotation(g,[bid_vane],0.,spin_j=False)
    set_rotation(g,[bid_blade],rpm_rotor,spin_j=False)

    for bid, nbi in zip(g.get_block_ids(),nb):
        g.set_bv("fblade", ts_tstream_type.float, bid, nbi)
        g.set_bv("nblade", ts_tstream_type.float, bid, nbi)

    set_xllim(g, 0.03)

    # Initial guess
    xg = np.array(vane_xc[:-1] + blade_xc[1:])*c
    Pog = np.repeat(stage.Po_Poin * Poin,2)
    Tog = np.repeat(stage.To_Toin * Toin,2)
    Mag = np.repeat(stage.Ma,2)
    Alg = np.repeat(stage.Al,2)

    for bid in g.get_block_ids():
        guess_block(g,bid,xg,Pog,Tog,Mag,Alg,ga, rgas)

    g.set_av('ga', ts_tstream_type.float, ga)
    g.set_av('cp', ts_tstream_type.float, cp)

    # g.write_hdf5(fname)

def run_remote(geomturbo, py_scripts, sh_script, conf_ini):
    """Copy a geomturbo file to gp-111 and run autogrid using scripts."""

    # Make tmp dir on remote
    tmpdir = os.popen(
            'ssh gp-111 mktemp -p ~/tmp/ -d').read().splitlines()[0]
    print(tmpdir)

    # Copy files across 
    files = [geomturbo] + py_scripts + [sh_script] + [conf_ini]
    for si in files:
        os.system('scp %s gp-111:%s' % ( si, tmpdir))

    # Run the shell script
    result = os.popen("ssh gp-111 'cd %s ; ./%s'" % (tmpdir, sh_script.split('/')[-1])).read()
    print(result)

    # Copy mesh back
    os.system('scp gp-111:%s/*.{g,bcs} %s' % (tmpdir, os.getcwd()))
