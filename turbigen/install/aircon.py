"""Install an air conditioner rotor into its box."""
import numpy as np
import turbigen.grid
import turbigen.util
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.spatial


def forward(
    g,
    machine,
    Lout,
    d,
    rint,
    eps_leak,
    L_leak,
    x_leak,
    tshroud,
    Lin,
    Khx=0.0,
    use_porous=True,
    skip_wdist=False,
):
    Dt = -np.radians(30)
    cosDt = np.cos(Dt)
    sinDt = np.sin(Dt)
    Rot = np.array(((cosDt, -sinDt), (sinDt, cosDt)))

    ob = g.cut_blade_surfs()[0].squeeze()

    # # Loop over blocks and find o-meshes
    # nj = g.inlet_patches[0].block.shape[1]
    # for b in g:
    #     if np.allclose(b[0,:,:].xrt,b[-1,:,:].xrt) and b.shape[1] == nj:
    #         ob = b[:,:,0]
    #         break

    plot_sects = False
    plot_offset = [0, 0.04, 0.08]
    if plot_sects:
        # Compare to datum CAD geometry
        xrrt_sects = [
            np.loadtxt(f"scripts/{prefix}_section_xrrt.csv", delimiter=",")
            for prefix in ("hub", "mid", "tip")
        ]

        offsets = np.zeros((3, 2, 1))
        offsets[0] = ((-0.001,), (0.0,))
        offset_r = np.zeros((3, 1))
        offset_r[0] = -0.001
        offset_r[1] = -0.001
        offset_r[2] = -0.001
        fig, ax = plt.subplots()

        prefixes = ("hub", "mid", "tip")
        for isect, xrrt_cad in enumerate(xrrt_sects):
            xcad = xrrt_cad[0].mean()
            ni, nj = ob.shape


            rrtnow = np.full((2, ob.shape[0]), np.nan)
            yz_now = np.full((2, ob.shape[0]), np.nan)

            for i in range(ni):
                rrtnow[0, i] = np.interp(xcad, np.flip(ob.x[i, :]), np.flip(ob.r[i, :]))
                rrtnow[1, i] = np.interp(
                    xcad, np.flip(ob.x[i, :]), np.flip(ob.rt[i, :])
                )
                yz_now[0, i] = np.interp(xcad, np.flip(ob.x[i, :]), np.flip(ob.y[i, :]))
                yz_now[1, i] = np.interp(xcad, np.flip(ob.x[i, :]), np.flip(ob.z[i, :]))

            tnow = rrtnow[1] / rrtnow[0]
            tcad = xrrt_cad[2] / xrrt_cad[1]
            ile_cad = np.argmin(xrrt_cad[1])
            ite_cad = np.argmax(xrrt_cad[1])
            ile_now = np.argmin(rrtnow[0])
            tcad -= tcad[ile_cad] - tnow[ile_now]  # + np.radians(-1)
            xrrt_cad[2] = tcad * xrrt_cad[1]

            r_cad = xrrt_cad[1]
            r_cad_norm = (r_cad - r_cad.min()) / r_cad.ptp()
            r_cad = rrtnow[0].min() + r_cad_norm * rrtnow[0].ptp() + offset_r[isect]

            yz_cad = np.stack((r_cad * np.sin(tcad), r_cad * np.cos(tcad)))

            tle = tcad[ile_cad]
            yz_radial = np.stack(
                (
                    r_cad[(ile_cad, ite_cad),] * np.sin(tle),
                    r_cad[(ile_cad, ite_cad),] * np.cos(tle),
                )
            )

            tte = tcad[ite_cad]
            yz_radial_te = np.stack(
                (
                    r_cad[(ile_cad, ite_cad),] * np.sin(tte),
                    r_cad[(ile_cad, ite_cad),] * np.cos(tte),
                )
            )

            # yz_cad += offsets[isect]

            # yz_now = _pol2cart(np.stack((rrtnow[0], tnow)))

            yz_now[0] *= -1
            yz_cad[0] *= -1

            yz_now = Rot @ yz_now
            yz_cad = Rot @ yz_cad

            yz_now[1] += plot_offset[isect]
            yz_cad[1] += plot_offset[isect]

            ax.plot(*yz_now, "b-")
            ax.plot(*yz_cad, "kx", ms=0.2)
            ax.text(*(np.median(yz_now,axis=-1)-np.array([0,0.01])), prefixes[isect].title())
            # ax.plot(*yz_radial, "k--")
            # ax.plot(*yz_radial_te, "k--")
            # ax.set_xlabel("y")
            # ax.set_ylabel("z")

            # fig, ax = plt.subplots()
            # ax.plot(*rrtnow,'b-')
            # ax.plot(*xrrt_cad[1:],'kx',ms=0.2)
            # ax.set_xlabel('r')
            # ax.set_ylabel('rt')
            # ax.axis('equal')
            # plt.show()

        ax.axis("equal")
        ax.axis("off")
        plt.savefig('beans.svg')
        plt.show()
        # print('Plotted aircon sections, quitting')
        # quit()

    # Set inlet pitch angle to zero
    g.inlet_patches[0].Beta = 0.

    gr = g

    with_hx = Khx

    def _meshblock(*args):
        return np.stack(np.meshgrid(*args, indexing="ij"), axis=0)

    Dmin = 0.0004
    Dmax = 0.006
    ER = 1.2

    # Boundary conditions
    Cout = gr.outlet_patches[0].get_cut().mix_out()[0]
    Po2 = Cout.Po
    To1 = Cout.To
    cp = Cout.cp
    ga = Cout.gamma
    mu = Cout.mu
    Vrguess = Cout.Vr
    Omega = gr[0].Omega.mean()
    # nj = xr_new.shape[2]
    njrotor = gr[0].shape[1]
    nkr = gr.inlet_patches[0].block.shape[2]
    old_outlet_patch = gr.outlet_patches[0]

    # for p in gr[0].rotating_patches:
    #     if (p.ijk_limits[0] == 0).all():
    #         gr[0].patches.remove(p)

    # Basic geometry
    b = 0.085  # Width betwwen back of HX and outer case
    c = Lout  # Length of outlet duct
    r3 = Cout.r  # Rotor exit radius
    dw = 0.03  # Gap after HX
    w1 = 0.6  # Side length of box

    # Blocking geometry
    w = w1 - dw
    rw = w / 2 * np.sqrt(2.0)
    rw1 = w1 / 2 * np.sqrt(2.0)
    dta = np.pi  # / 4.0
    Rcorner = 0.04

    xrhub, xrcas = machine.ann.get_coords()
    xrhub = xrhub.T
    xrcas = xrcas.T

    # # Extricate annulus lines
    # xrhub, xrcas = [np.concatenate(
    #     [gr[2].xr[:,:,jj,0],gr[5].xr[:,1:,jj,0],gr[0].xr[:,1:,jj,-1],]
    #     , axis=1) for jj in (0,-1)]

    # dxrhub = np.diff(xrhub, axis=1)
    # dxrcas = np.diff(xrcas, axis=1)

    # fig, ax = plt.subplots()
    # ax.plot(xrhub, "-x")
    # ax.plot(xrcas, "-+")

    # xrhub = gr[0][:, 0, 0].xr
    # xrcas = gr[0][:, -1, 0].xr

    # Leakage geometry
    rin = gr.inlet_patches[0].block.r[0, -1, 0]
    rhin = gr.inlet_patches[0].block.r[0, 0, 0].item()
    xin = gr.inlet_patches[0].block.x[0, -1, 0]
    x_bell = np.linspace(-d, x_leak, 50)
    r_bell_inner = rin - eps_leak - tshroud
    r_bell_outer = rin - eps_leak
    xr_bell_inner = np.stack((x_bell, r_bell_inner * np.ones_like(x_bell)))
    xr_bell_outer = np.stack((x_bell, r_bell_outer * np.ones_like(x_bell)))

    Next = 20
    xrextend = np.stack(
        (
            np.linspace(-d, x_leak, Next),
            np.full((Next,), rin),
        )
    )
    xrcas = np.concatenate((xrextend[:, :-1], xrcas), axis=1)

    # Inside leakage block
    drcas = np.diff(gr.inlet_patches[0].block.r[0, -2:, 0]).item()

    drinmax = np.max(np.diff(gr.inlet_patches[0].block.r[0, :, 0]))
    njLA = 13
    njLthick = 9
    dd = drcas
    rLA = turbigen.util.cluster_two_sided(
        r_bell_outer, rin, dd * 2.0, drcas * 8.0, ER, njLA
    )
    # rLA = np.linspace(r_bell_outer, rin, njLA)
    rLthick = np.linspace(r_bell_inner, r_bell_outer, njLthick)
    # turbigen.util.cluster_two_sided(
    # r_bell_inner, r_bell_outer, dd/2., dd * 2.0, ER, njLthick
    # )
    rLB = np.flip(
        turbigen.util.cluster_one_sided(
            r_bell_inner, rhin, dd * 6, drinmax, ER, njrotor - njLA - njLthick + 2
        )
    )
    njLB = len(rLB)
    rL = np.concatenate((rLB[:-1], rLthick[:-1], rLA))

    spfL = (rL - rhin) / (rin - rhin)
    spfL -= spfL[0]
    spfL /= spfL[-1]
    jj = np.linspace(0.0, njrotor - 1, njrotor)
    inose = np.argmax(xrhub[1] > xrhub[1, 0])
    frac = np.linspace(1.0, 0, inose)
    xrout = gr.inlet_patches[0].block.xr.copy()

    for i in range(inose):
        xrnow = gr.inlet_patches[0].block.xr[:, i, :, 0]
        spfnow = turbigen.util.cum_arc_length(xrnow)
        spfnow /= spfnow[-1]
        xr_new = scipy.interpolate.interp1d(spfnow, xrnow)(spfL)
        xrout[:, i, :, :] = np.expand_dims(
            frac[i] * xr_new + (1.0 - frac[i]) * xrnow, 2
        )

    gr.inlet_patches[0].block.xr = xrout

    # fig, ax = plt.subplots()
    # ax.plot(gr[0].x[:, :, 0], gr[0].r[:, :, 0], "k-", lw=0.2)
    # ax.plot(gr[0].x[:, :, 0].T, gr[0].r[:, :, 0].T, "k-", lw=0.2)
    # ax.plot(np.full_like(rL, xin), rL, "r-x", lw=0.5)

    dxinlet = np.diff(gr.inlet_patches[0].block.x[:, 0, 0])[0]
    niinlet = 41
    xinlet = np.flip(
        turbigen.util.cluster_one_sided(
            x_leak, x_leak - Lin, Dmin * 2.0, Dmax * 2.0, 1.2, niinlet
        )
    )
    # np.linspace(x_leak - Lin, x_leak, niinlet)
    inlet_block = gr.inlet_patches[0].block
    outlet_block = gr.outlet_patches[0].block
    xrtinlet = np.tile(inlet_block.xrt[:, (0,), :njLB, :], (1, niinlet, 1, 1))
    xrtinlet[0] = xinlet.reshape(-1, 1, 1)

    old_inlet_patch = gr.inlet_patches[0]
    old_inlet_patch.ijk_limits[1] = (0, -1)
    Po1 = old_inlet_patch.state.P
    inlet_block.patches.remove(old_inlet_patch)
    inlet_block.add_patch(turbigen.grid.PeriodicPatch(i=0, j=(0, njLB - 1)))
    inlet_block.add_patch(
        turbigen.grid.PeriodicPatch(i=0, j=(njLB + njLthick - 2, -1), label="to_leak")
    )
    pLB = [
        turbigen.grid.PeriodicPatch(i=-1),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
        turbigen.grid.InviscidPatch(j=0),
        old_inlet_patch,
    ]
    blocks = {}
    blocks["LB"] = turbigen.grid.PerfectBlock.from_coordinates(xrtinlet, gr[0].Nb, pLB)
    blocks["LB"].label = "LB"
    blocks["LB"].check_coordinates()
    blocks["LB"].Vx = 1.0
    blocks["LB"].Vr = 0.0
    blocks["LB"].Vt = 0.0
    blocks["LB"].Omega = Omega

    # ax.plot(blocks["LB"].x[:, :, 0], blocks["LB"].r[:, :, 0], "c-", lw=0.2)
    # ax.plot(blocks["LB"].x[:, :, 0].T, blocks["LB"].r[:, :, 0].T, "c-", lw=0.2)

    # ax.axis("equal")

    # xr_old = gr[0].xr[:,:,:,0]
    # spf_old = gr[0].spf[:,:,0]
    # ileak = np.argmax(xr_old[0,:,-1]>x_leak)
    # jinner = np.argmin(np.sum((xr_old[1,:ileak,:]-r_bell_inner)**2.,axis=0))
    # jouter = np.argmin(np.sum((xr_old[1,:ileak,:]-r_bell_outer)**2.,axis=0))
    # # Radial displacements
    # drinner = r_bell_inner-xr_old[1,:ileak,jinner]
    # drouter = r_bell_outer-xr_old[1,:ileak,jouter]
    # ileak2 = np.argmin(np.sum((xr_old[0,:,jinner:(jouter+1)]-x_leak)**2.,axis=1))
    # xr_new = xr_old.copy()
    # Lrelax = 2*tshroud
    # for i in range(ileak2+1):
    #     spf_relax = np.array([0.5, spf_old[i, jinner], spf_old[i,jouter], 1.])
    #     dr_relax = np.array([0.,drinner[i],drouter[i],0.])
    #     xr_new[1,i,:] = xr_old[1,i,:]+np.interp(spf_old[i,:],spf_relax, dr_relax)

    # for j in range(jinner-5, nj):
    #     x_relax = np.array([-Lrelax, 0., Lrelax])+x_leak
    #     dx_relax = np.array([0.,x_leak-xr_old[0,ileak2,j],0.])
    #     spf_relax = np.array([ spf_old[0, jinner-5], spf_old[0, jinner], spf_old[0,jouter], 1.])
    #     frac_relax = np.array([0.,1.,1.,0.])
    #     xr_new[0,:,j] = xr_old[0,:,j] +  np.interp(xr_old[0,:,j], x_relax, dx_relax)*np.interp(spf_old[0,j],spf_relax, frac_relax)

    # fig, ax = plt.subplots()
    # ax.plot(xr_new[0], xr_new[1], 'k-', lw=0.2)
    # ax.plot(xr_new[0].T, xr_new[1].T, 'k-', lw=0.2)
    # ax.plot(*xr_bell_inner,'b-')
    # ax.plot(*xr_bell_outer,'b-')
    # ax.plot(*xr_old[:,ileak2,:],'r:')
    # ax.plot(*xr_old[:,:,jinner],'m:')
    # ax.plot(*xr_old[:,:,jouter],'m:')
    # ax.axis('equal')
    # plt.show()
    # gr[0].xr = np.expand_dims(xr_new,-1)

    # xr_new[:,

    # ileak = np.argmin(np.abs(gr[0].x[:,-1,0]-(-x_leak)))
    # jleak = np.argmin(np.abs(gr[0].r[0,:,0]-(rin - eps_leak - tshroud)))
    # nrelax = 5
    # relax = np.linspace(0.,1.,nrelax).reshape(1,-1,1)
    # gr[0].x[ileak,(jleak-nrelax+1):(jleak+1),:] = -x_leak*relax + (1-relax)*gr[0].x[ileak,(jleak-nrelax+1):(jleak+1),:]

    # Numbers of points

    nie = 5  # Streamwise rotor exit O-block
    njout = 57  # Axial in exit blocks
    njS = 26

    nka = 801  # 201  # Circumferential butterfly blocks
    nia = 25  # Circumferential butterfly blocks
    nia1 = 31  # After HX Circumferential butterfly blocks
    nif = 49  # Radial in exit blocks
    niD = 25

    Nb = int(2.0 * np.pi / dta / 2)

    xmix = gr.outlet_patches[0].block.x[-1, :, 0].squeeze()
    xs2 = xmix[-1]
    Dxupper = d + xs2
    dxmix = np.abs(np.diff(xmix)[-1])
    xs2t = xs2 - tshroud

    tva_i0 = np.linspace(-dta, dta, nka)
    # Offset the casing curve to get outside of shroud
    # Get normals to each point
    dxrc = np.diff(xrcas, axis=1)
    dxrc_normal = np.stack((-dxrc[1], dxrc[0]))
    dxrc_normal /= turbigen.util.vecnorm(dxrc_normal)
    dxrc_normal_nodal = np.concatenate(
        (
            np.array((0.0, 1.0)).reshape(2, 1),
            0.5 * (dxrc_normal[:, 1:] + dxrc_normal[:, :-1]),
            np.array((-1.0, 0.0)).reshape(2, 1),
        ),
        axis=-1,
    )
    xrshroud_outer = xrcas + tshroud * dxrc_normal_nodal

    # fig, ax = plt.subplots()
    # ax.plot(xrcas[1])
    # ax.plot(xrshroud_outer[1])
    # plt.show()

    mshroud = turbigen.util.cum_arc_length(xrshroud_outer)
    Lshroud = mshroud[-1]
    mshroud /= Lshroud

    ##
    ## ABOVE SHROUD
    ##
    # xAS = xupper[-njAS:]
    # rAS0 = xrshroud_outer[1].min()
    # rAS1 = rAS0 + 0.3*(r3-rAS0)
    # rAS = turbigen.util.cluster_two_sided_free(rAS1, r3, Dmax/2., Dmax/2., Dmax, ER)
    # niAS = len(rAS)
    # CAS_i0 = np.stack((xAS, np.full((njAS,), rAS1)))
    # CAS_ni = np.stack((xAS, np.full((njAS,), r3)))
    # CAS_j0 = np.stack((np.full((niAS,), xAS[0]), rAS))
    # CAS_nj = np.stack((np.full((niAS,), xAS[-1]), rAS))
    # xrAS = np.expand_dims(turbigen.util.interpolate_transfinite([CAS_j0, CAS_i0, CAS_nj, CAS_ni]),-1)
    # xrtAS = np.append(
    #        np.tile(xrAS,(1,1,1,nka)),
    #        tAS,axis=0)
    # pAS = [
    #    # turbigen.grid.PeriodicPatch(j=-1),
    #    turbigen.grid.PeriodicPatch(k=0),
    #    turbigen.grid.PeriodicPatch(k=-1),
    # ]
    # blocks["AS"] = turbigen.grid.PerfectBlock.from_coordinates(xrtAS.copy(), Nb, pAS)
    # blocks["AS"].label = "AS"
    # blocks["AS"].check_coordinates()
    # blocks["AS"].Vx = 0.0
    # blocks["AS"].Vr = Vrguess
    # blocks["AS"].Vt = 0.0

    #
    # NEXT TO SHROUD
    #

    Linner = (x_leak - L_leak) + d
    L_clu_shroud1 = turbigen.util.cluster_two_sided_free(
        0.0, Linner, Dmin, Dmin, Dmax / 4.0, 1.2
    )
    L_clu_shroud2 = turbigen.util.cluster_two_sided_free(
        Linner, Lshroud, Dmin, Dmin, Dmax / 4.0, 1.1
    )
    clu_shroud = np.concatenate((L_clu_shroud1[:-1], L_clu_shroud2))
    clu_shroud /= clu_shroud[-1]
    Lcorner = tshroud * 3.0
    mcorner = 0.5
    xrS_corner = scipy.interpolate.interp1d(
        mshroud, xrshroud_outer + Lcorner * dxrc_normal_nodal, axis=1
    )(mcorner)
    icorner = np.argmax(clu_shroud > mcorner) + 1

    CS_j0 = scipy.interpolate.interp1d(mshroud, xrshroud_outer, axis=1)(
        clu_shroud[:icorner]
    )

    xCnj = np.array((xrshroud_outer[0, 0], xrS_corner[0]))
    rCnj = np.array((xrshroud_outer[1, 0] + Lcorner, xrS_corner[1]))
    xrCnj = np.stack((xCnj, rCnj))
    msj0 = turbigen.util.cum_arc_length(CS_j0)
    msj0 /= msj0[-1]
    CS_nj = scipy.interpolate.interp1d([0.0, 1.0], xrCnj, axis=1)(msj0)
    njupper = len(msj0)

    xCni = np.array((CS_j0[0, -1], xrS_corner[0]))
    rCni = np.array((CS_j0[1, -1], xrS_corner[1]))
    mCni = [0.0, 1.0]
    clu_Sj = (
        turbigen.util.cluster_one_sided(0.0, Lcorner, dxmix, Dmax, ER, njS) / Lcorner
    )
    clu_Sj /= clu_Sj[-1]
    xrCni = np.stack((xCni, rCni))
    CS_ni = scipy.interpolate.interp1d(mCni, xrCni, axis=-1)(clu_Sj)

    rCi0 = (
        turbigen.util.cluster_one_sided(0.0, Lcorner, dxmix, Dmax, ER, njS)
        + xrshroud_outer[1, 0]
    )
    CS_i0 = np.stack((np.full((njS,), xrshroud_outer[0, 0]), rCi0))

    # niS = len(clu_shroud)
    # CS_j0 = scipy.interpolate.interp1d(mshroud, xrshroud_outer, axis=1)(clu_shroud)
    # xCnj = np.array((xrshroud_outer[0,0], xrS_corner[0], xupper[njS-1]))
    # rCnj = np.array((xrshroud_outer[1,0]+Lcorner, xrS_corner[1], r3))
    # xrCnj = np.stack((xCnj,rCnj))
    # mCnj = turbigen.util.cum_arc_length(xrCnj)
    # mCnj /= mCnj[-1]
    # CS_nj = scipy.interpolate.interp1d(mCnj, xrCnj, axis=1)(clu_shroud)
    # rCi0 = turbigen.util.cluster_two_sided(0., Lcorner, Dmin, Dmax, ER, njS)+xrshroud_outer[1,0]
    # CS_i0 = np.stack((np.full((njS,),xrshroud_outer[0,0]), rCi0))
    # xCni = xupper[:njS]
    # CS_ni = np.stack((xCni,np.full((njS,),xrshroud_outer[1,-1])))

    iS3 = np.argmin(np.abs(CS_j0[0] - (x_leak - L_leak)))

    xrS = np.expand_dims(
        turbigen.util.interpolate_transfinite([CS_j0, CS_i0, CS_nj, CS_ni]),
        -1,
    )
    tS = np.tile(tva_i0.reshape(1, 1, 1, -1), (1,) + xrS.shape[1:])
    xrtS = np.append(np.tile(xrS, (1, 1, 1, nka)), tS, axis=0)
    pS = [
        turbigen.grid.PeriodicPatch(i=-1),
        turbigen.grid.PeriodicPatch(j=-1),
        turbigen.grid.PeriodicPatch(j=0, i=(0, iS3)),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
        turbigen.grid.RotatingPatch(j=0),
    ]
    pS[-1].Omega = Omega
    blocks["S"] = turbigen.grid.PerfectBlock.from_coordinates(xrtS.copy(), Nb, pS)
    blocks["S"].label = "S"
    blocks["S"].check_coordinates()
    blocks["S"].Vx = 0.0
    blocks["S"].Vr = Vrguess
    blocks["S"].Vt = 0.0
    blocks["S"].Omega = Omega

    #####
    #
    ##
    #
    ##

    CS1_i0 = CS_ni

    CS1_j0 = scipy.interpolate.interp1d(mshroud, xrshroud_outer, axis=1)(
        clu_shroud[icorner - 1 :]
    )

    xupper = turbigen.util.cluster_two_sided(
        xs2t, -d, Dmin, Dmax, ER, njS + njupper - 1
    )
    CS1_ni = np.stack((xupper[:njS], np.full((njS,), xrshroud_outer[1, -1])))

    xCnj = np.array((xrS_corner[0], xupper[njS - 1]))
    rCnj = np.array((xrS_corner[1], xrshroud_outer[1, -1]))
    xrCnj = np.stack((xCnj, rCnj))
    msj0 = turbigen.util.cum_arc_length(CS1_j0)
    msj0 /= msj0[-1]
    CS1_nj = scipy.interpolate.interp1d([0.0, 1.0], xrCnj, axis=1)(msj0)

    #
    xrS1 = np.expand_dims(
        turbigen.util.interpolate_transfinite([CS1_j0, CS1_i0, CS1_nj, CS1_ni]),
        -1,
    )
    tS1 = np.tile(tva_i0.reshape(1, 1, 1, -1), (1,) + xrS1.shape[1:])
    xrtS1 = np.append(np.tile(xrS1, (1, 1, 1, nka)), tS1, axis=0)
    pS1 = [
        turbigen.grid.PeriodicPatch(i=0),
        turbigen.grid.PeriodicPatch(i=-1),
        turbigen.grid.PeriodicPatch(j=-1),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
        turbigen.grid.RotatingPatch(j=0),
    ]
    pS1[-1].Omega = Omega
    blocks["S1"] = turbigen.grid.PerfectBlock.from_coordinates(xrtS1.copy(), Nb, pS1)
    blocks["S1"].label = "S1"
    blocks["S1"].check_coordinates()
    blocks["S1"].Vx = 0.0
    blocks["S1"].Vr = Vrguess
    blocks["S1"].Vt = 0.0
    blocks["S1"].Omega = Omega

    #
    #

    CS2_i0 = np.flip(CS_nj, axis=1)
    CS2_j0 = CS1_nj
    CS2_ni = np.stack((xupper[njS - 1 :], np.full((njupper,), xrshroud_outer[1, -1])))

    niS2 = CS2_j0.shape[1]
    # rCS2nj = turbigen.util.cluster_two_sided(CS_i0[1,0],xrshroud_outer[1,-1], Dmax/2, Dmax, ER, niS2)
    rCS2nj = np.linspace(CS_i0[1, -1], xrshroud_outer[1, -1], niS2)
    CS2_nj = np.stack((np.full((niS2,), xrshroud_outer[0, 0]), rCS2nj))

    #
    xrS2 = np.expand_dims(
        turbigen.util.interpolate_transfinite([CS2_j0, CS2_i0, CS2_nj, CS2_ni]),
        -1,
    )
    tS2 = np.tile(tva_i0.reshape(1, 1, 1, -1), (1,) + xrS2.shape[1:])
    xrtS2 = np.append(np.tile(xrS2, (1, 1, 1, nka)), tS2, axis=0)
    pS2 = [
        turbigen.grid.PeriodicPatch(i=0),
        turbigen.grid.PeriodicPatch(i=-1),
        turbigen.grid.PeriodicPatch(j=0),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
    ]
    blocks["S2"] = turbigen.grid.PerfectBlock.from_coordinates(xrtS2.copy(), Nb, pS2)
    blocks["S2"].label = "S2"
    blocks["S2"].check_coordinates()
    blocks["S2"].Vx = 0.0
    blocks["S2"].Vr = Vrguess
    blocks["S2"].Vt = 0.0
    blocks["S2"].Omega = Omega

    njS3 = 17
    rS3 = np.concatenate(
        (
            rLA[:-1],
            turbigen.util.cluster_two_sided(
                xrcas[1, 0], xrshroud_outer[1, 0], dd, dd * 5, ER, njS3
            ),
        )
    )
    njS3a = len(rS3)

    CS3_nj = CS_j0[:, : (iS3 + 1)]
    CS3_j0 = CS3_nj.copy()
    CS3_j0[1] = rS3[0]
    CS3_i0 = np.stack((np.full((njS3a,), xrcas[0, 0]), rS3))
    CS3_ni = np.stack((np.full((njS3a,), CS3_nj[0, -1]), rS3))
    xrS3 = np.expand_dims(
        turbigen.util.interpolate_transfinite(
            [CS3_j0, CS3_i0, CS3_nj, CS3_ni], plot=False
        ),
        -1,
    )
    # ax.axis('equal')
    # plt.show()
    tS3 = np.tile(tva_i0.reshape(1, 1, 1, -1), (1,) + xrS3.shape[1:])
    xrtS3 = np.append(np.tile(xrS3, (1, 1, 1, nka)), tS3, axis=0)
    pS3 = [
        turbigen.grid.RotatingPatch(i=-1),
        turbigen.grid.PeriodicPatch(j=-1),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
        turbigen.grid.MixingPatch(i=-1, j=(0, len(rLA) - 1)),
    ]
    pS3[0].Omega = Omega
    blocks["S3"] = turbigen.grid.PerfectBlock.from_coordinates(xrtS3.copy(), Nb, pS3)
    blocks["S3"].label = "S3"
    blocks["S3"].check_coordinates()
    blocks["S3"].Vx = 0.0
    blocks["S3"].Vr = Vrguess
    blocks["S3"].Vt = 0.0
    blocks["S3"].Omega = Omega

    njS4 = len(rLA)
    niS4 = 13
    CS4_i0 = CS3_ni[:, :njS4]
    CS4_ni = np.stack(
        (
            np.full_like(rLA, x_leak),
            rLA,
        )
    )
    xS3 = np.linspace(CS4_i0[0, 0], x_leak, niS4)
    CS4_j0 = np.stack((xS3, np.full_like(xS3, r_bell_outer)))
    CS4_nj = np.stack((xS3, np.full_like(xS3, xrcas[1, 0])))
    xrS4 = np.expand_dims(
        turbigen.util.interpolate_transfinite(
            [CS4_j0, CS4_i0, CS4_nj, CS4_ni], plot=False
        ),
        -1,
    )

    tS4 = np.tile(
        np.expand_dims(inlet_block.t[0, (njLB + njLthick - 2) :, :], (0, 1)),
        (1, xrS4.shape[1], 1, 1),
    )
    xrtS4 = np.append(np.tile(xrS4, (1, 1, 1, nkr)), tS4, axis=0)
    pS4 = [
        turbigen.grid.RotatingPatch(j=-1),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
        turbigen.grid.MixingPatch(i=0),
        turbigen.grid.PeriodicPatch(i=-1, label="from_leak"),
    ]
    pS4[0].Omega = Omega
    blocks["S4"] = turbigen.grid.PerfectBlock.from_coordinates(
        xrtS4.copy(), gr[0].Nb, pS4
    )
    blocks["S4"].label = "S4"
    blocks["S4"].check_coordinates()
    blocks["S4"].Vx = 0.0
    blocks["S4"].Vr = Vrguess
    blocks["S4"].Vt = 0.0
    blocks["S4"].Omega = Omega

    # (a,b,c,d) Butterfly blocks

    # Coordinate vectors

    xthick = turbigen.util.cluster_two_sided_free(
        xs2, xs2t, dxmix * 2, dxmix * 4, Dmax, ER
    )
    njthick = len(xthick)

    xva = np.concatenate((xmix[:-1], xthick[:-1], xupper))
    njhx = len(xva)  # Spanwise points across HX

    ### Post-interface
    rvD = np.linspace(r3, rint, niD)

    # Make r-t curves
    CD_i0 = np.stack((np.ones(nka) * r3, tva_i0))
    CD_ni = np.stack((np.ones(nka) * rint, tva_i0))
    CD_k0 = np.stack((rvD, np.ones(niD) * -dta))
    CD_nk = np.stack((rvD, np.ones(niD) * dta))

    # # Transfinite interp
    rtD = turbigen.util.interpolate_transfinite(
        [CD_k0, CD_i0, CD_nk, CD_ni], plot=False
    )
    assert rtD.shape == (2, niD, nka)

    # xD = np.ones((1, niD, njhx, nka)) * xva.reshape(1, 1, -1, 1)

    xDrelax = turbigen.util.cluster_two_sided(
        xva[0], xva[-1], Dmin * 5, Dmin * 5, ER, len(xva)
    )
    frelax = np.tile(np.linspace(0.0, 0.5, niD).reshape(1, -1, 1, 1), (1, 1, 1, nka))
    xD = frelax * xDrelax.reshape(1, 1, -1, 1) + (1 - frelax) * xva.reshape(1, 1, -1, 1)

    # fig, ax = plt.subplots()
    # ax.plot(xDrelax)
    # ax.plot(xva)
    # plt.show()
    # quit()

    rtD = np.tile(np.expand_dims(rtD, 2), (1, 1, njhx, 1))
    xrtD = np.concatenate((xD, rtD), axis=0)

    njst = njrotor + njthick - 2
    njen = njrotor + njthick + njS - 3

    pD = [
        turbigen.grid.RotatingPatch(i=0),
        turbigen.grid.PeriodicPatch(i=0, j=(njst, njen)),
        turbigen.grid.PeriodicPatch(i=0, j=(njen, -1)),
        turbigen.grid.MixingPatch(i=0, j=(0, njrotor - 1)),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
    ]
    pD[0].Omega = Omega

    P2 = old_outlet_patch.Pout

    if with_hx:
        pD.append(turbigen.grid.PeriodicPatch(i=-1))
    else:
        pp = turbigen.grid.OutletPatch(i=-1)
        pp.mdot_target = old_outlet_patch.mdot_target
        pp.Pout = P2
        pp.Kpid = old_outlet_patch.Kpid
        pD.append(pp)

    K = Khx

    blocks["D"] = turbigen.grid.PerfectBlock.from_coordinates(xrtD.copy(), Nb, pD)
    blocks["D"].label = "D"
    blocks["D"].check_coordinates()
    blocks["D"].Vx = 0.0
    blocks["D"].Vr = Vrguess
    blocks["D"].Vt = 0.0
    blocks["D"].Omega = 0.0

    # Porous patch location
    zva2 = np.flip(
        turbigen.util.cluster_one_sided(
            w / 2.0 - Rcorner * (1.0 - turbigen.util.sind(45)),
            0.0,
            Dmin * 2.0,
            Dmax,
            1.1,
            (nka - 1) // 2 + 1,
        )
    )
    yva2 = np.ones_like(zva2) * w / 2.0
    dy = np.zeros_like(yva2)
    wR = w / 2.0 - Rcorner
    yva2[zva2 > wR] -= Rcorner - np.sqrt(Rcorner**2 - (zva2[zva2 > wR] - wR) ** 2)

    zva = np.concatenate((np.flip(-zva2), zva2[1:]))
    yva = np.concatenate((np.flip(yva2), yva2[1:]))

    tva_ni = np.arctan(zva / yva)
    rva_ni = np.sqrt(yva**2.0 + zva**2.0)

    # rva_k = turbigen.util.cluster_two_sided(rint, rva_ni[-1], Dmax / 3, Dmax, ER, nia)
    rva_k = np.linspace(rint, rva_ni[-1], nia)

    # Make r-t curves
    C_i0 = np.stack((np.ones(nka) * rint, tva_i0))
    C_ni = np.stack((rva_ni, tva_ni))
    C_k0 = np.stack((rva_k, np.ones(nia) * -dta))
    C_nk = np.stack((rva_k, np.ones(nia) * dta))

    if with_hx:
        # # Transfinite interp
        rta = turbigen.util.interpolate_transfinite([C_k0, C_i0, C_nk, C_ni], plot=True)
        assert rta.shape == (2, nia, nka)

        xa = np.ones((1, nia, njhx, nka)) * xva.reshape(1, 1, -1, 1)
        rta = np.tile(np.expand_dims(rta, 2), (1, 1, njhx, 1))
        xrta = np.concatenate((xa, rta), axis=0)

        names = ["a"]  # , "b", "c", "d"]
        njst = njrotor + njthick - 2
        njen = njrotor + njthick + njS - 3
        for iname, name in enumerate(names):
            pa = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.PeriodicPatch(k=0),
                turbigen.grid.PeriodicPatch(k=-1),
            ]

            if with_hx:
                if use_porous:
                    pa.append(turbigen.grid.PorousPatch(i=-1))
                    pa[-1].porous_fac_loss = K
                else:
                    pa.append(turbigen.grid.PeriodicPatch(i=-1))
            else:
                pp = turbigen.grid.OutletPatch(i=-1)
                pp.mdot_target = old_outlet_patch.mdot_target
                pp.Kpid = old_outlet_patch.Kpid
                pa.append(pp)

            blocks[name] = turbigen.grid.PerfectBlock.from_coordinates(
                xrta.copy(), Nb, pa
            )
            blocks[name].t += iname * 2.0 * dta
            blocks[name].label = name
            blocks[name].check_coordinates()
            blocks[name].Vx = 0.0
            blocks[name].Vr = Vrguess
            blocks[name].Vt = 0.0
            blocks[name].Omega = 0.0

        # (a1) After Hx

        # Coordinate vectors
        xva1 = xva
        ddr = rw1 - rva_ni[-1]
        rva1_j = rva_ni[-1] + ddr * turbigen.util.cluster_two_sided_step(
            nia1, Dmax / 8.0 / ddr, Dmin / ddr * 2.0
        )

        tva1_i0 = tva_ni
        rva1_i0 = rva_ni
        # zva1 = np.linspace(w1 / 2.0, -w1 / 2.0, nka)
        zva1 = turbigen.util.cluster_two_sided(-w1 / 2.0, w1 / 2.0, Dmin, Dmax, ER, nka)
        tva1_ni = np.arctan(zva1 / (w1 / 2.0))
        rva1_ni = np.sqrt((w1 / 2.0) ** 2.0 + zva1**2.0)

        # Make r-t curves
        C1_i0 = np.stack((rva1_i0, tva1_i0))
        C1_ni = np.stack((rva1_ni, tva1_ni))
        C1_k0 = np.stack((rva1_j, np.ones(nia1) * -dta))
        C1_nk = np.stack((rva1_j, np.ones(nia1) * dta))

        # # Transfinite interp
        rta1 = turbigen.util.interpolate_transfinite([C1_k0, C1_i0, C1_nk, C1_ni])
        assert rta1.shape == (2, nia1, nka)

        xa1 = np.ones((1, nia1, njhx, nka)) * xva1.reshape(1, 1, -1, 1)
        rta1 = np.tile(np.expand_dims(rta1, 2), (1, 1, njhx, 1))
        xrta1 = np.concatenate((xa1, rta1), axis=0)

        names = ["a1"]  # , "b", "c", "d"]
        for iname, name in enumerate(names):
            pa1 = [
                turbigen.grid.PeriodicPatch(i=-1),
                turbigen.grid.PeriodicPatch(k=0),
                turbigen.grid.PeriodicPatch(k=-1),
            ]
            if use_porous:
                pa1.append(turbigen.grid.PorousPatch(i=0))
                pa1[-1].porous_fac_loss = K
            else:
                pa1.append(turbigen.grid.PeriodicPatch(i=0))
            blocks[name] = turbigen.grid.PerfectBlock.from_coordinates(
                xrta1.copy(), Nb, pa1
            )
            blocks[name].t += iname * 2.0 * dta
            blocks[name].label = name
            blocks[name].check_coordinates()
            blocks[name].Vx = 0.0
            blocks[name].Vr = Vrguess
            blocks[name].Vt = 0.0
            blocks[name].Omega = 0.0

        #
        #
        #

        xout = xva[-1] - turbigen.util.cluster_one_sided(0.0, c, Dmin, Dmax, ER, njout)
        xvf = np.concatenate((xva[:-1], xout))

        njf = len(xvf)

        assert np.all(np.diff(xvf) < 0.0)

        # Coordinate vectors
        zvf = turbigen.util.cluster_two_sided(-w1 / 2.0, w1 / 2.0, Dmin, Dmax, ER, nka)
        tvf_ni = np.arctan(zvf / (w1 / 2.0 + b))
        rvf_ni = np.sqrt((w1 / 2.0 + b) ** 2.0 + zvf**2.0)

        yvf = turbigen.util.cluster_two_sided(
            w1 / 2.0, w1 / 2.0 + b, Dmin, Dmax, ER, nif
        )
        tvf_k0 = np.arctan(-w1 / 2.0 / yvf)
        rvf_k0 = np.sqrt((w1 / 2.0) ** 2.0 + yvf**2.0)

        # Make r-t curves
        Cf_i0 = C1_ni
        Cf_ni = np.stack((rvf_ni, tvf_ni))
        Cf_k0 = np.stack((rvf_k0, -tvf_k0))
        Cf_nk = np.stack((rvf_k0, tvf_k0))

        # # # Transfinite interp
        rtf = turbigen.util.interpolate_transfinite([Cf_nk, Cf_i0, Cf_k0, Cf_ni])
        assert rtf.shape == (2, nif, nka)

        # Apply nozzle
        ARnoz = 0.5
        ycent = w1 / 2.0 + b / 2.0
        b2noz = b / 2.0 * np.sqrt(ARnoz)
        yvf_noz = np.linspace(ycent - b2noz, ycent + b2noz, nif)
        wnoz = w1 / 2.0 * np.sqrt(ARnoz)
        zvf_noz = np.linspace(-wnoz, wnoz, nka)
        ynoz, znoz = np.meshgrid(yvf_noz, zvf_noz, indexing="ij")
        tnoz = np.arctan(znoz / ynoz)
        rnoz = np.sqrt(znoz**2.0 + ynoz**2.0)

        xf = np.ones((1, nif, njf, nka)) * xvf.reshape(1, 1, -1, 1)
        rtf = np.tile(np.expand_dims(rtf, 2), (1, 1, njf, 1))
        xrtf = np.concatenate((xf, rtf), axis=0)

        for j in range(njhx, njf):
            xnow = xrtf[0, :, j, :].mean()
            xfrac = (xnow - xva[-1]) / (xvf[-1] - xva[-1])
            xrtf[1, :, j, :] = (1.0 - xfrac) * xrtf[1, :, j, :] + xfrac * rnoz
            xrtf[2, :, j, :] = (1.0 - xfrac) * xrtf[2, :, j, :] + xfrac * tnoz

        names = ["f"]  # , "g", "h", "i"]
        for iname, name in enumerate(names):
            pf = [
                turbigen.grid.PeriodicPatch(i=0, j=(0, njhx - 1)),
                turbigen.grid.OutletPatch(j=-1),
            ]
            pf[-1].mdot_target = old_outlet_patch.mdot_target
            pf[-1].Kpid = old_outlet_patch.Kpid
            blocks[name] = turbigen.grid.PerfectBlock.from_coordinates(
                xrtf.copy(), Nb, pf
            )
            blocks[name].t += iname * 2.0 * dta
            blocks[name].label = name
            blocks[name].check_coordinates()
            blocks[name].Vx = -Vrguess / 2.0
            blocks[name].Vr = 0.0
            blocks[name].Vt = 0.0
            blocks[name].Omega = 0.0

    g = turbigen.grid.Grid(list(blocks.values()))

    # Initial guess
    for b in g:
        b.set_P_T(Po1, To1)
        b.cp, b.gamma, b.mu = cp, ga, mu
        b.mu_turb = mu * np.ones(b.shape)
        # b.Omega = np.zeros_like(b.Omega)

    logger.info(f"With box cells/10^6={g.ncell/1e6}")
    logger.debug("Done")

    # print(len(g))
    gr.extend(g)
    outlet_block.patches.remove(old_outlet_patch)
    outlet_block.add_patch(turbigen.grid.MixingPatch(i=-1))

    # fig, ax = plt.subplots()
    # ax.axis("off")
    # for b in gr:
    #     if b.x.mean()>-0.1:
    #         _plot_block(b.xrt, ax, 0)
    # plt.show()
    # quit()

    # xr_hub_cad = np.loadtxt("scripts/xr_hub.dat")
    # xr_shd_cad = np.loadtxt("scripts/xr_shd.dat")
    # dx = xr_hub_cad[0, xr_hub_cad[1] < 0.22].max()
    # xr_hub_cad[0] -= dx
    # xr_shd_cad[0] -= dx
    # fig, ax = plt.subplots()
    # ax.plot(*xrhub, "b-")
    # ax.plot(*xrcas, "b-")
    # ax.plot(*xr_hub_cad, "kx")
    # ax.plot(*xr_shd_cad, "kx")
    # ax.plot(*xr_bell_inner, "g")
    # ax.plot(*xr_bell_outer, "g")
    # ax.plot(*xrshroud_outer, "m")
    # ax.axis("equal")
    # plt.show()
    # quit()

    # # Plot the annulus line
    # fig, ax = plt.subplots()
    # ax.plot(*xrhub, "b-")
    # ax.plot(*xrcas, "b-")
    # ax.plot(xrhub[0, (0, -1)], (0, 0), "r-.")
    # for b in gr:
    #     ax.plot(b.x[:, :, 0].mean(), b.r[:, :, 0].mean(), "m*")
    #     ax.text(b.x[:, :, 0].mean(), b.r[:, :, 0].mean(), b.label)
    #     ax.plot(b.x[:, :, 0], b.r[:, :, 0], "k-", lw=0.1)
    #     ax.plot(b.x[:, :, 0].T, b.r[:, :, 0].T, "k-", lw=0.1)
    #     # ax.plot(*xrshroud_outer, "m")
    #     # ax.plot(*xr_bell_inner, "g")
    #     # ax.plot(*xr_bell_outer, "g")
    # ax.axis("equal")
    # plt.show()

    # Find index to start nose at
    # ax.plot(*xrhub[:, (inose,)], "b*")

    # # Relax the meridional grid
    # jrelax = gr[0].shape[1] // 2
    # xr0 = gr[0][:inose, :jrelax, 0].xr
    # frac = np.linspace(0, 1.0, jrelax).reshape(1, -1)
    # runif = frac * xr0[1, :, -1, None] + (1.0 - frac) * xr0[1, :, 0, None]
    # relax = ((xr0[0, :, 0] - xr0[0, :, 0].min()) / (xr0[0, :, 0].ptp()))[:, None]
    # xr0[1, :, :] = relax * xr0[1, :, :] + (1.0 - relax) * runif
    # gr[0].r[:inose, :jrelax, :] = xr0[1, ..., None]

    # Set up coordinates for zero-radius block
    # Rk0 =

    # # xr0
    # # Plot the annulus line
    # fig, ax = plt.subplots()
    # xrplot = gr[0].xr[:, :, :, 0]
    # ax.plot(xrplot[0], xrplot[1], "k-", lw=0.1)
    # ax.plot(xrplot[0].T, xrplot[1].T, "k-", lw=0.1)
    # ax.axis("equal")
    # plt.show()

    inlet_block.add_patch(turbigen.grid.InviscidPatch(i=(0, inose), j=0))

    gr.match_patches()

    gr.apply_outlet(1e5)

    if not skip_wdist:
        logger.info("Setting wall distance in box")
        gr.calculate_wall_distance()

    return gr

    # return _make_aircon_box(
    #     g,
    #     machine,
    #     Lout,
    #     d,
    #     rint,
    #     eps_leak,
    #     L_leak,
    #     x_leak,
    #     tshroud,
    #     Lin,
    #     Khx,
    #     use_porous,
    # )


def inverse(g):
    nb = len(g)
    if nb == 8:
        gr = turbigen.grid.Grid(
            [
                g._blocks[0],
            ]
        )
    else:
        gr = turbigen.grid.Grid(
            g._blocks[:8],
        )

    old_outlet_patch = g.outlet_patches[0]
    old_mixing_patch = gr.mixing_patches[0]

    mixblk = gr.mixing_patches[0].block

    mixblk.patches.remove(old_mixing_patch)
    mixblk.patches.append(old_outlet_patch)

    return gr


logger = turbigen.util.make_logger()


def _plot_block(xrt, ax, jplot):
    x, y, z = _pol2cart(xrt[:, :, jplot, :])
    lw = 0.1
    ax.plot(y, z, "k-", lw=lw)
    ax.plot(y.T, z.T, "k-", lw=lw)
    ax.axis("equal")


def _pol2cart(xrt):
    if xrt.shape[0] == 3:
        x, r, t = xrt
        y = r * np.cos(t)
        z = r * np.sin(t)
        return np.stack((x, y, z))
    else:
        r, t = xrt
        y = r * np.cos(t)
        z = r * np.sin(t)
        return np.stack((y, z))
