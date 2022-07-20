"""Produce g and bcs files from geometry and mesh config files."""

import os, shutil
import numpy as np
from . import hmesh, geometry

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)


def trim_te(ps, is_stator):
    if is_stator:
        ite = np.argmax(ps[-1,:])
    else:
        ite = np.argmin(ps[-1,:])
    ite += 1

    return ps[:,:ite]

def fillet(ps,ss,r=None):
    """Add a fillet.

    ps : pressure side coordinates, ps[point on section,x/r/rt]"""

    ps_orig = np.copy(ps)
    ss_orig = np.copy(ss)

    # save meridional data
    xr_ps = ps[:,(0,1)]
    xr_ss = ss[:,(0,1)]

    # discard radial coord and transpose
    ps = ps[:,(0,2)].T
    ss = ss[:,(0,2)].T
    # now indexed [x/rt,section]

    # Ensure te goes in +ve theta dirn
    te = np.vstack((ps[:,-1],ss[:,-1])).T
    if np.diff(te[1,:]<0):
        te = np.flip(te,1)

    if r is None:
        r = 0.2 * np.linalg.norm(np.diff(te))

    # Get unit vectors for the ps, ss, and te directions
    vnn = np.hstack((np.diff(ps[:,-2:]), np.diff(ss[:,-2:]), np.diff(te)))
    v = vnn / np.linalg.norm(vnn, axis=0)

    # Bisect angles
    alpha = np.hstack( ( 0.5 * np.arccos(np.dot(v[:,0],-v[:,2])),
                         0.5 * np.arccos(np.dot(v[:,1],v[:,2]))))

    # Fillet length
    L = np.abs(r/np.tan(alpha))

    lps = np.sqrt(np.sum(np.diff(ps, axis=1)**2., axis=0))
    lss = np.sqrt(np.sum(np.diff(ss, axis=1)**2., axis=0))
    lte = np.sqrt(np.sum(np.diff(te, axis=1)**2., axis=0))
    if r>lte/2:
        raise ValueError('Fillet too big!')

    # If the fillet is too big for the mesh, throw away a point
    # and call again
    if L[0] > lps[-1]:
        # print('lps too long')
        return fillet(np.delete(ps_orig,-2,0),ss_orig,r)
    elif L[1] > lss[-1]:
        # print('lss too long')
        return fillet(ps_orig,np.delete(ss_orig,-2,0),r)


    # Perpendicular vectors in x-rt plane
    vp = np.copy(np.flip(v,0))
    vp[0,:] = -vp[0,:]

    # Lay out tangent points
    A = ps[:,-1]-L[0]*v[:,0];
    B = ps[:,-1]+L[0]*v[:,2];
    C = ss[:,-1]-L[1]*v[:,2];
    D = ss[:,-1]-L[1]*v[:,1];

    # Correct the normal vectors to point inwards
    if B[1]>C[1]:
        for i in range(3):
            vp[:,i] = -vp[:,i]

    # Centres of arcs
    E = A+vp[:,0]*r;
    F = D-vp[:,1]*r;

    # Generate fillet coordinates
    narc = 21
    ang = np.vstack([np.linspace(0.,np.pi - ai*2.,narc) for ai in alpha]).T
    ang = np.flip(ang,0)
    fil = np.stack((
        -r*vp[:,2]*np.cos(ang[:,[1]])-v[:,2]*r*np.sin(ang[:,[1]]) + E,
        -r*vp[:,2]*np.cos(ang[:,[0]])+v[:,2]*r*np.sin(ang[:,[0]]) + F),2)

    ## Find where to attach fillets
    ib_ps = (ps[0,:]>fil[0,0,0]).argmax()
    ib_ss = (ss[0,:]>fil[0,0,1]).argmax()
    ps_fil = np.concatenate((ps[:,:ib_ps],fil[:,:,0].T),1)
    ss_fil = np.concatenate((ss[:,:ib_ss],fil[:,:,1].T),1)

    ## Add radial coords back on
    ps_fil_r = np.interp(ps_fil[0,:],xr_ps[:,0],xr_ps[:,1])[None,:]
    ss_fil_r = np.interp(ss_fil[0,:],xr_ss[:,0],xr_ss[:,1])[None,:]

    ps_fil_xrrt = np.insert(ps_fil,1,ps_fil_r,axis=0).T
    ss_fil_xrrt = np.insert(ss_fil,1,ss_fil_r,axis=0).T

    return ps_fil_xrrt, ss_fil_xrrt


def write_geomturbo(fname, ps, ss, h, c, nb, tips, cascade=False):
    """Write blade and annulus geometry to AutoGrid GeomTurbo file.

    Parameters
    ----------

    fname : File name to write
    ps    : Nested list of arrays of pressure-side coordinates,
            ps[row][section][point on section, x/r/rt]
            We allow different sizes for each section and row.
    ss    : Same for suction-side coordinates.
    h     : Array of hub line coordinates, h[axial location, x/r].
    c     : Same for casing line.
    nb    : Iterable of numbers of blades for each row."""

    # Determine numbers of points
    ni_h = np.shape(h)[0]
    ni_c = np.shape(c)[0]

    n_row = len(ps)
    n_sect = [len(psi) for psi in ps]
    ni_ps = [[np.shape(psii)[0] for psii in psi] for psi in ps]
    ni_ss = [[np.shape(ssii)[0] for ssii in ssi] for ssi in ss]

    fid = open(fname,'w')

    # # Autogrid requires R,X,T coords
    # ps = ps[i][:,[1,0,2]]
    # ss = ss[i][:,[1,0,2]]

    if cascade:
        # Swap the coordinates
        for i in range(n_row):
            for k in range(n_sect[i]):
                ps[i][k] = ps[i][k][:,(1,2,0)]
                ss[i][k] = ss[i][k][:,(1,2,0)]
    else:
        # Convert RT to T
        for i in range(n_row):
            for k in range(n_sect[i]):
                ps[i][k][:,2] = ps[i][k][:,2] / ps[i][k][:,1]
                ss[i][k][:,2] = ss[i][k][:,2] / ss[i][k][:,1]

    # Write the header
    fid.write('%s\n' % 'GEOMETRY TURBO')
    fid.write('%s\n' % 'VERSION 5.5')
    fid.write('%s\n' % 'bypass no')
    if cascade:
        fid.write('%s\n\n' % 'cascade yes')
    else:
        fid.write('%s\n\n' % 'cascade no')

    # Write hub and casing lines (channel definition)
    fid.write('%s\n' % 'NI_BEGIN CHANNEL')

    # Build the hub and casing line out of basic curves
    # Start the data definition
    fid.write('%s\n' % 'NI_BEGIN basic_curve')
    fid.write('%s\n' % 'NAME thehub')
    fid.write('%s %i\n' % ('DISCRETISATION',10) )
    fid.write('%s %i\n' % ('DATA_REDUCTION',0) )
    fid.write('%s\n' % 'NI_BEGIN zrcurve')
    fid.write('%s\n' % 'ZR')

    # Write the length of hub line
    fid.write('%i\n' % ni_h)

    # Write all the points in x,r
    for i in range(ni_h):
        fid.write('%1.11f\t%1.11f\n' % tuple(h[i,:]))

    fid.write('%s\n' % 'NI_END zrcurve')
    fid.write('%s\n' % 'NI_END basic_curve')

    # Now basic curve for shroud
    fid.write('%s\n' % 'NI_BEGIN basic_curve')
    fid.write('%s\n' % 'NAME theshroud')

    fid.write('%s %i\n' % ('DISCRETISATION',10) )
    fid.write('%s %i\n' % ('DATA_REDUCTION',0) )
    fid.write('%s\n' % 'NI_BEGIN zrcurve')
    fid.write('%s\n' % 'ZR')

    # Write the length of shroud
    fid.write('%i\n' % ni_c)

    # Write all the points in x,r
    for i in range(ni_c):
        fid.write('%1.11f\t%1.11f\n' % tuple(c[i,:]))

    fid.write('%s\n' % 'NI_END zrcurve')
    fid.write('%s\n' % 'NI_END basic_curve')

    # Now lay out the real shroud and hub using the basic curves
    fid.write('%s\n' % 'NI_BEGIN channel_curve hub')
    fid.write('%s\n' % 'NAME hub')
    fid.write('%s\n' % 'VERTEX CURVE_P thehub 0')
    fid.write('%s\n' % 'VERTEX CURVE_P thehub 1')
    fid.write('%s\n' % 'NI_END channel_curve hub')

    fid.write('%s\n' % 'NI_BEGIN channel_curve shroud')
    fid.write('%s\n' % 'NAME shroud')
    fid.write('%s\n' % 'VERTEX CURVE_P theshroud 0')
    fid.write('%s\n' % 'VERTEX CURVE_P theshroud 1')
    fid.write('%s\n' % 'NI_END channel_curve shroud')

    fid.write('%s\n' % 'NI_END CHANNEL')

    # CHANNEL STUFF DONE
    # NOW DEFINE ROWS
    for i in range(n_row):
        fid.write('%s\n' % 'NI_BEGIN nirow')
        fid.write('%s%i\n' % ('NAME r',i+1))
        fid.write('%s\n' % 'TYPE normal')
        fid.write('%s %f\n' % ('PERIODICITY',nb[i]))
        fid.write('%s %i\n' % ('ROTATION_SPEED',0))

        hdr = ['NI_BEGIN NINonAxiSurfaces hub',
               'NAME non axisymmetric hub',
               'REPETITION 0',
               'NI_END   NINonAxiSurfaces hub',
               'NI_BEGIN NINonAxiSurfaces shroud',
               'NAME non axisymmetric shroud',
               'REPETITION 0',
               'NI_END   NINonAxiSurfaces shroud',
               'NI_BEGIN NINonAxiSurfaces tip_gap',
               'NAME non axisymmetric tip gap',
               'REPETITION 0',
               'NI_END   NINonAxiSurfaces tip_gap']

        fid.writelines('%s\n' % l for l in hdr)

        fid.write('%s\n' % 'NI_BEGIN NIBlade')
        fid.write('%s\n' % 'NAME Main Blade')

        if tips[i] is not None:
            fid.write('%s\n' % 'NI_BEGIN NITipGap')
            fid.write('%s %f\n' % ('WIDTH_AT_LEADING_EDGE',tips[i][0]))
            fid.write('%s %f\n' % ('WIDTH_AT_TRAILING_EDGE',tips[i][1]))
            fid.write('%s\n' % 'NI_END NITipGap')

        fid.write('%s\n' % 'NI_BEGIN nibladegeometry')
        fid.write('%s\n' % 'TYPE GEOMTURBO')
        fid.write('%s\n' % 'GEOMETRY_MODIFIED 0')
        fid.write('%s\n' % 'GEOMETRY TURBO VERSION 5')
        fid.write('%s %f\n' % ('blade_expansion_factor_hub',0.1))
        fid.write('%s %f\n' % ('blade_expansion_factor_shroud',0.1))
        fid.write('%s %i\n' % ('intersection_npts',10))
        fid.write('%s %i\n' % ('intersection_control',1))
        fid.write('%s %i\n' % ('data_reduction',0))
        fid.write('%s %f\n' % ('data_reduction_spacing_tolerance',1e-006))
        fid.write('%s\n' % ( 'control_points_distribution '
                  '0 9 77 9 50 0.00622408226922942 0.119480980447523'))
        fid.write('%s %i\n' % ('units',1))
        fid.write('%s %i\n' % ('number_of_blades',1))

        fid.write('%s\n' % 'suction')
        fid.write('%s\n' % 'SECTIONAL')
        fid.write('%i\n' % n_sect[i])
        for k in range(n_sect[i]):
            fid.write('%s %i\n' % ('# section',k+1))
            if cascade:
                fid.write('%s\n' % 'XYZ')
            else:
                fid.write('%s\n' % 'ZRTH')
            fid.write('%i\n' % ni_ss[i][k])
            for j in range(ni_ss[i][k]):
                fid.write('%1.11f\t%1.11f\t%1.11f\n' % tuple(ss[i][k][j,:]))

        fid.write('%s\n' % 'pressure')
        fid.write('%s\n' % 'SECTIONAL')
        fid.write('%i\n' % n_sect[i])
        for k in range(n_sect[i]):
            fid.write('%s %i\n' % ('# section',k+1))
            if cascade:
                fid.write('%s\n' % 'XYZ')
            else:
                fid.write('%s\n' % 'ZRTH')
            fid.write('%i\n' % ni_ps[i][k])
            for j in range(ni_ps[i][k]):
                fid.write('%1.11f\t%1.11f\t%1.11f\n' % tuple(ps[i][k][j,:]))
        fid.write('%s\n' % 'NI_END nibladegeometry')

        # choose a leading and trailing edge treatment

        #    fid.write('%s\n' % 'BLUNT_AT_LEADING_EDGE')
        fid.write('%s\n' % 'BLENT_AT_LEADING_EDGE')
        #    fid.write('%s\n' % 'BLENT_TREATMENT_AT_TRAILING_EDGE')
        fid.write('%s\n' % 'NI_END NIBlade')

        fid.write('%s\n' % 'NI_END nirow')

    fid.write('%s\n' % 'NI_END GEOMTURBO')

    fid.close()

def run_remote(geomturbo, py_scripts, sh_script, gbcs_output_dir):
    """Copy a geomturbo file to gp-111 and run autogrid using scripts."""

    # Make tmp dir on remote
    tmpdir = os.popen(
            'ssh gp-111 mktemp -p ~/tmp/ -d').read().splitlines()[0]

    # Copy files across 
    files = [geomturbo] + py_scripts + [sh_script]
    # for si in files:
    os.system('scp %s gp-111:%s' % ( " ".join(files), tmpdir))

    # Run the shell script
    os.popen("ssh gp-111 'cd %s ; ./%s'" % (tmpdir, sh_script)).read()

    # Copy mesh back
    os.system('scp gp-111:%s/*.{g,bcs} %s' % (tmpdir, gbcs_output_dir))

def make_g_bcs(
    Dstg, A, dx_c, tte, min_Rins=None, recamber=None, stag=None, resolution=1.
):
    """Generate an OH-mesh for a turbine stage."""

    # Change scaling factor on grid points

    # Distribute the spacings between stator and rotor
    dx_c = np.array([[dx_c[0], dx_c[1] / 2.0], [dx_c[1] / 2.0, dx_c[2]]])

    # Streamwise grids for stator and rotor
    x_c, ilte = hmesh.streamwise_grid(dx_c)
    x = [x_ci * Dstg.cx[0] for x_ci in x_c]

    # Generate radial grid
    Dr = np.array([Dstg.Dr[:2], Dstg.Dr[1:]])
    r = hmesh.merid_grid(x_c, Dstg.rm, Dr)

    # hub and casing lines
    rh = [ri[:,0] for ri in r]
    rc = [ri[:,-1] for ri in r]

    # Evaluate radial blade angles
    r1 = r[0][ilte[0][0], :]
    spf = (r1 - r1.min()) / r1.ptp()

    if Dstg.psi < 0.:
        chi = np.stack((Dstg.free_vortex_blade(spf,True), Dstg.free_vortex_vane(spf,True)))
    else:
        chi = np.stack((Dstg.free_vortex_vane(spf,), Dstg.free_vortex_blade(spf)))

    # If recambering, then tweak the metal angles
    if not recamber is None:
        dev = np.reshape(recamber, (2, 2, 1))
        dev[1] *= -1  # Reverse direction of rotor angles
        chi += dev

    # Get sections (normalised by axial chord for now)
    sect = [
        geometry.radially_interpolate_section(
            spf, chii, spf, tte, Ai, stag=stagi, loop=False
        )
        for chii, Ai, stagi in zip(chi, A, stag)
    ]

    # Adjust pitches to account for surface length
    So_cx = np.array([geometry._surface_length(si) for si in sect])

    if Dstg.psi >0.:
        s = np.array(Dstg.s)*So_cx
    else:
        s = np.array(Dstg.s)

    # Offset the rotor so it is downstream of stator
    x_offset = x[0][-1] - x[1][0]
    x[1] = x[1] + x_offset

    # Now assemble data for Autogrid
    ps = []
    ss = []
    for i, row_sect in enumerate(sect):
        ps.append([])
        ss.append([])
        for j, radial_sect in enumerate(row_sect):
            rnow = r[i][ilte[i][0]:(ilte[i][1]+1), j]
            x_cnow = x_c[i][ilte[i][0]:(ilte[i][1]+1)]
            xmax = np.max(radial_sect[:,0,:])
            for side, xrt in zip([ps,ss],radial_sect):
                r_interp = np.interp(xrt[0], x_cnow, rnow)
                xrt *= Dstg.cx[i] / xmax
                if i>0:
                    xrt[0]+=x_offset
                xrrt  = np.insert(xrt,1,r_interp,0)
                side[-1].append(xrrt.T)

    # Hub and casing lines in AG format
    x[0] = x[0][:-1]
    rc[0] = rc[0][:-1]
    rh[0] = rh[0][:-1]

    h = np.concatenate([np.column_stack((xi,rhi)) for xi, rhi in zip(x, rh)],0)
    c = np.concatenate([np.column_stack((xi,rci)) for xi, rci in zip(x, rc)],0)

    # Determine number of blades and angular pitch
    nb = np.round(2.0 * np.pi * Dstg.rm / s).astype(int)  # Nearest whole number

    tips = [None, None]
    write_geomturbo('mesh.geomTurbo', ps, ss, h, c, nb, tips)

    # Do autogrid mesh
    TURBIGEN_ROOT = '/rds/project/gp10006/rds-gp10006-pullan-mhi/jb753/turbigen'
    for f in ['script_ag.py2', 'script_igg.py2', 'script_sh']:
        shutil.copy(os.path.join(TURBIGEN_ROOT,'ag_mesh', f),'.')
    run_remote( 'mesh.geomTurbo', ['script_ag.py2', 'script_igg.py2'], 'script_sh', '.')

    return x, ilte, nb
