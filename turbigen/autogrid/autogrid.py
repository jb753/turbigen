"""Given annulus and blade coordinates, make g/bcs in AutoGrid.

In general, we run AutoGrid on a remote machine, because it is not installed on
HPC. Meshing is done by feeding a Python 2 script into AutoGrid, which reads a
configuration JSON file with parameters."""

import os
import sys
import subprocess
import glob
import json
import shutil
import numpy as np
from tempfile import mkdtemp
from time import sleep
import turbigen.util

logger = turbigen.util.make_logger()

# Configuration for remote access by SSH
# For best reliability, use a ControlMaster in ssh config to reuse connection
REMOTE = "gp-111"  # Destination host where AutoGrid is running

SSH_ENV_VARS = ["SSH_AUTH_SOCK", "SSH_AGENT_PID"]
# Get the named vars for ssh agent setup from execution environment and format
# as a string to prepend commands with
try:
    SSH_ENV_STR = " ".join(["%s=%s" % (v, os.environ[v]) for v in SSH_ENV_VARS])
except KeyError:
    SSH_ENV_STR = ""

# The scripts we feed to autogrid are stored in same dir as this module
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SH_SCRIPT = "script_sh"  # Script that calls AutoGrid on the python files
SCRIPTS = [
    os.path.join(THIS_DIR, f) for f in ("script_ag.py2", "script_igg.py2", SH_SCRIPT)
]
CONF_NAME = "mesh_conf.json"


def _write_geomturbo(
    fname,
    ps,
    ss,
    h,
    c,
    zcst,
    nb,
    tips,
    rpm,
    ps_split,
    ss_split,
    le_blend=True,
    te_blend=False,
    cascade=False,
):
    """Write blade and annulus coordinates to AutoGrid GeomTurbo file.

    Parameters
    ----------

    fname : File name to write
    ps    : Nested list of arrays of pressure-side coordinates,
            ps[row][section][point on section, x/r/t]
            We allow different sizes for each section and row.
    ss    : Same for suction-side coordinates.
    h     : Array of hub line coordinates, h[axial location, x/r].
    c     : Same for casing line.
    zcst  : zcst[nrow-1][npts,2]
    nb    : Iterable of numbers of blades for each row."""

    # Determine numbers of points
    ni_h = np.shape(h)[0]
    ni_c = np.shape(c)[0]
    n_row = len(ps)

    fid = open(fname, "w")

    def writeln(s):
        fid.write(s + "\n")

    def writebld(gap_now, ps_now, ss_now, name):
        writeln("NI_BEGIN NIBlade")
        # writeln("NAME Main Blade")
        writeln(f"NAME {name}")

        n_sect = len(ps_now)
        ni_ps = [np.shape(psii)[0] for psii in ps_now]
        ni_ss = [np.shape(ssii)[0] for ssii in ss_now]

        if gap_now:
            if np.shape(gap_now) == ():
                gap_now = gap_now * np.ones(2)

            if gap_now[0]:
                writeln("NI_BEGIN NITipGap")
                writeln("WIDTH_AT_LEADING_EDGE %f" % gap_now[0])
                writeln("WIDTH_AT_TRAILING_EDGE %f" % gap_now[1])
                writeln("NI_END NITipGap")

        writeln("NI_BEGIN nibladegeometry")
        writeln("TYPE GEOMTURBO")
        writeln("GEOMETRY_MODIFIED 0")
        writeln("GEOMETRY TURBO VERSION 5")
        writeln("blade_expansion_factor_hub %f" % 0.1)
        writeln("blade_expansion_factor_shroud %f" % 0.1)
        writeln("intersection_npts %d" % 10)
        writeln("intersection_control %d" % 1)
        writeln("data_reduction %d" % 0)
        writeln("data_reduction_spacing_tolerance %f" % 1e-6)
        # Not sure what this hardcoded line does... sorry.
        writeln(
            "control_points_distribution "
            "0 9 77 9 50 0.00622408226922942 0.119480980447523"
        )
        writeln("units %d" % 1)
        writeln("number_of_blades %d" % 1)

        writeln("suction")
        writeln("SECTIONAL")
        writeln(str(n_sect))
        for k in range(n_sect):
            writeln("# section %d" % (k + 1))
            if cascade:
                writeln("XYZ")
            else:
                writeln("ZRTH")
            writeln(str(ni_ss[k]))
            for j in range(ni_ss[k]):
                writeln("%1.11f\t%1.11f\t%1.11f" % tuple(ss_now[k][j, :]))

        writeln("pressure")
        writeln("SECTIONAL")
        writeln(str(n_sect))
        for k in range(n_sect):
            writeln("# section %d" % (k + 1))
            if cascade:
                writeln("XYZ")
            else:
                writeln("ZRTH")
            writeln(str(ni_ps[k]))
            for j in range(ni_ps[k]):
                writeln("%1.11f\t%1.11f\t%1.11f" % tuple(ps_now[k][j, :]))
        writeln("NI_END nibladegeometry")

        # choose a leading and trailing edge treatment
        if le_blend:
            writeln("BLENT_AT_LEADING_EDGE")
        else:
            writeln("BLUNT_AT_LEADING_EDGE")

        if te_blend:
            writeln("BLENT_AT_TRAILING_EDGE")
        else:
            writeln("BLUNT_AT_TRAILING_EDGE")

        writeln("NI_END NIBlade")

    # Transform coordinates
    if cascade:
        raise NotImplementedError("Cascade needs work!")
        # # Swap the coordinates
        # for i in range(n_row):
        #     for k in range(n_sect[i]):
        #         ps[i][k] = ps[i][k][:, (1, 2, 0)]
        #         ss[i][k] = ss[i][k][:, (1, 2, 0)]

    # Write the header
    writeln("GEOMETRY TURBO")
    writeln("VERSION 5.5")
    writeln("bypass no")
    if cascade:
        writeln("cascade yes")
    else:
        writeln("cascade no")

    writeln("")

    # Write hub and casing lines (channel definition)
    writeln("NI_BEGIN CHANNEL")

    # Build the hub and casing line out of basic curves
    # Start the data definition
    writeln("NI_BEGIN basic_curve")
    writeln("NAME thehub")
    writeln("DISCRETISATION %d" % 10)
    writeln("DATA_REDUCTION %d" % 0)
    writeln("NI_BEGIN zrcurve")
    writeln("ZR")

    # Write the length of hub line
    writeln(str(ni_h))

    # Write all the points in x,r
    for i in range(ni_h):
        writeln("%1.11f\t%1.11f" % tuple(h[i, :]))

    writeln("NI_END zrcurve")
    writeln("NI_END basic_curve")

    # Now basic curve for shroud
    writeln("NI_BEGIN basic_curve")
    writeln("NAME theshroud")

    writeln("DISCRETISATION %d" % 10)
    writeln("DATA_REDUCTION %d" % 0)
    writeln("NI_BEGIN zrcurve")
    writeln("ZR")

    # Write the length of shroud
    writeln(str(ni_c))

    # Write all the points in x,r
    for i in range(ni_c):
        writeln("%1.11f\t%1.11f" % tuple(c[i, :]))

    writeln("NI_END zrcurve")
    writeln("NI_END basic_curve")

    # Now lay out the real shroud and hub using the basic curves
    writeln("NI_BEGIN channel_curve hub")
    writeln("NAME hub")
    writeln("VERTEX CURVE_P thehub 0")
    writeln("VERTEX CURVE_P thehub 1")
    writeln("NI_END channel_curve hub")

    writeln("NI_BEGIN channel_curve shroud")
    writeln("NAME shroud")
    writeln("VERTEX CURVE_P theshroud 0")
    writeln("VERTEX CURVE_P theshroud 1")
    writeln("NI_END channel_curve shroud")

    writeln("NI_END CHANNEL")
    # CHANNEL STUFF DONE

    if zcst:
        for irow in range(n_row - 1):
            _, n_pts = np.shape(zcst[irow])
            writeln("NI_BEGIN NIRSInterface")
            writeln("row_name r%d" % (i + 1))
            writeln("type outlet")
            writeln("NI_BEGIN geometry")
            writeln("NAME undefined")
            writeln("NI_BEGIN zrcurve")
            writeln("ZR polyline")
            writeln("%d" % n_pts)
            for i_pts in range(n_pts):
                writeln("%f %f" % tuple(zcst[irow][i]))
            writeln("NI_END zrcurve")
            writeln("NI_END geometry")
            writeln("NI_END NIRSInterface")

    # NOW DEFINE ROWS
    for i in range(n_row):
        if ps[i] is None:
            continue

        writeln("NI_BEGIN nirow")
        writeln("  NAME r%d" % (i + 1))
        writeln("  TYPE normal")
        writeln("  PERIODICITY %f" % nb[i])
        writeln("  ROTATION_SPEED %f" % rpm[i])

        # No non-axisymetric surfaces
        hdr = [
            "NI_BEGIN NINonAxiSurfaces hub",
            "NAME non axisymmetric hub",
            "REPETITION 0",
            "NI_END   NINonAxiSurfaces hub",
            "NI_BEGIN NINonAxiSurfaces shroud",
            "NAME non axisymmetric shroud",
            "REPETITION 0",
            "NI_END   NINonAxiSurfaces shroud",
            "NI_BEGIN NINonAxiSurfaces tip_gap",
            "NAME non axisymmetric tip gap",
            "REPETITION 0",
            "NI_END   NINonAxiSurfaces tip_gap",
        ]
        for ln in hdr:
            writeln(ln)

        writebld(tips[i], ps[i], ss[i], "Main Blade")

        if ps_split[i] is not None:
            writebld(tips[i], ps_split[i], ss_split[i], "splitter 1")

        writeln("NI_END nirow")

    writeln("NI_END GEOMTURBO")

    fid.close()


def _add_via(s, via):
    """Add a jump host to a command."""
    return "ssh -q %s '%s %s'" % (via, SSH_ENV_STR, s)


def _scp_to_remote(to_path, from_path, remote, via=None):
    """Copy a file from local machine to remote machine."""
    cmd_str = "scp -q %s %s:%s" % (
        os.path.abspath(from_path),
        remote,
        to_path,
    )
    if via:
        cmd_str = _add_via(cmd_str, via)
    if os.WEXITSTATUS(os.system(cmd_str)):
        raise Exception("Could not scp to remote, command %s" % cmd_str)


def _scp_from_remote(to_path, from_path, remote, via=None):
    """Copy a file back from remote machine to local machine."""
    cmd_str = "scp -q %s:%s %s" % (
        remote,
        from_path,
        os.path.abspath(to_path),
    )
    if via:
        cmd_str = _add_via(cmd_str, via)
    if os.WEXITSTATUS(os.system(cmd_str)):
        raise Exception("Could not scp from remote, command %s" % cmd_str)


def _execute_on_remote(cmd, remote, via, ntry=3):
    """Run a shell command on remote and return the output."""
    cmd_str = 'ssh -o BatchMode=yes -q %s "%s"' % (
        remote,
        cmd,
    )
    if via:
        cmd_str = _add_via(cmd_str, via)

    for itry in range(max(1, ntry)):
        try:
            out = subprocess.check_output(
                cmd_str, shell=True, stderr=subprocess.PIPE
            ).decode("ascii")
            success = True
            break
        except subprocess.CalledProcessError as e:
            success = False
            eout = e
            if ntry:
                delay = (itry + 1) * 5
                logger.info(f"Running remote command failed, retrying after {delay}s")
                sleep(delay)

    if not success:
        raise Exception(
            f"""Running remote command failed thrice, exit code {eout.returncode}
COMMAND: {cmd_str}
STDOUT: {eout.output.decode(sys.getfilesystemencoding()).strip()}
STDERR: {eout.stderr.decode(sys.getfilesystemencoding()).strip()}"""
        ) from None

    return out


def _run_remote(
    geomturbo,
    confjson,
    gbcs_output_dir,
    remote_workdir,
    remote,
    via=None,
    verbose=False,
):
    """Copy a geomturbo file to remote and run autogrid on it."""

    # Try to avoid races when multiple jobs running
    logger.debug("Sleeping to avoid races...")
    sleep(np.random.rand() * 2.0)

    # Make tmp dir on remote
    logger.debug("Making temp dir on remote ")
    tmpdir = _execute_on_remote(
        f"mktemp -p {remote_workdir} -d", remote, via
    ).splitlines()[0]
    logger.debug(tmpdir)

    logger.debug("Copying meshing config to remote... ")
    mesh_conf_remote = os.path.join(tmpdir, CONF_NAME)
    mesh_conf_local = os.path.abspath(confjson)
    _scp_to_remote(mesh_conf_remote, mesh_conf_local, remote, via)
    logger.debug("Deleting local temp file... ")

    # Copy files across
    logger.debug("Copying geometry file... ")
    _scp_to_remote(
        os.path.join(tmpdir, "mesh.geomTurbo"), os.path.abspath(geomturbo), remote, via
    )
    logger.debug("Copying scripts file... ")
    _scp_to_remote(tmpdir, " ".join(SCRIPTS), remote, via)

    sleep(0.5)

    # Run the shell script
    queuefile = os.path.join(remote_workdir, "queue.txt")
    logger.debug(f"Adding job to queue file {remote}:{queuefile} and waiting... ")
    try:
        _execute_on_remote(
            "cd %s ; bash %s %s" % (tmpdir, SH_SCRIPT, queuefile), remote, via
        )
    except subprocess.CalledProcessError as e:
        raise e

    # Copy mesh back
    sleep(0.5)
    logger.debug("Copying mesh back... ")
    for ext in ("g", "bcs"):
        remote_mesh_files = os.path.join(tmpdir, f"mesh.{ext}")
        _scp_from_remote(gbcs_output_dir, remote_mesh_files, remote, via)

    # Check the g and bcs arrived
    logger.debug("Checking g and bcs have arrived... ")
    sleep(5.0)
    success = True
    for suff in (".g", ".bcs"):
        fpath = os.path.join(os.path.abspath(gbcs_output_dir), "*" + suff)
        if not glob.glob(fpath):
            success = False

    if success:
        logger.debug("All done!")

    return success


def make_mesh(output_stem, section, annulus, zcst, nblade, tip, split, Omega, conf):
    """Make mesh in g/bcs format from coordinates using AutoGrid.

    Parameters
    ----------
    output_stem : str
        Where to place the output mesh files, `path/to/output.{g,bcs}`.
    section : list[nrow][2][nsect][3,npt]
        Nested list with coordinates for the blade sections with dimensions:
        number of rows, pressure or suction side, number of sections, and
        finally a 2D array with one row for x r rt coordinates of one section.
    annulus : list[2][npt,2]
        Nested list with coordinates of the annulus, dimensions:
        hub or casing, one row for each of x r coordinates.
    zcst : list[nrow-1][2,npt]
        Nested list with coordinates of the rotor-stator interfaces.
    nblade :  array[nrow]
        Integer numbers of blades in each row.
    tip : array[nrow]
        Tip gap in each blade row.
    Omega : array[nrow]
        Shaft angular velocity, rad/s.
    conf : dict
        Dictionary of configuration parameters for AutoGrid meshing script.

    Returns
    -------
    success: bool
        True if the meshing process completed successfully.

    """

    output_dir, output_basename = os.path.split(output_stem)

    # Format the geometry ready for writing to geomTurbo
    ps = [s[0] for s in section]
    ss = [s[1] for s in section]
    hub, cas = annulus

    if split:
        ps_split = []
        ss_split = []
        for irow in range(len(ps)):
            if split[irow] is not None:
                ps_split.append(split[irow][0])
                ss_split.append(split[irow][1])
            else:
                ps_split.append(None)
                ss_split.append(None)
    else:
        ps_split = [None for s in section]
        ss_split = [None for s in section]

    assert hub.ndim == 2
    assert cas.ndim == 2
    assert hub.shape[1] == 2
    assert cas.shape[1] == 2

    # Write the conf and geomturbo to a temporary directory
    base_tmp = output_dir
    if not os.path.isdir(base_tmp):
        os.mkdir(base_tmp)
    tmp_dir = mkdtemp(dir=base_tmp)

    if "verbose" in conf:
        verbose = conf.pop("verbose")
    else:
        verbose = False

    conf_path = os.path.join(tmp_dir, "mesh_conf.json")
    with open(conf_path, "w") as f:
        json.dump(conf, f)
    assert os.path.exists(conf_path)

    # AutoGrid expects only one zero-radius point on hub
    if (hub[:, 1] == 0.0).any():
        inose = np.where(np.diff(hub[:, 1]) > 0.0)[0][0]
        hub = np.concatenate((hub[(inose,), :], hub[(inose + 3) :, :]))

    rpm = Omega / 2.0 / np.pi * 60.0
    geomturbo_path = os.path.join(tmp_dir, "mesh.geomTurbo")
    _write_geomturbo(
        geomturbo_path, ps, ss, hub, cas, [], nblade, tip, rpm, ps_split, ss_split
    )

    via = conf.get("via", None)
    remote = conf["remote"]
    if not remote:
        raise Exception("No `remote_host` for AutoGrid meshing specified in config.")

    # Check we can connect to the jump host
    if via:
        try:
            logger.debug(f"Test connection to via host {via}...")
            _execute_on_remote("hostname", via, None, ntry=0)
        except Exception:
            raise Exception("Cannot connect to via host %s" % via) from None

        # Check the ssh-agent worker is running on via
        logger.debug("Checking for ssh-agent on via host...")
        pid = _execute_on_remote(
            r"""ps aux | grep $(whoami) |  tr -s ' ' | cut -d' ' -f2,11 | grep ssh-agent | cut -d' ' -f1""",
            via,
            None,
            ntry=0,
        ).strip()
        if not pid:
            raise Exception(
                f"""ssh-agent is not running on via_host {via}
    You need to start the ssh-agent, enter your password to unlock the keys,
    and load the details into the current terminal by running:
    ssh -t {via} 'eval $(ssh-agent) && ssh-add' && source get_ssh_agent.sh {via}"""
            ) from None
        if not pid == os.environ.get("SSH_AGENT_PID"):
            raise Exception(
                f"""ssh-agent is running on via_host {via}
    but we do not have the details in current shell. Fix by running:
    source get_ssh_agent.sh {via}
    If the problem persists, then it is possible that the already
    running ssh-agent process is stuck somehow. So kill it and restart
    by running:
    ssh -t {via} 'pkill ssh-agent && eval $(ssh-agent) && ssh-add' && source get_ssh_agent.sh {via}
    """
            ) from None

    # Check we can connect to the AG box
    try:
        logger.debug(f"Test connection to remote host {remote}...")
        _execute_on_remote("hostname", remote, via, ntry=0)
    except Exception:
        raise Exception(
            f"""Cannot connect to remote host {remote}
You may need to start the ssh-agent, enter your password to unlock the keys,
and load the details into the current terminal by running:
ssh -t {via} 'eval $(ssh-agent) && ssh-add' && source get_ssh_agent.sh {via}"""
        ) from None

    # Check the AG worker is running on remote
    try:
        logger.debug("Checking the AutoGrid server is running...")
        _execute_on_remote("pgrep ag_server.sh", remote, via, ntry=1).strip()
    except Exception:
        raise Exception("ag_server.sh is not running on %s" % remote) from None

    # Check the AG worker is running on remote
    if not conf["remote_workdir"]:
        raise Exception("No remote_workdir set in mesh config")

    # Execute the meshing process on remote machine
    success = _run_remote(
        geomturbo_path, conf_path, tmp_dir, conf["remote_workdir"], remote, via, verbose
    )
    if not success:
        return False

    # For some reason, even though the previous routine verifies that the g and
    # bcs files have arrived, the following move command can sometimes fail to
    # find them. So sleep a little while.
    sleep(1.0)

    # Copy into desired output dir
    for ext in (".g", ".bcs"):
        shutil.move(glob.glob(os.path.join(tmp_dir, "*" + ext))[0], output_stem + ext)

    # Delete input files and temp dir
    os.remove(geomturbo_path)
    os.remove(conf_path)
    os.rmdir(tmp_dir)

    return True
