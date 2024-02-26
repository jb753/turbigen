import numpy as np
import turbigen.autogrid
import os
import turbigen.util
from turbigen.base import BaseConfig

logger = turbigen.util.make_logger()


class OHMeshConfig(BaseConfig):
    """Contains all configuration options with default values."""

    _name = "OH mesh"

    spf_ref = 0.5
    """Set blade-to-blade mesh parameters based on geometry at this span fraction."""

    dspf_mid = 0.03
    """Spanwise grid spacing at midspan, as a fraction of span."""

    remote_host = ""
    """Remote host on which AutoGrid server is running."""

    span_interpolation = 2.0
    """Spanwise % spacing between optimisations of the mesh, interpolate between."""

    via_host = ""
    """Jump host for SSH connection to the AutoGrid server."""

    queue_path = "~/.ag_queue.txt"
    """File on remote host to append our queued meshing jobs."""

    workdir = ""

    gbcs_path = ""
    """Location of pre-made mesh, or 'reuse' to take from `workdir/mesh.{g,bcs}`."""

    ni_inlet = 65
    """Number of streamwise points in inlet."""

    ni_outlet = 65
    """Number of streamwise points in outlet."""

    refine_factor = 0
    """Divide each edge into 2**(refine_factor) sub-edges."""

    fix_h_blocks = True

    wake_control = False
    wake_deviation = 0.0

    skewness_control = 0
    orthogonality_control = 0.5
    nsmooth = 500
    nsmooth_multigrid = 100
    blade_streamwise_weight = 3.0

    untwist_outlet = False

    round_TE = False

    nj_tip = 25
    frac_inlet = 0.9
    frac_outlet = 0.15
    relax_outlet = 1

    reset_blade_streamwise = True

    def __setattr__(self, key, value):
        if key not in dir(self):
            raise TypeError(f"Invalid OH-mesh configuration variable '{key}'")
        elif not isinstance(value, type(getattr(self, key))):
            raise TypeError(
                f"Invalid type for OH-mesh configuration variable {key}={value},"
                f" should be {type(getattr(self,key))}"
            )
        else:
            super().__setattr__(key, value)

    def __init__(self, **kwargs):
        """Initialise a configuration, overriding defaults with keyword arguments."""

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_autogrid_dict(self, chi_ref, dhub, dcas, dsurf, splitter):
        nj = int(1.0 / self.dspf_mid * 2.0)
        stagger_topo = _get_stagger_topology(chi_ref)

        nrow = len(stagger_topo)

        if self.via_host == "":
            via = None
        else:
            via = self.via_host

        # Default configs
        return {
            "verbose": True,
            "is_cascade": False,
            "fix_h_blocks": self.fix_h_blocks,
            "nrow": nrow,
            "nx_up": self.ni_inlet,
            "nx_dn": self.ni_outlet,
            "nx_mix": 9,
            "dr_hub": dhub,
            "dr_cas": dcas,
            "nr": nj,
            "round_TE": self.round_TE,
            "drt_row": dsurf.tolist(),
            "stagger": stagger_topo,
            "const_cells_pc": 45.0,
            "nr_tip_gap": self.nj_tip,
            "relax_outlet": self.relax_outlet,
            "ER_boundary_layer": 1.1,
            "ER_blade_surf": 1.2,
            "n_te": 17,
            "frac_up": self.frac_inlet,
            "frac_dn": self.frac_outlet,
            "prefix": "mesh",
            "spf_ref": self.spf_ref,
            "remote": self.remote_host,
            "via": via,
            "queue_file": self.queue_path,
            "nsmooth": self.nsmooth,
            "nsmooth_multigrid": self.nsmooth_multigrid,
            "orthogonality_control": self.orthogonality_control,
            "skewness_control": self.skewness_control,
            "reset_blade_streamwise": self.reset_blade_streamwise,
            "wake_control": self.wake_control,
            "wake_deviation": self.wake_deviation,
            "untwist_outlet": self.untwist_outlet,
            "splitter": splitter,
            "span_interp": self.span_interpolation,
            "blade_streamwise_weight": self.blade_streamwise_weight,
        }


def _get_stagger_topology(Al):
    nAl = len(Al)
    assert np.mod(nAl, 2) == 0
    Nrow = nAl // 2

    # Threshold for 'normal' angles
    Al[np.abs(Al) < 45.0] = 0.0

    return [
        np.sign(
            Al[
                (irow * 2, irow * 2 + 1),
            ]
        )
        .astype(int)
        .tolist()
        for irow in range(Nrow)
    ]


def make_grid(machine, mesh_config, dhub, dcas, dsurf, unbladed, skip=False):
    logger.info("Generating OH-mesh...")

    dsurf = np.mean(dsurf, axis=0)

    # File paths
    workdir = mesh_config.workdir
    if workdir == "":
        raise Exception("workdir for OH meshing not set")
    output_stem = os.path.join(os.path.abspath(workdir), "mesh")

    if mesh_config.gbcs_path == "":
        chi_ref = np.concatenate(
            [bld.get_chi(mesh_config.spf_ref) for bld in machine.bld]
        )

        # We fill in the rotation speeds later
        Omega = np.zeros_like(dsurf)
        logger.info("Making a new mesh.")
        splitter = []
        if machine.split:
            for irow, splt in enumerate(machine.split):
                if splt:
                    splitter.append(irow)
        else:
            splitter = []
        ag_config = mesh_config.to_autogrid_dict(chi_ref, dhub, dcas, dsurf, splitter)
        success = turbigen.autogrid.autogrid.make_mesh(
            output_stem, *machine.get_coords(), Omega, ag_config
        )

        if not success:
            raise Exception("Meshing failed.")

    elif mesh_config.gbcs_path == "reuse":
        logger.info(f"Reusing existing {output_stem}." + r"{g,bcs}")

    else:
        output_stem = os.path.abspath(mesh_config.gbcs_path)
        logger.info(f"Loading {output_stem}." + r"{g,bcs}")

    bcs_path = output_stem + ".bcs"
    g_path = output_stem + ".g"

    g = turbigen.autogrid.reader.read(g_path, bcs_path)

    if rf := mesh_config.refine_factor:
        for b in g:
            b.refine(rf)

    return g
