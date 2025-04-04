"""Write traverse plane or blade surface cuts to files."""

import os
import turbigen.util
import numpy as np

logger = turbigen.util.make_logger()


def post(grid, machine, meanline, _, postdir, spf, nstream=201):
    """write_annulus(spf)

    Write x--r curves at specified span fractions.


    Parameters
    ----------
    spf: list
        Span fractions of the meridional curves to write.

    """

    logger.info("Writing annulus curves...")
    mlim = (0.0, machine.ann.npts - 1)
    m = np.linspace(*mlim, nstream)
    xr = np.stack([machine.ann.evaluate_xr(m, spfi) for spfi in spf])
    fname = os.path.join(postdir, "annulus_xr")
    np.savez(fname, xr=xr)
