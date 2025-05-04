from turbigen.meanline import MeanLineDesigner
import numpy as np


class Fan(MeanLineDesigner):
    """Mean-line designer for an axial fan rotor."""

    @staticmethod
    def forward(So1, DPo, mdot, phi, psi, htr, etatt):
        """Calculate mean-line from inlet and design variables.

        Parameters
        ----------
        So1: float
            Stagnation state at inlet.
        DPo: float
            Stagnation pressure rise across the fan [Pa].
        mdot: float
            Mass flow rate through the fan [kg/s].
        phi: float
            Flow coefficient.
        psi: float
            Loading coefficient.
        htr: float
            Hub-to-tip ratio.
        etatt: float
            Total-to-total isentropic efficiency.

        Returns
        -------
        rrms: (2,) array
            Mean radii of the fan at inlet and exit [m].
        A: (2,) array
            Annulus areas at inlet and exit [m^2].
        Omega: float
            Shaft angular velocity [rad/s].
        Vxrt: (3, 2) array
            Velocity vectors at inlet and exit [m/s].
        S: (2) array of thermodynamic states
            Static states at inlet and exit.

        """

        # Insert code to calculate rrms, A, Omega, Vxrt, states
        # ...
        raise NotImplementedError

        # Return assembled mean-line object
        return (
            rrms,  # Mean radii
            A,  # Annulus areas
            Omega,  # Shaft angular velocity
            Vxrt,  # Velocity vectors
            S,  # Thermodynamic states
        )

    @staticmethod
    def backward(mean_line):
        """Reverse a cascade mean-line to design variables.

        Parameters
        ----------
        mean_line: MeanLine
            A mean-line object specifying the flow in a cascade.

        Returns
        -------
        out : dict
            Dictionary of aerodynamic design variables.
            The fields have the same meanings as in :func:`forward`.

        """

        # The output should be a dictionary keyed by the args to forward
        return {
            # 'DPo': ...,
            # 'mdot': ...,
            # 'phi': ...,
            # 'psi': ...,
            # 'htr': ...,
            # 'etatt': ...,
        }
