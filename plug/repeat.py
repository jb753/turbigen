import turbigen.iterators
import dataclasses
import numpy as np


# @dataclasses.dataclass
# class Repeat(turbigen.iterators.IteratorConfig):
#     """Settings for repeating stage."""
#
#     relaxation_factor: float = 0.5
#     """Factor controlling size of changes."""
#
#     To_frac: float = 0.5
#     """Fraction of varation in To to pass upstream."""
#
#     rtol: float = 0  # 0.001
#     """Relative tolerance for convergence of Po and To."""
#
#     atol: float = 0.01
#     """Absolute tolerance for convergence of angles."""
#
#     dAlpha_max: float = 20.0
#     """Clip the variations in yaw."""
#
#     dBeta_max: float = 10.0
#     """Clip the variations in pitch."""
#
#     dTo_max: float = 0.1
#     """Clip the variations in To."""
#
#     dPo_max: float = 0.1
#     """Clip the variations in Po."""
#
#     def check(self, config):
#         del config
#
#     def update(self, config) -> bool:
#         """Pass the outlet profiles upstream."""
#
#         log_data = {}
#
#         # Cut the outlet patch
#         C = config.grid.outlet_patches[0].get_cut()
#
#         # Mix out to uniformity to get reference state
#         Cm = C.mix_out()[0]
#
#         # Pitchwise mass-average the boundary condition quantities
#         # Mass flow rate per unit meridional area
#         mdot = C.pitchwise_integrate(C.rhoVm)
#         Po = C.pitchwise_integrate(C.rhoVm * C.Po) / mdot
#         To = C.pitchwise_integrate(C.rhoVm * C.To) / mdot
#         Alpha = C.pitchwise_integrate(C.rhoVm * C.Alpha) / mdot
#         Beta = C.pitchwise_integrate(C.rhoVm * C.Beta) / mdot
#
#         # Assemble into a matrix
#         spf = C.spf.mean(axis=2).squeeze()
#         profiles = np.stack([Po, To, Alpha, Beta], axis=1)[0]
#
#         # Subtract the meanline values
#         avg = np.array([Cm.Po, Cm.To, Cm.Alpha, Cm.Beta])
#         profiles -= avg[:, None]
#
#         # Normalise Po and To
#         profiles[0] /= Cm.Po
#         profiles[1] /= Cm.To
#
#         # Apply clipping to the normalised profiles
#         clip = [self.dPo_max, self.dTo_max, self.dAlpha_max, self.dBeta_max]
#         for i in range(4):
#             profiles[i] = np.clip(profiles[i], -clip[i], clip[i])
#
#         # Apply to the config object
#         inlet = config.inlet
#
#         # No previous inlet, initialise
#         if inlet.spf is None:
#             inlet.profiles = profiles
#             err = np.max(np.abs(profiles[1]))
#
#         # Compare with the previous inlet
#         else:
#             # Interpolate the previous profiles to the new span fraction
#             profiles_old = np.stack(
#                 [np.interp(spf, inlet.spf, inlet.profiles[i]) for i in range(4)],
#             )
#
#             # Calculate To errors
#             err = np.max(np.abs(profiles[1] - profiles_old[1]))
#
#             # Apply relaxation factor
#             rf = self.relaxation_factor
#             rf1 = 1.0 - rf
#             inlet.profiles = rf * profiles + rf1 * profiles_old
#
#         inlet.spf = spf
#
#         return err < self.rtol, {"Repeat_dTo": err}
