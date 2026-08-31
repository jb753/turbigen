"""Vulture whitelist: names that are used dynamically and cannot be seen statically.

Passed to vulture as an extra source file by bin/check_dead_code.sh. Each name
referenced here is treated as "used". Keep this list short and justified.
"""

# bconds distortion profiles: these are Node config fields read by name through
# `self.column("DTo", spf)` driven by the COLUMNS classvar, never by attribute.
# (DPo escapes detection only because the string "DPo" also appears literally.)
DTo
DAlpha
DBeta

# solver.Ember config fields: deliberately conservative defaults that are handed
# to the ember solver by field introspection, not read in this repo.
cfl
n_stage
n_levels
fac_mgrid
sf_resid
