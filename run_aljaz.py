# Aljaz's sweep
import submit

"""
ran:
[1.080, 1.169],
[1.115, 1.632],
[1.036, 2.109],
[0.973, 2.206],
[0.867, 2.287],
[0.733, 2.357],
[0.664, 2.026],
[0.512, 1.831],
[0.430, 1.432],
[0.392, 1.009], - failed (this is design 9)
[0.549, 1.013],
[0.639, 1.138],
[0.724, 1.253], design 12 is the most efficient one
[0.785, 1.501],
[0.840, 1.784],
[0.905, 1.961]
"""

designs = [
[1.080, 1.169],
[1.115, 1.632],
[1.036, 2.109],
[0.973, 2.206],
[0.867, 2.287],
[0.733, 2.357],
[0.664, 2.026],
[0.512, 1.831],
[0.430, 1.432],
[0.392, 1.009],
[0.549, 1.013],
[0.639, 1.138],
[0.724, 1.253],
[0.785, 1.501],
[0.840, 1.784],
[0.905, 1.961],
[0.392, 1.009],
]

# The structure to adjust for different designs.
params = submit.read_params('default_params.json')
base_run_dir = 'aljaz2'
for design in designs:
    for var, val in zip(('phi','psi'), design):
        params['mean-line'][var] = val
    submit.run(params, base_run_dir)
