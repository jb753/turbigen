"""All available post-processing routines are described below with the arguments they take."""

import turbigen.post
import turbigen.util
import os
import glob
import sys
from functools import partial as _partial

# List all the modules in the post directory
post_dir = turbigen.post.__path__[0]
module_paths = glob.glob(os.path.join(post_dir, "*.py"))
module_names = [os.path.split(p)[-1][:-3] for p in module_paths]

# Get the object representing this module
# So we can setattr on it
_self = sys.modules[__name__]

# Loop over all post modules available
for n in module_names:

    # Import the module
    post_func = turbigen.util.load_post(n).post

    # Use partial to hide non user facing args
    post_partial = _partial(
        post_func,
        grid=None,
        machine=None,
        meanline=None,
        postdir=None,
    )

    # Copy the docstring
    post_partial.__doc__ = post_func.__doc__

    # Set as a named attribute of this module
    setattr(_self, n, post_partial)
