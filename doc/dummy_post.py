"""Available post-processing routines are listed below with the arguments they take."""

import turbigen.post
import turbigen.util
import os
import glob
import sys

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
    _post_func = turbigen.util.load_post(n).post

    # Set as a named attribute of this module
    setattr(_self, n, _post_func)
