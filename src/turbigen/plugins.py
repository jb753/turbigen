import logging
import traceback
import sys
import importlib.util
import inspect

logger = logging.getLogger("turbigen")

REGISTRY = {
    "mean_line_forward": {},
    "mean_line_backward": {},
}


def find_plugins(plugdir):
    """Find and load plugins from the plugdir."""

    logger.warning(f"Importing plugins from directory: {plugdir}")

    # Find all python files recursively in the plugdir
    py_files = list(plugdir.rglob("*.py"))

    for py_file in py_files:
        #
        # Exclude hidden files and directories
        if any(part.startswith(".") for part in py_file.parts):
            continue

        # Attempt to import the module
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.warning(f"Loaded plugin: {py_file}")
        except Exception:
            logger.error(f"Failed to import plugin: '{py_file}'")
            traceback.print_exc()
            sys.exit(1)

    # Need absolute import to ensure we are dealing with the same REGISTRY
    from turbigen.plugins import REGISTRY

    # Ensure that both forward and backward functions are registered
    # for any mean line type
    forward_types = set(REGISTRY["mean_line_forward"].keys())
    backward_types = set(REGISTRY["mean_line_backward"].keys())
    all_types = forward_types.union(backward_types)
    for mean_line_type in all_types:
        if mean_line_type not in forward_types:
            logger.error(f"Mean line type '{mean_line_type}' missing forward function.")
            sys.exit(1)
        if mean_line_type not in backward_types:
            logger.error(
                f"Mean line type '{mean_line_type}' missing backward function."
            )
            sys.exit(1)

    logger.warning("Successfully loaded plugins:")
    logger.warning("Mean line types:")
    for mean_line_type in all_types:
        sig = inspect.signature(REGISTRY["mean_line_forward"][mean_line_type])
        # Remvoe 'mean_line' from signature for clarity
        params = [p.name for p in list(sig.parameters.values())[1:]]
        logger.warning(f"  {mean_line_type}({', '.join(params)})")


def register_mean_line(func):
    """Add a mean line plugin to the registry."""

    # Check the name ends with _forward or _backward
    name = func.__name__
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Mean-line function name must end in '_forward' or '_backward', got {name}"
        )
    mean_line_type, direction = parts
    if direction not in ("forward", "backward"):
        raise ValueError(
            f"Mean-line function name must end in '_forward' or '_backward', got {name}"
        )

    if direction == "forward":
        # forward signature first arg is meanline
        sig_fwd = inspect.signature(func)
        params_fwd = list(sig_fwd.parameters.values())
        if len(params_fwd) < 1 or params_fwd[0].name != "mean_line":
            raise Exception(
                f"Mean line type '{mean_line_type}' forward function first argument must be 'mean_line', got '{params_fwd[0].name if params_fwd else 'none'}'."
            )
    else:
        # backward signature must take a single mean_line argument
        sig_bwd = inspect.signature(func)
        params_bwd = list(sig_bwd.parameters.values())
        if len(params_bwd) != 1 or params_bwd[0].name != "mean_line":
            raise Exception(
                f"Mean line type '{mean_line_type}' backward function must take a single argument 'mean_line', got '{params_bwd[0].name if params_bwd else 'none'}'."
            )

    # Add to module registry
    # Need absolute import to ensure we are dealing with the same REGISTRY
    from turbigen.plugins import REGISTRY

    REGISTRY[f"mean_line_{direction}"][mean_line_type] = func

    return func


if __name__ == "__main__":
    import pathlib

    import turbigen.plugins

    # Example usage
    plugdir = pathlib.Path("./plug")
    find_plugins(plugdir)
