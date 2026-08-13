"""Loading user-supplied designs.

A design registers itself by being defined, so loading a plugin is just
importing the file it lives in.
"""

import importlib.util
import logging

logger = logging.getLogger("turbigen")


def load_plugins(plug_dir):
    """Import every Python file under `plug_dir`, so its designs register."""
    plug_dir = __import__("pathlib").Path(plug_dir)
    if not plug_dir.is_dir():
        raise NotADirectoryError(f"Plugin directory {plug_dir} does not exist.")

    logger.info(f"Loading plugins from {plug_dir}")

    for py_file in sorted(plug_dir.rglob("*.py")):
        if any(part.startswith((".", "_")) for part in py_file.parts):
            continue
        spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as err:
            logger.error(f"Failed to import plugin {py_file}", exc_info=True)
            raise RuntimeError(f"Failed to import plugin {py_file}: {err}") from err
        logger.info(f"Imported plugin {py_file}")
