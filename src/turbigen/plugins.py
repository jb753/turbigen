"""Finding and loading user-defined designs.

A design registers itself by being defined, because
:meth:`turbigen.node.Node.__init_subclass__` records it when the class body is
executed. So an ordinary ``import`` is all registration needs, and the machinery
here exists only for files that are not importable in the normal way --- a
scratch design sitting in a project directory rather than an installed package.

Registration matters only for *deserialisation*: the registry's single job is to
map a ``type:`` string in a config file to a class. Code that instantiates a
design directly, as a notebook or a test does, never consults it. Discovery is
therefore run from the config-loading boundary and nowhere else: not on import,
and not from the working directory.

Discovery walks up from the directory holding the config file, in the manner of
git looking for ``.git``, and loads the first :data:`PLUGIN_DIR_NAME` directory
it finds. Because the search is anchored on the config file rather than the
working directory, a case directory can be copied anywhere and still work, and
a config written into an output subdirectory still finds the same plugins as
the run that produced it.
"""

import importlib.util
import logging
import os
from pathlib import Path

logger = logging.getLogger("turbigen")

PLUGIN_DIR_NAME = "turbigen_plugins"
"""Directory name searched for by :func:`discover`."""

_LOADED = set()
"""Plugin files already imported, so that loading twice is a no-op."""


def find_plugin_dir(start):
    """Return the nearest plugin directory at or above `start`, or None.

    Walks up to the filesystem root, taking the first match. Directories owned
    by another user are skipped with a warning: the search reaches every
    ancestor of the config file, which on a shared filesystem may include
    directories the user does not control, and importing Python found there
    would be running someone else's code.

    Where the platform has no such notion of ownership --- Windows, where
    :func:`os.getuid` does not exist --- the directory is still used, because
    refusing every plugin on a single-user desktop would be a worse answer than
    the check it cannot make. But it says so, once, rather than looking as
    though it checked.
    """
    start = Path(start).resolve()
    uid = os.getuid() if hasattr(os, "getuid") else None

    for directory in (start, *start.parents):
        candidate = directory / PLUGIN_DIR_NAME
        if not candidate.is_dir():
            continue

        if uid is None:
            logger.warning(
                f"Loading {candidate} without checking who owns it: this "
                "platform has no user ids. Read it before running a config "
                "from a directory you do not control."
            )
        else:
            try:
                owner = candidate.stat().st_uid
            except OSError as err:
                logger.warning(f"Ignoring {candidate}, cannot be read: {err}")
                continue
            if owner != uid:
                logger.warning(f"Ignoring {candidate}, owned by another user")
                continue

        return candidate

    return None


def load_plugins(plug_dir):
    """Import every Python file in `plug_dir`, so its designs register.

    Files and subdirectories whose names begin with a dot or an underscore are
    skipped.
    """
    plug_dir = Path(plug_dir)
    if not plug_dir.is_dir():
        raise NotADirectoryError(f"Plugin directory {plug_dir} does not exist.")

    for py_file in sorted(plug_dir.rglob("*.py")):
        # Test the path relative to plug_dir, so that a dot or underscore in
        # some ancestor of the project does not hide every plugin below it.
        relative = py_file.relative_to(plug_dir)
        if any(part.startswith((".", "_")) for part in relative.parts):
            continue

        # Importing the same file twice would redefine its classes, and the
        # registry cannot tell a redefinition from two different designs
        # claiming one name. Skip it, as the import system does.
        resolved = py_file.resolve()
        if resolved in _LOADED:
            logger.debug(f"Already imported plugin {py_file}")
            continue
        _LOADED.add(resolved)

        spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as err:
            logger.error(f"Failed to import plugin {py_file}", exc_info=True)
            raise RuntimeError(f"Failed to import plugin {py_file}: {err}") from err
        logger.debug(f"Imported plugin {py_file}")


def discover(start):
    """Find and load the plugin directory above `start`.

    Returns the directory loaded, or None if there was not one. The outcome is
    logged either way, so that both "why did this design load" and "why did it
    not" are answerable from the log.
    """
    plug_dir = find_plugin_dir(start)

    if plug_dir is None:
        logger.debug(f"No {PLUGIN_DIR_NAME} directory found at or above {start}")
        return None

    logger.info(f"Loading plugins from {plug_dir}")
    load_plugins(plug_dir)
    return plug_dir
