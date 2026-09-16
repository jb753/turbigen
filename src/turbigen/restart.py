"""Carrying a flow field from one run into the next.

A run writes what it reached to ``restart.npz`` beside its config, and a later
run can start from it instead of the crude meridional guess.

Not in the YAML, though `ARCHITECTURE.md` once proposed it. Nothing reads a
flow field out of a config file: automatic iteration passes the grid in memory,
and a manual restart names a path anyway. Putting megabytes of base64 in the
document would only slow down every reader that wants the twenty numbers under
``result:``, and cost a 4/3 inflation for the privilege.

Primitives are stored, not the conserved variables. Conserved energy is
measured from its fluid's datum, and the datum moves between runs --- the
mesher derives it from each design --- so a conserved field written by one run
would be quietly misread by the next. Pressure, temperature and velocity cross
unchanged.

The critical indices of each block are stored with it: its ends and every patch
boundary, per dimension. A remesh moves a leading edge or a tip gap relative to
the ends of its block, and a field stretched end to end would put the flow that
was at the edge a node or two away from it. Holding those indices fixed is what
ember already does from grid to grid; the file only has to carry the source
side. Where they are missing or the topology does not match, the field is
stretched end to end as it always was, because a restart is a guess and a
slightly misplaced one is still worth having.

Alongside them sits a stamp: a digest of the design the field solves. It
records provenance and gates nothing here, because applying a field is asking
whether it is a useful place to start and the answer is usually yes even when
the design has moved. The strict question --- is this field the solution to
*this* config --- is asked by :mod:`turbigen.cli` before it will write an
answer down, and by nobody else.
"""

import hashlib
import json
import logging

import ember.block_util
import numpy as np

logger = logging.getLogger("turbigen")

RESTART_NAME = "restart.npz"
"""What a run calls the flow field it leaves behind, and `--restart` looks for."""

STATE = ember.block_util.STATE
"""What is stored per block: ember's own transfer variables, all dimensional.

Taken from ember rather than restated, so a field written here is exactly what
:meth:`ember.grid.Grid.interp_from_arrays` expects to be handed back.
"""

STAMP_KEY = "design_stamp"
"""Where the provenance hash sits in the archive, beside the block arrays."""

CRIT_KEYS = ("crit_i", "crit_j", "crit_k")
"""Suffixes of each block's critical index arrays, one per dimension."""

STAMPED = (
    "fluid",
    "mean_line",
    "annulus",
    "blades",
    "mesh",
    "operating_point",
    "inlet_profile",
)
"""Config sections that decide what the stored field *is*.

Everything the grid and the flow in it depend on: the machine, the mesh it was
put on, and the conditions it was marched under. `solver` is deliberately out
--- it decides how the answer was reached and not what it is, so raising CFL or
step count must not invalidate a field --- and so are the sections that drive
the surrounding search (`iterate`, `chic`, `batch`, `database`) or only look at
the answer afterwards (`post_process`, `job`).
"""


def design_stamp(config):
    """Return a digest of the part of `config` that determines a flow field.

    Two configs with the same stamp describe the same machine, mesh and
    operating point, so a field written under one *is* the solution to the
    other. That is a stricter question than the one :func:`apply` asks, and
    only :mod:`turbigen.cli`'s report verb asks it --- see the note there.

    Hash the *resolved* config, after `turbigen.iterate.resolve`, or the two
    sides will not be comparing the same thing: resolve moves design-only knobs
    onto their targets, so an unresolved config names the guess a run started
    from rather than the viscosity it used.
    """
    data = config.to_dict()
    subset = {key: data[key] for key in STAMPED if key in data}

    # Sorted and canonically separated so the digest depends on the values and
    # not on field order or on how the dict happened to be built. `default`
    # catches numpy scalars, which `Node.to_dict` passes straight through:
    # their repr is version-dependent, where `item()` is a plain Python number.
    text = json.dumps(subset, sort_keys=True, separators=(",", ":"), default=_plain)

    return hashlib.sha256(text.encode()).hexdigest()


def _plain(value):
    """Return `value` as something json can write, for the stamp only."""
    item = getattr(value, "item", None)
    if item is not None:
        return item()
    return repr(value)


def read_stamp(path):
    """Return the stamp stored at `path`, or None if it carries none.

    None means a field written before stamps existed, or by something that did
    not record one. Unknown provenance rather than bad provenance, and the
    caller is expected to treat it as the conservative case.
    """
    try:
        data = np.load(path)
    except Exception as err:
        logger.debug(f"Could not read a stamp from {path}: {err}")
        return None

    if STAMP_KEY not in data.files:
        return None

    return str(data[STAMP_KEY])


def critical_indices(block):
    """Return, per dimension, the indices of `block` a restart holds fixed.

    The two ends and every patch boundary, sorted and unique. The same rule as
    ``ember.block_util._patch_crit``, taken one block at a time so that half of
    it can be written to a file; the two must stay in step, or a restart from a
    file and one from a grid in memory would map the same pair differently.
    """
    return [
        np.unique(
            [0, block.shape[d] - 1]
            + [int(index) for patch in block.patches for index in patch.ijk_lim_abs[d]]
        ).astype(np.int32)
        for d in range(3)
    ]


def save(path, grid, config=None):
    """Write the flow field in `grid` to `path`.

    `config` is stamped in when given, so that a later reader can tell whether
    the field is the solution to a particular design or merely a good guess at
    one. Optional because a field is worth saving either way.
    """
    arrays = {}
    for i_block, block in enumerate(grid):
        for name in STATE:
            arrays[f"b{i_block}_{name}"] = np.asarray(
                getattr(block, name), dtype=np.float32
            )
        for key, indices in zip(CRIT_KEYS, critical_indices(block)):
            arrays[f"b{i_block}_{key}"] = indices

    if config is not None:
        arrays[STAMP_KEY] = np.array(design_stamp(config))

    np.savez_compressed(path, **arrays)
    logger.debug(f"Wrote a restart field for {len(grid)} block(s) to {path}")


def apply(grid, path):
    """Write the flow field stored at `path` into `grid`, in place.

    Blocks are matched by position and interpolated in index space where the
    resolution differs, so a mesh that changed with the design is fine as long
    as its topology did not. Patch boundaries are held where they started when
    the file records them and they correspond one for one with this grid's, so
    leading edge maps to leading edge; otherwise the block is stretched end to
    end, as a field written before they were stored always was.

    **The stamp is not checked here, deliberately.** Every chained restart in
    the system is a field from a *different* design: `iterate` starts each pass
    from the last one's field, `chic` walks the operating point along a
    characteristic, and `database.warm_start` begins at a neighbour's answer.
    Refusing a mismatch would break all three. Whether a field is the solution
    to a given config is a separate and much stricter question, asked by
    whoever needs the answer rather than by everyone who wants a head start.
    """
    data = np.load(path)

    # Blocks are counted from the state arrays alone. The stamp and the
    # critical indices are not blocks, and counting them would make a field
    # look like it came from a machine with extra blocks in it.
    n_block = len(
        {
            prefix
            for prefix, _, name in (key.partition("_") for key in data.files)
            if name in STATE
        }
    )
    if n_block != len(grid):
        raise ValueError(
            f"The restart field has {n_block} block(s) but this grid has "
            f"{len(grid)}; it was written for a different machine."
        )

    for i_block, block in enumerate(grid):
        arrays = [data[f"b{i_block}_{name}"] for name in STATE]
        crit, reason = _crit_pair(data, i_block, block, arrays[0].shape)
        if crit is None:
            logger.debug(
                f"Block {i_block} of the restart field is stretched end to end: "
                f"{reason}."
            )
        ember.block_util.interp_from_arrays(block, arrays, crit=crit)

    logger.info(f"Started from the restart field in {path}")


def _crit_pair(data, i_block, block, src_shape):
    """Return the ``(src, block)`` critical indices for one block, and why not.

    None, with the reason, whenever the stored indices cannot be trusted to
    correspond to this block's: absent, malformed, or a different count in any
    dimension. That falls back to the end-to-end mapping for the whole block
    rather than one dimension of it, so a restart is either patch-aware or
    exactly what it was before.
    """
    keys = [f"b{i_block}_{key}" for key in CRIT_KEYS]
    if not all(key in data.files for key in keys):
        return None, "the file records no critical indices"

    target = critical_indices(block)
    pairs = []
    for d, key in enumerate(keys):
        stored = np.asarray(data[key], dtype=np.int32)
        if (
            stored.ndim != 1
            or stored.size < 2
            or stored[0] != 0
            or stored[-1] != src_shape[d] - 1
            or np.any(np.diff(stored) <= 0)
        ):
            return None, f"its {'ijk'[d]} critical indices are malformed"
        if stored.size != target[d].size:
            return None, (
                f"it has {stored.size} critical {'ijk'[d]} indices against "
                f"{target[d].size} here"
            )
        pairs.append((stored, target[d]))

    return pairs, None
