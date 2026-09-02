"""Quantities measured from a solved field and kept in the result.

A :class:`Metric` maps a config and a result to a dict of named numbers ---
scalars, or nested lists of them. Unlike a :class:`~turbigen.post.Post` it
returns data rather than figures, and unlike an
:class:`~turbigen.iterate.Iterator` nothing acts on what it measures: a metric
is a passive observation of the flow, written to ``result: metrics:`` so that a
run archived today can be mined later.

Keys are the metric's own business, derived from its type and parameters the
way an iterator derives ``dchi_TE[0]`` --- there is no user-supplied label.
"""

import logging

import numpy as np

from turbigen.node import Node

logger = logging.getLogger("turbigen")


class Metric(Node):
    """Base for quantities measured from a solved field."""

    def evaluate(self, config, result):
        """Return ``{name: value}`` measured from `result`.

        Each `value` is a number or a nested list of numbers. Return an empty
        dict when the run gives nothing to measure --- no grid, or a diverged
        march --- exactly as a from-solution iterator's ``error`` does.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement evaluate(self, config, result)"
        )


def measure(config, result):
    """Return every configured metric's values, merged and made YAML-clean.

    Each metric is wrapped: it is an observation added after the CFD has already
    been paid for, so one that raises is logged and skipped rather than allowed
    to sink the run's output --- the same guard `solve` puts around mix-out and
    the design-comparison table.
    """
    merged = {}
    for m in config.metrics:
        try:
            values = m.evaluate(config, result)
        except Exception as err:
            logger.warning(f"Metric {m.type!r} could not be measured: {err}")
            continue

        for name, value in values.items():
            if name in merged:
                logger.warning(f"Two metrics both write {name!r}; keeping the last.")
            merged[name] = np.asarray(value, dtype=float).tolist()

    return merged
