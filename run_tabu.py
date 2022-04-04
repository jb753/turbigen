"""Run a tabu search."""

from turbigen import submit

param = submit.ParameterSet.from_default()
submit.run_search(param, "base_t4")
