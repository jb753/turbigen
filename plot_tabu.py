"""Plot a tabu search."""

from turbigen import tabu, submit


base_dir = "run/e0863449/"
mem_file = base_dir + "/mem_tabu.json"
datum_file = base_dir + "/datum_param.json"

Param = submit.ParameterSet.from_json(datum_file)
print(Param)

ts = tabu.TabuSearch(None, None, 12, 6, None, j_obj=(0,))
ts.load_memories(mem_file)
ts.plot("tabu.pdf")

quit()
