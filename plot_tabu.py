"""Plot a tabu search."""

import os.path
from turbigen import tabu, submit


runs = ['702fec29', 'c3e2ea8f', 'b72fb607', '2eda0de5', 'd61b1b27', 'e278bfd6', 'e0863449', '166a589a']

for run in runs:
    base_dir = os.path.join("run",run)
    mem_file = os.path.join(base_dir, "mem_tabu.json")
    datum_file = os.path.join(base_dir, "datum_param.json")

    Param = submit.ParameterSet.from_json(datum_file)
    print(run, Param)

    ts = tabu.TabuSearch(None, None, 12, 6, None, j_obj=(0,))
    ts.verbose = False
    ts.load_memories(mem_file)
    print(ts.mem_med.get(0)[0])
    print(ts.mem_med.get(0)[1])

quit()
