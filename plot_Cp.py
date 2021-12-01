import numpy as np
import matplotlib.pyplot as plt
import json

fname = 'smith_sweep_second/0000/dbslice.npz'
metaname = 'smith_sweep_second/0000/meta.json'

with open(metaname, "r") as f:
    meta = json.load(f)



fig, ax = plt.subplots(2,1)
ax[0].plot(meta['blockage_unst'])
ax[1].plot(meta['eff_lost_unst'])
plt.savefig("blockage.pdf")

Data = np.load(fname)
Cp = Data['Cp']
x_c = Data['x_c']
print(Cp.shape)
print(x_c.shape)

Cp_min = np.min(Cp,-1)
Cp_max = np.max(Cp,-1)
Cp_av = np.mean(Cp,-1)

fig, ax = plt.subplots()
ax.plot(x_c,Cp_min[0,...],'--k')
ax.plot(x_c,Cp_max[0,...],'--k')
ax.plot(x_c,Cp_av[0,...],'-k')
plt.savefig("Cp.pdf")
