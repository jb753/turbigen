# turbigen

## Single-row fan calculation

1. Request an interactive node:
```
    sintr_0 -p ampere -t 12:00:00 -N 1 -A pullan-sl3-gpu --gres=gpu:1
```

2. Activate Turbostream environment:
```
    source /usr/local/software/turbostream/ts3610_a100/bashrc_module_ts3610_a100
```

3. Edit the `make_row.py` script with desired design parameters and run, to
   generate an `input.hdf5` in a `fan_example/` directory.:
```
    python make_row.py 
```

4. Now change to the director and run Turbostream:
```
    cd fan_example
    mpirun -npernode 1 -np 1 turbostream input.hdf5 output 1 > log.txt &
```

5. Follow the progress of the calculation using:
```
   less +F log.txt 
```

6. Adapt your post-processing script to look at the results in
   `output_avg.hdf5`.
