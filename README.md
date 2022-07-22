# turbigen

## Basic compressor calculation

1. Request an interactive node:
```
    sintr_0 -p ampere -t 1:00:00 -N 1 -A pullan-mhi-sl2-gpu --gres=gpu:1 --qos=intr
```

2. Create input files and do the calculation:
```
    run_turbostream_with_env.sh red_compressor/compressor.json
```
