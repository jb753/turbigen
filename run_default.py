import submit, os

# Load default values for the parameters
params = submit.read_params('default_params.json')

# Change the base run dir to separate different groups of runs
base_run_dir = 'run'

# Submit a job to the cluster
submit.run(params, base_run_dir)