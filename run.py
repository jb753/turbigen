"""Generate and run a Turbostream job using a set of input parameters."""
import design, turbostream, hmesh
import json, glob, os, shutil

def write_hdf5_from_dict(params, input_file_name):
    """Generate a Turbostream input file from a dictionary of parameters."""

    # Mean-line design using the non-dimensionals
    stg = design.nondim_stage_from_Lam(**params['mean-line'])

    # Set geometry using dimensional bcond and 3D design parameter
    bcond_and_3d_params = dict(params['bcond'], **params['3d'])
    geometry = design.get_geometry(stg, **bcond_and_3d_params)

    # Generate mesh using geometry and the meshing parameters
    mesh = hmesh.stage_grid( stg, *geometry, **params['mesh'])

    # Make the TS grid and write out
    g = turbostream.make_grid( stg, *mesh, **params['bcond'] )
    g.write_hdf5(input_file_name)

def read_params(params_file_name):
    """Generate Turbostream input file from JSON file on disk."""

    with open(params_file_name,"r") as f:
        return json.load(f)

def _new_id( base_dir ):
    """Return an integer for the first vacant run identifier number."""

    # Get all direcories in base dir and convert to integers
    dnames = glob.glob(base_dir + "/*")
    runids = [-1,]
    for i in range(len(dnames)):
        try:
            runids.append(int(os.path.basename(dnames[i])))
        except ValueError:
            pass

    # Add one to the maximum identifier
    new_sim_id = max(runids) + 1

    return new_sim_id

def run(params, base_dir):
    """Submit a cluster job corresponding to the given parameter set."""

    # Make a working directory with unique filename
    new_id = _new_id( base_dir )
    workdir = os.path.join(base_dir,"%04d" % new_id)
    print("Creating %s" % workdir )
    os.mkdir(workdir)

    # copy in a slurm submit script
    slurm_template = "submit.sh"
    shutil.copy(os.path.join(slurm_template),workdir)
    new_slurm = os.path.join(workdir,slurm_template)

    # Append run id to the slurm job name
    os.system("sed -i 's/turbigen/turbigen_%04d/' %s" % (new_id, new_slurm))

    # Generate input file
    write_hdf5_from_dict(params, os.path.join(workdir,'input_1.hdf5'))

    # Save the parameters
    with open(os.path.join(workdir,'params.json'),"w") as f:
        json.dump(params, f)

    # Change to the working director and run
    os.chdir(workdir)
    jid = int(os.popen("sbatch %s" % slurm_template).read().split(" ")[-1])
    os.chdir('..')

    return jid

if __name__=="__main__":

    # Load default values for the parameters
    params = read_params('default_params.json')

    # Change the base run dir to separate different groups of runs
    base_run_dir = 'run'

    # Submit a job to the cluster
    run(params, base_run_dir)
