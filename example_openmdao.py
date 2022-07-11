"""Trying OpenMDAO."""
import openmdao.api as om
import os, uuid
from time import sleep

def _make_workdir(base_dir):
    """Inside base_dir, make new work dir in four-digit integer format."""

    # Make a working directory with unique filename
    case_str = str(uuid.uuid4())[:8]
    workdir = os.path.join(base_dir, case_str)
    os.mkdir(workdir)

    # Return the working directory so that we can save input files there
    return workdir

class ParaboloidExternalCodeComp(om.ExternalCodeComp):

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.base_dir = 'om_test'
        self.input_file_name = 'paraboloid_input.dat'
        self.output_file_name = 'paraboloid_output.dat'

        # If you want to write your command as a list, the code below will also work.
        # self.options['command'] = [
        #     sys.executable, 'extcode_paraboloid.py', self.input_file, self.output_file
        # ]

    def setup_partials(self):
        # Use finite difference method 
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        # Make a new workdir
        workdir = _make_workdir(self.base_dir)
        input_file_path = os.path.join(workdir,self.input_file_name)
        output_file_path = os.path.join(workdir,self.output_file_name)

        self.options['external_input_files'] = [input_file_path]
        self.options['external_output_files'] = [output_file_path]
        self.options['command'] = ('python3 extcode_paraboloid.py {} {}').format(input_file_path, output_file_path)


        # generate the input file for the paraboloid external code
        with open(input_file_path, 'w') as input_file:
            input_file.write('%.16f\n%.16f\n' % (x, y))

        # the parent compute function actually runs the external code
        super().compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_xy
        with open(output_file_path, 'r') as output_file:
            f_xy = float(output_file.read())

        outputs['f_xy'] = f_xy

# build the model
prob = om.Problem()
model = prob.model

model.add_subsystem('p', ParaboloidExternalCodeComp())

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('p.x', lower=-50, upper=50)
prob.model.add_design_var('p.y', lower=-50, upper=50)
prob.model.add_objective('p.f_xy')

prob.setup()

# Set initial values.
prob.set_val('p.x', 3.0)
prob.set_val('p.y', -4.0)

# run the optimization
prob.run_driver();
