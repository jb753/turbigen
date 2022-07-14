"""Trying OpenMDAO."""
import openmdao.api as om
import os
import numpy as np
from scipy.optimize import newton, root_scalar

from turbigen import submit, design, geometry


class BaseTurbostreamComp(om.ExternalCodeComp):
    """Run TS from a parameter set.

    To make a subclass:
        * Define abstract methods params_from_inputs and outputs_from_metadata
        * Declare inputs and outputs in setup method, then call super().setup()

    """

    INPUT_FILE_NAME = "params.json"
    OUTPUT_FILE_NAME = "meta.json"

    def initialize(self):
        self.options.declare("datum_params")
        self.options.declare("base_dir")

    def params_from_inputs(self):
        raise NotImplementedError("Should define this method in subclasses.")

    def outputs_from_metadata(self):
        raise NotImplementedError("Should define this method in subclasses.")

    def compute(self, inputs, outputs):

        # Make a new workdir
        workdir = submit._make_rundir(self.options["base_dir"])
        input_file_path = os.path.join(workdir, self.INPUT_FILE_NAME)
        output_file_path = os.path.join(workdir, self.OUTPUT_FILE_NAME)

        # Set external command options
        self.options["external_input_files"] = [input_file_path]
        self.options["external_output_files"] = [output_file_path]
        self.options["command"] = [
            "run_turbostream_with_env.sh",
            input_file_path,
        ]

        # Use abstract method to generate the parameter set from general inputs
        param_now = self.params_from_inputs(inputs)

        # Save parameters file for TS
        param_now.write_json(input_file_path)

        # the parent compute function actually runs the external code
        super().compute(inputs, outputs)

        # parse the output file from the external code
        m = submit.load_results(output_file_path)

        # Insert outputs in-place using abstract method
        self.outputs_from_metadata(m, outputs)


class DeviationTurbostreamComp(BaseTurbostreamComp):
    """Run Turbostream with recambering to correct for deviation."""

    def initialize(self):
        self.options.declare("row_index")
        super().initialize()

    def setup(self):

        self.add_input("recamber", val=0.0)
        self.add_input("efficiency", val=self.options["datum_params"].eta)
        self.add_output("deviation")
        self.add_output("efficiency_out")

        stg = design.nondim_stage_from_Lam(
            **self.options["datum_params"].nondimensional
        )
        self.Al_target = np.array([stg.Al[1], stg.Alrel[2]])

        # self.declare_partials(of="*", wrt="*", method="fd", step=0.5)

        super().setup()

    def params_from_inputs(self, inputs):

        param_now = self.options["datum_params"].copy()
        param_now.eta = inputs["efficiency"][0]
        irow = self.options["row_index"]
        param_now.recamber[irow * 2 + 1] = inputs["recamber"][0]
        print(param_now.recamber)

        return param_now

    def outputs_from_metadata(self, metadata, outputs):

        Al_now = np.array([metadata["Al"][1], metadata["Alrel"][3]])
        Al_err = Al_now - self.Al_target
        outputs["efficiency_out"] = metadata["eta"]
        irow = self.options["row_index"]
        outputs["deviation"] = Al_err[irow]

class SectionTurbostreamComp(BaseTurbostreamComp):
    """Run Turbostream with parameterised blade section."""


    def initialize(self):
        self.options.declare("row_index")
        super().initialize()

    def setup(self):
        irow = self.options["row_index"]
        params = self.options["datum_params"]
        A_datum = params.A[irow]

        Rle, thick, beta = geometry.Rle_thick_beta_from_A(A_datum, params.tte)

        self.add_input("stagger", val=params.stag[irow])
        # self.add_input("recamber_le", val=0.)
        # self.add_input("recamber_te", val=params.recamber[irow*2+1])
        self.add_input("radius_le", val=Rle)
        self.add_input("beta", val=beta)
        self.add_input("thick_ps", val=thick[0,0])
        self.add_input("thick_ss", val=thick[1,0])

        self.add_output("lost_efficiency_rel")
        self.add_output("err_rel_phi")
        self.add_output("err_rel_psi")
        self.add_output("err_rel_Lam")

        # All partial derivatives approximated by finite difference
        # Most variables use a relative step
        self.declare_partials(of="*", wrt="*", method="fd", step=0.1, step_calc='rel_element')
        # For recamber which may go through zero, override with constant step
        # self.declare_partials(of="*", wrt="recamber_*", method="fd", step=0.2, step_calc='abs')

        super().setup()

    def params_from_inputs(self, inputs):

        print('In : %s' % str(inputs))

        param_now = self.options["datum_params"].copy()

        irow = self.options["row_index"]

        param_now.stag[irow] = inputs["stagger"][0]

        # param_now.recamber[irow*2] = inputs["recamber_le"][0]
        # param_now.recamber[irow*2+1] = inputs["recamber_te"][0]

        Rle = inputs["radius_le"][0]
        beta = inputs["beta"][0]
        thick = np.array([[inputs["thick_ps"][0]],[inputs["thick_ss"][0]]])
        param_now.A[irow] = geometry.A_from_Rle_thick_beta(Rle, thick, beta, param_now.tte)

        return param_now

    def outputs_from_metadata(self, metadata, outputs):

        params = self.options["datum_params"]

        outputs["lost_efficiency_rel"] = metadata["eta_lost"]/(1.-params.eta)
        for v in ["phi", "psi", "Lam"]:
            outputs["err_rel_" + v] = (metadata[v]/getattr(params, v)-1.)/params.rtol

        print('Out: %s' % str(outputs))

def correct_deviation(params):

    recamber = np.zeros((2,))
    row_indices = [0, 1, 0]

    for k, irow in enumerate(row_indices):

        params_now = params.copy()
        params_now.recamber[(1,3),] = recamber

        # build the model
        prob = om.Problem()
        model = prob.model
        model.add_subsystem(
            "ts",
            DeviationTurbostreamComp(
                row_index=irow, datum_params=params_now, base_dir="om_test4"
            ),
        )

        prob.setup()

        dev_all = []

        def iterate(x):
            prob.set_val("ts.recamber", x)
            prob.run_model()
            prob.set_val("ts.efficiency", prob.get_val("ts.efficiency_out")[0])
            dev = prob.get_val("ts.deviation")[0]
            dev_all.append(dev)
            return dev

        # save the data
        tol = 0.2
        brak = 3. * np.array([-1., 1.])
        if k==2:
            if np.abs(dev_all[-1])>tol:
                recamber[irow] = root_scalar(
                    iterate, bracket=recamber[irow]+brak/2., xtol=tol
                ).root
            else:
                break
        else:
            try:
                recamber[irow] = root_scalar(
                    iterate, bracket=brak, xtol=tol
                ).root
            except:
                recamber[irow] = root_scalar(
                    iterate, bracket=brak*2., xtol=tol
                ).root

    effy = prob.get_val("ts.efficiency_out")[0]

    params.recamber[1], params.recamber[3] = recamber
    params.eta = effy


# params = submit.ParameterSet.from_default()
# params.Co = (0.7, 0.7)
# correct_deviation(params)
# params.write_json('datum_deviation_corrected.json')

params = submit.ParameterSet.from_json('datum_deviation_corrected.json')
params.guess_file = os.path.join(os.getcwd(),'datum_guess.hdf5')
params.rtol = 0.01

# Set up model
prob = om.Problem()
model = prob.model
model.add_subsystem(
    "ts",
    SectionTurbostreamComp(
        row_index=0, datum_params=params, base_dir="om_sect"
    ),
)

# Design variables
prob.model.add_design_var("ts.stagger", lower=0., upper=90.)  # For vane, stag +ve
# prob.model.add_design_var("ts.recamber_le", lower=-5., upper=10.)
# prob.model.add_design_var("ts.recamber_te", lower=-5., upper=5.)
prob.model.add_design_var("ts.radius_le", lower=0.05, upper=0.5)
prob.model.add_design_var("ts.beta", lower=12., upper=36.)
prob.model.add_design_var("ts.thick_ps", lower=0.05, upper=0.4)
prob.model.add_design_var("ts.thick_ss", lower=0.05, upper=0.4)

# Constraints
prob.model.add_constraint('ts.err_rel_phi', lower=-1., upper=1.)
prob.model.add_constraint('ts.err_rel_psi', lower=-1., upper=1.)
prob.model.add_constraint('ts.err_rel_Lam', lower=-1., upper=1.)

# Set up optimizer
prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["disp"] = True

prob.model.add_objective("ts.lost_efficiency_rel")
prob.setup()

prob.run_driver()

print(prob.get_val("ts.lost_efficiency_rel")[0]*(1.-params.eta))
