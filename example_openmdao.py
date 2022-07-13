"""Trying OpenMDAO."""
import openmdao.api as om
import os
import numpy as np
from scipy.optimize import newton

from turbigen import submit, design


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
        self.options["command"] = ["run_turbostream_with_env.sh", input_file_path]

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

        self.declare_partials(of="*", wrt="*", method="fd", step=0.5)

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
        print(Al_err)
        irow = self.options["row_index"]
        dev = np.abs(Al_err[irow])
        outputs["efficiency_out"] = metadata["eta"]
        outputs["deviation"] = dev


def correct_deviation(params):

    recamber = np.array([-1.,-1.])

    for irow in [0,1,0]:

        # build the model
        prob = om.Problem()
        model = prob.model
        model.add_subsystem(
            "ts",
            DeviationTurbostreamComp(
                row_index=irow, datum_params=params, base_dir="om_test4"
            ),
        )

        prob.setup()

        def iterate(x):
            prob.set_val("ts.recamber", x)
            print(prob.get_val("ts.efficiency"))
            prob.run_model()
            prob.set_val("ts.efficiency", prob.get_val("ts.efficiency_out")[0])
            return prob.get_val("ts.deviation")[0]

        # save the data
        recamber[irow] = newton(iterate, x0=recamber[irow], x1=recamber[irow]-.5, tol=0.1)

    effy = prob.get_val("ts.efficiency_out")[0]

    params.recamber[1], params.recamber[3] = recamber
    params.eta = effy


params = submit.ParameterSet.from_default()
params.Ma2 = 0.3
params.Co = (0.7,0.7)

correct_deviation(params)
