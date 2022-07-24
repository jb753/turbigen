"""Trying OpenMDAO."""
import openmdao.api as om
import os
import numpy as np

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

        try:
            # the parent compute function actually runs the external code
            super().compute(inputs, outputs)
        except:
            raise om.AnalysisError(
                "TS failed %s" % os.path.join(*os.path.split(workdir)[-2:])
            )

        # parse the output file from the external code
        m = submit.load_results(output_file_path)

        # Insert outputs in-place using abstract method
        self.outputs_from_metadata(m, outputs)


class MeanLineTurbostreamComp(BaseTurbostreamComp):
    """Run Turbostream with adjustments to recamber, loss ratio, effy."""

    def initialize(self):
        self.options.declare("row_index")
        super().initialize()

    def setup(self):

        self.add_input("recamber", val=0.0)
        self.add_input("efficiency", val=self.options["datum_params"].eta)
        self.add_input("loss_rat", val=self.options["datum_params"].loss_rat)

        self.add_output("deviation")
        self.add_output("efficiency_out")
        self.add_output("loss_rat_out")
        self.add_output("runid")

        stg = design.nondim_stage_from_Lam(
            **self.options["datum_params"].nondimensional
        )
        self.Al_target = np.array([stg.Al[1], stg.Alrel[2]])

        super().setup()

    def params_from_inputs(self, inputs):

        print("In : %s" % str(inputs))

        param_now = self.options["datum_params"].copy()
        param_now.eta = inputs["efficiency"][0]
        param_now.loss_rat = inputs["loss_rat"][0]
        irow = self.options["row_index"]
        param_now.recamber[irow * 2 + 1] = inputs["recamber"][0]

        return param_now

    def outputs_from_metadata(self, metadata, outputs):

        Al_now = np.array([metadata["Al"][1], metadata["Alrel"][3]])
        Al_err = Al_now - self.Al_target
        outputs["efficiency_out"] = metadata["eta"]
        outputs["loss_rat_out"] = metadata["loss_rat"]
        irow = self.options["row_index"]
        outputs["deviation"] = Al_err[irow]
        outputs["runid"] = metadata["runid"]

        print("Out: %s" % str(outputs))


class SectionTurbostreamComp(BaseTurbostreamComp):
    """Run Turbostream with parameterised blade section."""

    def initialize(self):
        self.options.declare("row_index")
        self.options.declare("penalise_constraints")
        super().initialize()

    def setup(self):

        self.add_input("stagger_rel", val=1.0)
        self.add_input("radius_le_rel", val=1.0)
        self.add_input("beta_rel", val=1.0)
        self.add_input("thick_ps_rel", val=1.0)
        self.add_input("thick_ss_rel", val=1.0)

        self.add_output("lost_efficiency_rel")
        self.add_output("err_phi_rel")
        self.add_output("err_psi_rel")
        self.add_output("err_Lam_rel")

        self.add_output("runid")

        # All partial derivatives approximated by finite difference
        # Most variables use a relative step
        self.declare_partials(
            of="*", wrt="*", method="fd", step=0.1, step_calc="rel_element"
        )
        # For recamber which may go through zero, override with constant step
        # self.declare_partials(of="*", wrt="recamber_*", method="fd", step=0.2, step_calc='abs')

        super().setup()

    def params_from_inputs(self, inputs):

        print("In : %s" % str(inputs))

        param_datum = self.options["datum_params"]
        param_now = param_datum.copy()

        irow = self.options["row_index"]

        param_now.stag[irow] = inputs["stagger_rel"][0] * param_datum.stag[irow]

        # param_now.recamber[irow*2] = inputs["recamber_le"][0]
        # param_now.recamber[irow*2+1] = inputs["recamber_te"][0]

        Rle_d, thick_d, beta_d = geometry.Rle_thick_beta_from_A(
            param_datum.A[irow], param_datum.tte
        )

        Rle = inputs["radius_le_rel"][0] * Rle_d
        beta = inputs["beta_rel"][0] * beta_d
        thick = (
            np.array([[inputs["thick_ps_rel"][0]], [inputs["thick_ss_rel"][0]]])
            * thick_d[0, 0]
        )
        param_now.A[irow] = geometry.A_from_Rle_thick_beta(
            Rle, thick, beta, param_now.tte
        )

        return param_now

    def outputs_from_metadata(self, metadata, outputs):

        params = self.options["datum_params"]

        outputs["lost_efficiency_rel"] = metadata["eta_lost"] / (
            1.0 - params.eta
        )
        for v in ["phi", "psi", "Lam"]:
            outputs["err_" + v + "_rel"] = (
                metadata[v] / getattr(params, v) - 1.0
            ) / params.rtol

        outputs["runid"] = metadata["runid"]

        W = self.options["penalise_constraints"]
        if W:
            for v in ["phi", "psi", "Lam"]:
                constr = np.abs(outputs["err_" + v + "_rel"]) - 1.0
                if constr > 0.0:
                    outputs["lost_efficiency_rel"] += constr * W

        print("Out: %s" % str(outputs))


def correct_deviation(params):
    recamber = np.zeros((2,))
    row_indices = [0, 1, 0]

    base_dir = "run/mean-line"

    for k, irow in enumerate(row_indices):

        print("* row %d" % irow)
        params_now = params.copy()
        params_now.recamber[
            (1, 3),
        ] = recamber

        # build the model
        prob = om.Problem()
        model = prob.model
        model.add_subsystem(
            "ts",
            MeanLineTurbostreamComp(
                row_index=irow, datum_params=params_now, base_dir=base_dir
            ),
            promotes=["*"],
        )

        prob.setup()

        cache = {}

        def iterate(x):
            if x in cache:
                return cache[x]
            prob.set_val("recamber", x)
            prob.run_model()
            prob.set_val("efficiency", prob.get_val("efficiency_out")[0])
            prob.set_val("loss_rat", prob.get_val("loss_rat_out")[0])
            dev = prob.get_val("deviation")[0]
            cache[x] = dev
            return dev

        # Attempt to bracket zero-deviation point

        # Look for a positive deviation
        recam_upper = recamber[irow] + 0.0
        recam_lower = recamber[irow] + 0.0
        flip = -1.0 if irow else 1.0
        while True:
            print("recam", recam_upper, recam_lower)
            dev_upper = iterate(recam_upper) * flip
            if dev_upper > 0.0:
                print("found positive dev")
                break
            else:
                if recam_upper > recam_lower:
                    recam_lower = recam_upper + 0.0
                recam_upper += 1.0

        # Look for a negative deviation
        while True:
            print("recam", recam_upper, recam_lower)
            dev_lower = iterate(recam_lower) * flip
            if dev_lower < 0.0:
                print("found negative dev")
                break
            else:
                if recam_lower < recam_upper:
                    recam_upper = recam_lower + 0.0
                recam_lower -= 1.0

        # tol = 0.5
        # brak = (recam_lower, recam_upper)
        # recamber[irow] = root_scalar( iterate, bracket=brak, xtol=tol).root

        # We now have the zero-deviation point inside a 1 degree interval
        # So just linearly interpolate to get good enough
        recamber[irow] = recam_lower - dev_lower * (
            recam_upper - recam_lower
        ) / (dev_upper - dev_lower)

        output_hdf5_path = os.path.abspath(
            os.path.join(
                base_dir, str(int(prob.get_val("runid")[0])), "output_avg.hdf5"
            )
        )
        params.eta = prob.get_val("efficiency_out")[0]
        params.loss_rat = prob.get_val("loss_rat_out")[0]
        params.recamber[1], params.recamber[3] = recamber
        params.guess_file = output_hdf5_path


def run_once(params):
    if params.ilos == -1:
        base_dir = "./om_poisson"
    elif params.ilos == 1:
        base_dir = "./om_ml"
    elif params.ilos == 2:
        base_dir = "./om_sa"
    else:
        pass

    # Set up model
    prob = om.Problem()
    model = prob.model
    model.add_subsystem(
        "ts",
        SectionTurbostreamComp(
            row_index=0, datum_params=params, base_dir=base_dir
        ),
        promotes=["*"],
    )
    prob.setup()
    prob.run_model()

    output_hdf5_path = os.path.abspath(
        os.path.join(
            base_dir, str(int(prob.get_val("runid")[0])), "output.hdf5"
        )
    )

    return output_hdf5_path


# Poisson calc
params = submit.ParameterSet.from_json("datum_deviation_corrected_sa_2.json")

# # correct_deviation(params)
# # params.write_json('datum_deviation_corrected_sa_2.json')
# run_once(params)


# Set up model
prob = om.Problem()
model = prob.model
model.add_subsystem(
    "ts",
    SectionTurbostreamComp(
        row_index=0,
        datum_params=params,
        base_dir="run/opt_stator",
        penalise_constraints=100.0,
    ),
    promotes=["*"],
)

prob.setup()
prob.run_model()
quit()

# Design variables
prob.model.add_design_var(
    "stagger_rel", lower=0.5, upper=1.5
)  # For vane, stag +ve
# prob.model.add_design_var("ts.recamber_le", lower=-5., upper=10.)
# prob.model.add_design_var("ts.recamber_te", lower=-5., upper=5.)
prob.model.add_design_var("radius_le_rel", lower=0.5, upper=1.5)
prob.model.add_design_var("beta_rel", lower=0.5, upper=1.5)
prob.model.add_design_var("thick_ps_rel", lower=0.5, upper=1.5)
prob.model.add_design_var("thick_ss_rel", lower=0.5, upper=1.5)

# Constraints
prob.model.add_constraint("err_phi_rel", lower=-1.0, upper=1.0)
prob.model.add_constraint("err_psi_rel", lower=-1.0, upper=1.0)
prob.model.add_constraint("err_Lam_rel", lower=-1.0, upper=1.0)

# Set up optimizer
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["disp"] = True
prob.driver.options["tol"] = 0.01
# prob.driver.opt_settings["rhobeg"] = 0.2
prob.driver.options["optimizer"] = "Nelder-Mead"

prob.model.add_objective("lost_efficiency_rel")
prob.setup()

prob.run_driver()

print(prob.get_val("lost_efficiency_rel")[0] * (1.0 - params.eta))
