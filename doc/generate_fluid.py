"""Make documentation for fluid objects."""

import turbigen.fluid
import inspect

prop_names = {
    "rho": "Density",
    "u": "Internal energy",
    "T": "Temperature",
    "P": "Pressure",
    "h": "Enthalpy",
    "s": "Entropy",
}


def generate(cls):
    rst_str = ""

    # Base class first
    doc = inspect.getdoc(cls)
    rst_str += doc

    # Get names of setter methods
    setters = [
        m[0]
        for m in inspect.getmembers(cls, predicate=inspect.isfunction)
        if m[0].startswith("set_")
    ]
    # Start table
    setter_str = ".. list-table::\n   :widths: 50 25 25\n\n"
    for method in setters:
        props = method.split("_")[1:]
        methods_str = f"   * - ``{cls.__name__}.{method}({', '.join(props)})``"
        params_str = "\n     - ".join([prop_names[p] for p in props])
        setter_str += f"\n{methods_str}\n     - {params_str}"

    rst_str = rst_str.replace("xxx", setter_str)

    # Get names and docstrings of quantities that use @property decorator
    quantities = [
        m[0]
        for m in inspect.getmembers(cls)
        if isinstance(m[1], property) and not m[0].startswith("_")
    ]
    # Start quantities table
    quantities_str = ".. list-table::\n   :widths: 25 50 25\n   :header-rows: 1\n\n"
    quantities_str += "   * - Property\n     - Description\n     - Units\n"
    for quantity in quantities:
        doc = inspect.getdoc(getattr(cls, quantity))
        # get units
        name, units = doc.split(" [")
        units = units.split("]")[0]
        quantities_str += (
            f"\n   * - ``{cls.__name__}.{quantity}``\n     - {name}\n     - {units}"
        )
    rst_str = rst_str.replace("yyy", quantities_str)

    # Write the rst string to a file
    with open("doc/fluid.rst", "w") as f:
        f.write(rst_str)

    return rst_str


if __name__ == "__main__":
    print(generate(turbigen.fluid.State))
