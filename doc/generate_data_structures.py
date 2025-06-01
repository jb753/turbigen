"""Make documentation for fluid, flowfield, meanline data structures."""

import turbigen.fluid
import turbigen.base
import inspect

prop_names = {
    "rho": "Density",
    "u": "Internal energy",
    "T": "Temperature",
    "P": "Pressure",
    "h": "Enthalpy",
    "s": "Entropy",
}


def generate_fluid(cls):
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
    setters.sort(key=str.lower)

    # Start table
    setter_str = ".. list-table::\n   :widths: 50 25 25\n   :header-rows: 1\n\n"
    setter_str += "   * - Method\n     - Arguments\n     - "
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
    quantities.sort(key=str.lower)
    # Start quantities table
    quantities_str = ".. list-table::\n   :widths: 25 55 20\n   :header-rows: 1\n\n"
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

    return rst_str


def generate_flowfield(cls):
    rst_str = ""

    # Base class first
    doc = inspect.getdoc(cls)
    rst_str += doc

    # Get names and docstrings of quantities that use @property decorator
    quantities = [m[0] for m in inspect.getmembers(cls) if not m[0].startswith("_")]
    quantities.sort(key=str.lower)
    # Start quantities table
    quantities_str = ".. list-table::\n   :widths: 25 60 15\n   :header-rows: 1\n\n"
    quantities_str += "   * - Property\n     - Description\n     - Units\n"
    for quantity in quantities:
        doc = inspect.getdoc(getattr(cls, quantity))
        print(quantity)
        try:
            assert len(doc.split(" [")) == 2
            assert "\n" not in doc
            name, units = doc.split(" [")
        except (AttributeError, ValueError, AssertionError):
            continue
        units = units.split("]")[0]
        quantities_str += (
            f"\n   * - ``{cls.__name__}.{quantity}``\n     - {name}\n     - {units}"
        )
    rst_str = rst_str.replace("yyy", quantities_str)

    return rst_str


if __name__ == "__main__":
    state_str = generate_fluid(turbigen.fluid.State)

    rst_str = """
Data Structures
===============

This page documents the internal data structures used in :program:`turbigen`.
It is intended as a reference for users extending the program using custom
plugins or developers modifying the source code.


"""

    rst_str += state_str + "\n\n"

    rst_str += generate_flowfield(turbigen.base.FlowField)
    # Write the rst string to a file
    with open("doc/data_structures.rst", "w") as f:
        f.write(rst_str)
