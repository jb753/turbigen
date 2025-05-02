"""Make documentation for an abstract base class and subclasses."""

import turbigen.meanline
from turbigen import util
import inspect


def generate_subclass(cls, fname):
    rst_str = ""

    # Base class first
    doc = inspect.getdoc(cls).split("xxx")
    rst_str += doc[0]

    # Now subclasses
    for subclass in cls.__subclasses__():
        rst_str += "\n\n"
        rst_str += inspect.getdoc(subclass)
        rst_str += "\n\n"

        cls_name = util.camel_to_snake(subclass.__name__)
        try:
            rst_str += format_yaml_snippet(subclass._design_vars, cls_name)
        except AttributeError:
            pass

    if len(doc) > 1:
        rst_str += doc[1]

    # Write the rst string to a file
    with open(f"doc/{fname}.rst", "w") as f:
        f.write(rst_str)


def format_yaml_snippet(data_dict, title):
    lines = ["mean_line:"]
    lines.append(f"  type: {title}")
    for name, (desc, shape, default) in data_dict.items():
        shape_str = ", " + str(shape) + " vector" if shape else ""
        default_str = "" if default is None else str(default)
        lines.append(f"  # {desc}{shape_str}")
        lines.append(f"  {name}: {default_str}")

    # Apply rst indentation
    lines = ["    " + line for line in lines]
    lines.insert(0, ".. code-block:: yaml\n")

    blurb = "To use this architecture, add the following snippet in your configuration file:\n\n"

    return blurb + "\n".join(lines)


if __name__ == "__main__":
    print(generate_subclass(turbigen.meanline.MeanLineDesigner, "meanline"))
