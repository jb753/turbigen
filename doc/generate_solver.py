"""Make documentation for an abstract base class and subclasses."""

import turbigen.solvers.base
from turbigen.solvers import ember, ts3, ts4
import inspect
import dataclasses


def generate_subclass(cls, fname):
    rst_str = ""

    # Base class first
    doc = inspect.getdoc(cls).split("xxx")
    rst_str += doc[0]

    # Now subclasses
    for subclass in cls.__subclasses__():
        print(subclass)
        rst_str += "\n\n"
        rst_str += inspect.getdoc(subclass)
        rst_str += "\n\n"
        rst_str += generate_rst_table(subclass)

    if len(doc) > 1:
        rst_str += doc[1]

    # Write the rst string to a file
    with open(f"doc/{fname}.rst", "w") as f:
        f.write(rst_str)


def generate_rst_table(cls):
    # Extract all dataclass fields
    fields = dataclasses.fields(cls)
    names = [f.name for f in fields]

    # Extract docstrings from the class source
    source_lines = inspect.getsourcelines(cls)[0]
    doc_map = {}
    current_field = None

    for line in source_lines:
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        elif ":" in line and "=" in line:
            name = line.split(":")[0].strip()
            if name in names:
                current_field = name
                print(f"Found field: {name}")
        elif '"""' in line or "'''" in line:
            doc = line.strip(' """\'')
            if current_field:
                print(f"Found docstring: {doc}")
                doc_map[current_field] = doc
                current_field = None

    doc_map.pop("workdir", None)

    # RST table header
    lines = [
        "Configuration options",
        "~~~~~~~~~~~~~~~~~~~~~",
        "",
        ".. list-table::",
        "   :widths: 10 10 10 70",
        "   :header-rows: 1",
        "",
        "   * - Name",
        "     - Type",
        "     - Default",
        "     - Description",
    ]

    # Sort alphabetically
    fields = sorted(fields, key=lambda f: f.name)

    for f in fields:
        name = f.name
        type_ = f.type.__name__ if hasattr(f.type, "__name__") else str(f.type)
        default = f.default if f.default != dataclasses.MISSING else "Required"
        if not (doc := doc_map.get(name, None)):
            continue
        if len(str(default)) > 10:
            default = str(default)[:10] + "..."
        try:
            getattr(turbigen.solvers.base.BaseSolver, name)
            print(f"Skipping field: {name}")
            continue
        except AttributeError:
            pass
        # Catch empty defaults
        if default == "":
            default = " "
        lines.append(f"   * - ``{name}``")
        lines.append(f"     - ``{type_}``")
        lines.append(f"     - ``{default}``")
        lines.append(f"     - {doc}")

    return "\n".join(lines)


if __name__ == "__main__":
    generate_subclass(turbigen.solvers.base.BaseSolver, "solver")
