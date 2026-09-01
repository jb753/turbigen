"""Check that class members follow the standard ordering convention.

Order within each class must be:
  1. private (methods and properties, name starts with _)
  2. classmethods / staticmethods
  3. set_* methods, alphabetical
  4. get_* methods, alphabetical
  5. other public methods, alphabetical
  6. public properties, alphabetical

Sort order is case-insensitive with leading underscores stripped.

Adapted from ember's tools/check_member_order.py. Turbigen's classes mostly
don't follow this convention yet, so rather than fix the whole codebase in one
pass, CHECKED restricts enforcement to a chosen (file, class) allowlist and
grows as classes are brought into line. Start with MeanLine
(src/turbigen/meanline.py): iterate.py also defines a class named MeanLine
(an Iterator subclass, unrelated), which is why the allowlist is keyed on
file *and* class name rather than name alone.
"""

import ast
import sys
from pathlib import Path

PROP_DECORATORS = {"property", "cached_property"}
GROUP_ORDER = ["private", "classmethod", "set", "get", "other_public", "property"]
GROUP_SORTED = {"set", "get", "other_public", "property"}

CHECKED = {
    ("src/turbigen/meanline.py", "MeanLine"),
}


def _sort_key(s):
    return s.lstrip("_").lower()


def _decorator_names(node):
    names = set()
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.add(dec.id)
        elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            names.add(dec.func.id)
        elif isinstance(dec, ast.Attribute):
            names.add(dec.attr)
    return names


def _member_group(name, dec_names):
    if name.startswith("_"):
        return "private"
    if "classmethod" in dec_names or "staticmethod" in dec_names:
        return "classmethod"
    if name.startswith("set_"):
        return "set"
    if name.startswith("get_"):
        return "get"
    if PROP_DECORATORS & dec_names:
        return "property"
    return "other_public"


def check_file(path):
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"{path}: SyntaxError: {e}")
        return True

    path_str = path.as_posix()
    violations = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if (path_str, cls.name) not in CHECKED:
            continue

        members = []
        for node in cls.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            dec_names = _decorator_names(node)
            if dec_names & {"setter", "deleter"}:
                continue
            members.append(
                (node.name, _member_group(node.name, dec_names), node.lineno)
            )

        seen_rank = -1
        for name, group, lineno in members:
            rank = GROUP_ORDER.index(group)
            if rank < seen_rank:
                violations.append(
                    f"  {path}:{lineno}: {cls.name}.{name} ({group}) appears after a later group"
                )
            seen_rank = max(seen_rank, rank)

        for group in GROUP_SORTED:
            names = [n for n, g, _ in members if g == group]
            expected = sorted(names, key=_sort_key)
            if names != expected:
                lineno = next(ln for n, g, ln in members if g == group)
                violations.append(
                    f"  {path}:{lineno}: {cls.name} [{group}] not alphabetical:"
                    f" {names} != {expected}"
                )

    return violations


def main():
    paths = (
        [Path(p) for p in sys.argv[1:]]
        if sys.argv[1:]
        else sorted(Path("src/turbigen").glob("**/*.py"))
    )
    all_violations = []
    for path in paths:
        all_violations.extend(check_file(path))

    if all_violations:
        print("Class member ordering violations:")
        for v in all_violations:
            print(v)
        sys.exit(1)


if __name__ == "__main__":
    main()
