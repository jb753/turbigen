"""Tests for assembling a config document out of several files.

No designs and no CFD: everything here is dicts in and a dict out, so the
fixtures are the smallest mappings that make each rule visible rather than
realistic configs. What a real config does with the result is covered by
`test_cli.py`.

Test cases:
- test_a_file_without_includes_reads_unchanged: the identity case
- test_an_included_section_appears: the point of the feature
- test_the_including_file_wins_one_level_deep: type and cfl survive an override
- test_a_list_is_replaced_not_merged: blades cannot merge by index
- test_a_mapping_and_a_scalar_replace_each_other: only two mappings merge
- test_sibling_includes_are_composed: disjoint sections assemble
- test_siblings_claiming_one_key_are_refused: no precedence between equals
- test_an_include_can_include: fragments compose into a library
- test_a_nested_include_resolves_against_its_own_file: whoever names it
- test_a_diamond_resolves: two files may include one third
- test_a_cycle_names_the_loop: and a loop is not a diamond
- test_a_missing_include_names_who_included_it: both halves of the mistake
- test_a_string_include_says_how_to_write_it: one spelling, with a way out
- test_a_non_list_include_is_refused: and anything else is just wrong
- test_an_included_result_is_dropped: an answer belongs to the run that got it
- test_the_dropped_key_is_the_one_case_uses: and the two spellings agree
- test_resolution_does_not_depend_on_the_working_directory: the old defect
- test_a_key_written_twice_is_refused: at the top level
- test_a_nested_key_written_twice_is_refused: and anywhere below it
- test_a_duplicate_inside_a_list_is_refused: including inside a sequence
- test_a_duplicate_in_an_included_file_is_refused: every file, not just the top
- test_an_empty_file_is_an_empty_mapping: rather than a None to trip over
"""

import pytest

from turbigen2 import case, include


def write(tmp_path, **files):
    """Write each named file, returning the directory holding them.

    Names are keyword arguments, so a file in a subdirectory or with a dot in
    it is passed as ``**{"frag/leaf.yaml": ...}``.
    """
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


#
# MERGING
#


def test_a_file_without_includes_reads_unchanged(tmp_path):
    write(tmp_path, top="solver: {type: ember, n_step: 100}\nmesh: {type: h}\n")

    assert include.read(tmp_path / "top") == {
        "solver": {"type": "ember", "n_step": 100},
        "mesh": {"type": "h"},
    }


def test_an_included_section_appears(tmp_path):
    write(
        tmp_path,
        top="include: [solver.yaml]\nmesh: {type: h}\n",
        **{"solver.yaml": "solver: {type: ember, n_step: 2500}\n"},
    )

    assert include.read(tmp_path / "top") == {
        "solver": {"type": "ember", "n_step": 2500},
        "mesh": {"type": "h"},
    }


def test_the_including_file_wins_one_level_deep(tmp_path):
    """The case the depth exists for: override one setting, keep the block.

    Deeper would splice mappings that declare different `type:` keys into one
    node with fields from two designs, which fails validation somewhere
    unrelated to the mistake.
    """
    write(
        tmp_path,
        top="include: [solver.yaml]\nsolver: {n_step: 100}\n",
        **{"solver.yaml": "solver: {type: ember, n_step: 2500, cfl: 5.0}\n"},
    )

    assert include.read(tmp_path / "top") == {
        "solver": {"type": "ember", "n_step": 100, "cfl": 5.0}
    }


def test_a_list_is_replaced_not_merged(tmp_path):
    """Merging `blades:` by index would give a row nobody asked for."""
    write(
        tmp_path,
        top="include: [rows.yaml]\nblades: [{count: 3}]\n",
        **{"rows.yaml": "blades: [{count: 1}, {count: 2}]\n"},
    )

    assert include.read(tmp_path / "top") == {"blades": [{"count": 3}]}


def test_a_mapping_and_a_scalar_replace_each_other(tmp_path):
    write(
        tmp_path,
        top="include: [base.yaml]\na: 1\nb: {x: 1}\n",
        **{"base.yaml": "a: {x: 1}\nb: 2\n"},
    )

    assert include.read(tmp_path / "top") == {"a": 1, "b": {"x": 1}}


#
# SIBLINGS
#


def test_sibling_includes_are_composed(tmp_path):
    write(
        tmp_path,
        top="include: [solver.yaml, mesh.yaml]\n",
        **{"solver.yaml": "solver: {type: ember}\n", "mesh.yaml": "mesh: {type: h}\n"},
    )

    assert include.read(tmp_path / "top") == {
        "solver": {"type": "ember"},
        "mesh": {"type": "h"},
    }


def test_siblings_claiming_one_key_are_refused(tmp_path):
    """Two files in one list have no precedence between them.

    A file overriding what it includes is a hierarchy and is allowed; two
    equals disagreeing is a question the file has not answered, and resolving
    it by list order would be a rule nobody should have to know.
    """
    write(
        tmp_path,
        top="include: [site.yaml, tuning.yaml]\n",
        **{
            "site.yaml": "solver: {type: ember, n_step: 2500}\n",
            "tuning.yaml": "solver: {cfl: 5.0}\n",
        },
    )

    with pytest.raises(ValueError, match="no precedence between them"):
        include.read(tmp_path / "top")


def test_a_sibling_conflict_says_how_to_settle_it(tmp_path):
    write(
        tmp_path,
        top="include: [one.yaml, two.yaml]\n",
        **{"one.yaml": "solver: {a: 1}\n", "two.yaml": "solver: {b: 2}\n"},
    )

    with pytest.raises(ValueError, match=r"set 'solver' in top itself"):
        include.read(tmp_path / "top")


#
# RECURSION
#


def test_an_include_can_include(tmp_path):
    write(
        tmp_path,
        top="include: [middle.yaml]\n",
        **{
            "middle.yaml": "include: [leaf.yaml]\nmesh: {type: h}\n",
            "leaf.yaml": "solver: {type: ember}\n",
        },
    )

    assert include.read(tmp_path / "top") == {
        "solver": {"type": "ember"},
        "mesh": {"type": "h"},
    }


def test_a_nested_include_resolves_against_its_own_file(tmp_path):
    """Relative to whoever named it, not to the file at the top."""
    write(
        tmp_path,
        top="include: [frag/middle.yaml]\n",
        **{
            "frag/middle.yaml": "include: [leaf.yaml]\n",
            "frag/leaf.yaml": "solver: {type: ember}\n",
        },
    )

    assert include.read(tmp_path / "top") == {"solver": {"type": "ember"}}


def test_a_diamond_resolves(tmp_path):
    """Two files may include one third; that is not a loop."""
    write(
        tmp_path,
        top="include: [left.yaml, right.yaml]\n",
        **{
            "left.yaml": "include: [shared.yaml]\nmesh: {type: h}\n",
            "right.yaml": "job: {type: slurm}\n",
            "shared.yaml": "fluid: {type: perfect}\n",
        },
    )

    assert set(include.read(tmp_path / "top")) == {"mesh", "job", "fluid"}


def test_a_cycle_names_the_loop(tmp_path):
    write(
        tmp_path,
        top="include: [a.yaml]\n",
        **{"a.yaml": "include: [b.yaml]\n", "b.yaml": "include: [a.yaml]\n"},
    )

    with pytest.raises(ValueError, match="include each other in a loop"):
        include.read(tmp_path / "top")


#
# WHAT IS REFUSED
#


def test_a_missing_include_names_who_included_it(tmp_path):
    write(tmp_path, top="include: [absent.yaml]\n")

    with pytest.raises(ValueError, match="absent.yaml") as excinfo:
        include.read(tmp_path / "top")

    # Both halves of the mistake: what was asked for, and who asked.
    assert "top" in str(excinfo.value)


def test_a_string_include_says_how_to_write_it(tmp_path):
    write(tmp_path, top="include: solver.yaml\n")

    with pytest.raises(ValueError, match=r"include: \[solver.yaml\]"):
        include.read(tmp_path / "top")


def test_a_non_list_include_is_refused(tmp_path):
    write(tmp_path, top="include: {solver: solver.yaml}\n")

    with pytest.raises(ValueError, match="not a list of config file names"):
        include.read(tmp_path / "top")


#
# AN INCLUDED ANSWER
#


def test_an_included_result_is_dropped(tmp_path):
    """Including a finished output.yaml to inherit its design is a fair thing
    to want; inheriting the answer it achieved is not."""
    write(
        tmp_path,
        top="include: [output.yaml]\n",
        **{"output.yaml": "mean_line: {psi: 1.6}\nresult: {converged: true}\n"},
    )

    assert include.read(tmp_path / "top") == {"mean_line": {"psi": 1.6}}


def test_a_top_level_result_is_kept(tmp_path):
    """Only an *included* answer is dropped; `case.read` still finds its own."""
    write(tmp_path, top="mean_line: {psi: 1.6}\nresult: {converged: true}\n")

    assert include.read(tmp_path / "top")["result"] == {"converged": True}


def test_the_dropped_key_is_the_one_case_uses():
    """The string is duplicated because case imports config, which imports
    this, so the name cannot come the other way. Duplicated, not drifting."""
    assert case.RESULT_KEY in include.DROPPED_FROM_INCLUDES


#
# WHERE FILES ARE LOOKED FOR
#


def test_resolution_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    """The defect this replaces.

    The old loader tried the bare name against the working directory first, so
    the same config designed a different machine depending on where it was
    invoked from. A decoy of the same name is planted where the old code would
    have found it.
    """
    write(
        tmp_path,
        **{
            "case/top": "include: [solver.yaml]\n",
            "case/solver.yaml": "solver: {n_step: 2500}\n",
            "elsewhere/solver.yaml": "solver: {n_step: 999999}\n",
        },
    )

    monkeypatch.chdir(tmp_path / "elsewhere")

    assert include.read(tmp_path / "case" / "top") == {"solver": {"n_step": 2500}}


#
# KEYS WRITTEN TWICE
#
# Every loader keeps the last of a repeated key and says nothing, so a setting
# written twice is a setting silently ignored. The old package wrote this check
# and wired it to `read_yaml_list` alone, so no config was ever checked by it.
#


def test_a_key_written_twice_is_refused(tmp_path):
    write(tmp_path, top="solver: {a: 1}\nmesh: {type: h}\nsolver: {b: 2}\n")

    with pytest.raises(ValueError, match=r"sets 'solver' twice at the top level"):
        include.read(tmp_path / "top")


def test_a_nested_key_written_twice_is_refused(tmp_path):
    write(tmp_path, top="solver:\n  n_step: 100\n  n_step: 200\n")

    with pytest.raises(ValueError, match=r"sets 'n_step' twice in solver"):
        include.read(tmp_path / "top")


def test_a_duplicate_names_its_line(tmp_path):
    write(tmp_path, top="a: 1\nb: 2\nc: 3\na: 4\n")

    with pytest.raises(ValueError, match="on line 4"):
        include.read(tmp_path / "top")


def test_a_duplicate_inside_a_list_is_refused(tmp_path):
    write(tmp_path, top="blades:\n  - count: 1\n    count: 2\n")

    with pytest.raises(ValueError, match=r"twice in blades\[0\]"):
        include.read(tmp_path / "top")


def test_a_duplicate_in_an_included_file_is_refused(tmp_path):
    """Every file is checked, not only the one that was named."""
    write(
        tmp_path,
        top="include: [solver.yaml]\n",
        **{"solver.yaml": "solver: {a: 1}\nsolver: {b: 2}\n"},
    )

    with pytest.raises(ValueError, match="solver.yaml sets 'solver' twice"):
        include.read(tmp_path / "top")


def test_an_empty_file_is_an_empty_mapping(tmp_path):
    write(tmp_path, top="include: [empty.yaml]\nmesh: {type: h}\n", **{"empty.yaml": ""})

    assert include.read(tmp_path / "top") == {"mesh": {"type": "h"}}
