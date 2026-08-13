"""Guard what MeanLine takes over from ember's Block.

MeanLine used to mirror Block's API by hand, one forwarding property per
quantity, and this file checked the mirror was complete. MeanLine now *is* a
Block, so that coverage is automatic and the old check is meaningless.

What is worth pinning instead is the opposite risk. Because every ember
property is inherited, a name defined here silently takes precedence over
ember's. That is deliberate for four members and would be a bug for any other:
a future ember release adding, say, a `span` or `mdot` property would be
shadowed by ours without a word. This test fails when the overridden set
changes, so the collision has to be looked at rather than absorbed.
"""

import ember.block

import turbigen.meanline_new

# Members of Block that MeanLine deliberately replaces.
EXPECTED_OVERRIDES = {
    # The added Am and Omega storage.
    "_data_keys",
    # Omega is nodal data here, not scalar block metadata, so that each row
    # carries its own blade speed through slicing.
    "Omega",
    "Omega_nd",
    "set_Omega",
    # set_L_ref additionally rescales the non-dimensionally stored area.
    "set_L_ref",
}


def _overridden_members():
    """Names defined on MeanLine that also exist on Block."""
    return {
        name
        for name in vars(turbigen.meanline_new.MeanLine)
        if not name.startswith("__")
        and hasattr(ember.block.Block, name)
    }


def test_meanline_overrides_only_what_it_means_to():
    """No MeanLine member shadows a Block member by accident."""
    actual = _overridden_members()

    unexpected = actual - EXPECTED_OVERRIDES
    assert not unexpected, (
        f"MeanLine shadows Block members that are not declared intentional: "
        f"{sorted(unexpected)}. Either rename the MeanLine member or add it to "
        f"EXPECTED_OVERRIDES with a note on why the override is correct."
    )

    stale = EXPECTED_OVERRIDES - actual
    assert not stale, (
        f"EXPECTED_OVERRIDES lists members MeanLine no longer overrides: "
        f"{sorted(stale)}. Drop them from the list."
    )


def test_inherited_block_properties_are_not_reimplemented():
    """The flow-field API comes straight from Block, not from a copy of it."""
    for name in (
        "Po",
        "To",
        "Ma",
        "Ma_rel",
        "Alpha",
        "Alpha_rel",
        "Beta",
        "s",
        "h",
        "ho",
        "ho_rel",
        "Po_rel",
        "V",
        "V_rel",
        "Vm",
        "U",
        "rho",
        "conserved",
    ):
        assert getattr(turbigen.meanline_new.MeanLine, name) is getattr(
            ember.block.Block, name
        ), f"{name} is reimplemented on MeanLine rather than inherited"
