"""Check that we can use compressible flow functions with or without the compflow package."""
import builtins
import pytest
import importlib


def test_without_compflow_package():

    # Override builtin import to pretend we don't have compflow
    # See https://stackoverflow.com/a/60229056
    old_import = builtins.__import__

    def new_import(name, *args, **kwargs):
        if name == "compflow":
            raise ImportError()
        return old_import(name, *args, **kwargs)

    setattr(builtins, "__import__", new_import)

    # Forcibly re-import module and check for warning
    with pytest.warns(UserWarning):
        import turbigen.compflow

        importlib.reload(turbigen.compflow)

    # Restore import function and put the old compflow back
    setattr(builtins, "__import__", old_import)
    importlib.reload(turbigen.compflow)
