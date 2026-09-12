"""Registry of mean-line designers, including user-supplied plugins."""

import importlib.util
import inspect
import logging

logger = logging.getLogger("turbigen")

REGISTRY = {
    "designer": {},
}

_BUILTINS_LOADED = False


def _load_builtin_designers():
    """Import the designers shipped with turbigen, once.

    Deferred rather than a module-level import because the built-in designers
    import this module to reach :func:`register_designer`. Called from
    :func:`get_registry` so that registration cannot depend on some unrelated
    module happening to be imported first, which is how the built-ins were
    previously reaching the registry.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Set before importing: the import triggers register_designer, which calls
    # get_registry, which would otherwise recurse back into here.
    _BUILTINS_LOADED = True
    import turbigen_ref.meanline_design_new  # noqa: F401


def get_registry():
    """Return the plugin registry, with built-in designers loaded."""
    _load_builtin_designers()
    return REGISTRY


def load_plugins(plugdir):
    """Find and load plugins from the plugdir."""

    logger.info(f"Loading plugins from directory: {plugdir}")

    # Make sure the built-ins are in place before user code can override them
    get_registry()

    for py_file in plugdir.rglob("*.py"):
        # Exclude hidden files and directories
        if any(part.startswith(".") for part in py_file.parts):
            continue

        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info(f"Imported plugin module: {py_file}")
        except Exception as err:
            logger.error(f"Failed to import plugin: '{py_file}'", exc_info=True)
            raise RuntimeError(f"Failed to import plugin '{py_file}': {err}") from err


def check_plugins():
    """Verify the registry is usable."""
    reg = get_registry()
    if not reg["designer"]:
        raise RuntimeError(
            "No mean-line designers are registered. The built-in designers "
            "failed to load, and no plugin directory supplied any."
        )


def list_plugins():
    """Log the available mean-line types and their design variables."""
    reg = get_registry()
    logger.info("Available mean line types:")
    for name, designer in sorted(reg["designer"].items()):
        import turbigen_ref.designer

        params = turbigen_ref.designer.design_params(designer)
        shown = ", ".join(
            k if v is turbigen_ref.designer.REQUIRED else f"{k}={v!r}"
            for k, v in params.items()
        )
        logger.info(f"  {name}(n_row={designer.n_row}): {shown}")


def register_designer(name):
    """Register a :class:`turbigen_ref.designer.Designer` subclass under `name`.

    Used as a decorator on the class::

        @register_designer("axial_turbine")
        class AxialTurbine(Designer):
            n_row = 2
            ...

    The registered object is an *instance*, so a designer may hold whatever
    state its methods need.
    """

    def decorator(cls):
        import turbigen_ref.designer

        if not (inspect.isclass(cls) and issubclass(cls, turbigen_ref.designer.Designer)):
            raise TypeError(
                f"register_designer('{name}') must decorate a subclass of "
                f"turbigen_ref.designer.Designer, got {cls!r}."
            )

        if not isinstance(cls.n_row, int) or cls.n_row < 1:
            raise ValueError(
                f"Designer '{name}' must set n_row to a positive integer, "
                f"got {cls.n_row!r}."
            )

        for method in ("forward", "backward"):
            if method not in vars(cls) and getattr(cls, method) is getattr(
                turbigen_ref.designer.Designer, method
            ):
                raise TypeError(f"Designer '{name}' does not implement {method}().")

        fwd = list(inspect.signature(cls.forward).parameters.values())
        if len(fwd) < 2 or fwd[1].name != "ml":
            got = fwd[1].name if len(fwd) > 1 else "none"
            raise TypeError(
                f"Designer '{name}': forward()'s first argument after self must "
                f"be named 'ml', got '{got}'."
            )

        bwd = list(inspect.signature(cls.backward).parameters.values())
        if len(bwd) != 2 or bwd[1].name != "ml":
            raise TypeError(
                f"Designer '{name}': backward() must take exactly one argument "
                f"after self, named 'ml'."
            )

        if name in REGISTRY["designer"]:
            existing = type(REGISTRY["designer"][name]).__name__
            raise ValueError(
                f"Designer name '{name}' is already registered by {existing}."
            )

        cls.name = name
        REGISTRY["designer"][name] = cls()

        return cls

    return decorator
