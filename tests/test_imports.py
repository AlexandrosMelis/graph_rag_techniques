import importlib
import pkgutil

import pytest

import graph_rag

MODULES = sorted(
    info.name
    for info in pkgutil.walk_packages(graph_rag.__path__, prefix="graph_rag.")
    if not info.name.endswith("__main__")
)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith("graph_rag"):
            pytest.skip(f"third-party dependency not installed: {exc.name}")
        raise
