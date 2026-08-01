"""Prevent collected test modules from registering pytest plugins."""

from types import ModuleType

from pytest import Config, UsageError


def pytest_configure(config: Config) -> None:
    """Disable plugin registration from collected test modules."""

    def reject_module_plugins(module: ModuleType) -> None:
        specification = getattr(module, "pytest_plugins", ())
        if specification:
            raise UsageError(
                "module-level pytest_plugins are disabled by the contract "
                f"gate: {module.__name__}"
            )

    setattr(config.pluginmanager, "consider_module", reject_module_plugins)
