"""Reject ToolManager and model capability catalog interchange."""

from avalan.model import ModelCapabilityCatalog
from avalan.tool.manager import ToolManager


def require_catalog(catalog: ModelCapabilityCatalog) -> None:
    """Require model-only capability state."""
    print(catalog)


def require_tool_manager(manager: ToolManager) -> None:
    """Require execution-only tool state."""
    print(manager)


tool_manager = ToolManager.create_instance()
capability_catalog = ModelCapabilityCatalog.create()

require_catalog(tool_manager)
require_tool_manager(capability_catalog)
