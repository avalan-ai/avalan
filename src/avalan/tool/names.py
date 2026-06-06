def matches_tool_namespace(tool_name: str, namespace: str | None) -> bool:
    """Return whether a tool name belongs to a namespace."""
    if not namespace:
        return True
    return tool_name == namespace or tool_name.startswith(f"{namespace}.")
