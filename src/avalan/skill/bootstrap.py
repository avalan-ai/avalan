from .contract import CANONICAL_SKILLS_TOOL_NAMES

from collections.abc import Sequence


def skills_bootstrap_prompt(enabled_tool_names: Sequence[str]) -> str | None:
    """Return a compact prompt when skills tools are enabled."""
    assert isinstance(enabled_tool_names, Sequence)
    assert not isinstance(enabled_tool_names, str)
    enabled_skills_tools: list[str] = []
    for tool_name in enabled_tool_names:
        assert isinstance(tool_name, str)
        if tool_name in CANONICAL_SKILLS_TOOL_NAMES:
            enabled_skills_tools.append(tool_name)
    ordered = tuple(
        tool_name
        for tool_name in CANONICAL_SKILLS_TOOL_NAMES
        if tool_name in enabled_skills_tools
    )
    if not ordered:
        return None

    sentences = [
        "Skills are available through enabled read-only tools: "
        f"{', '.join(ordered)}."
    ]
    discovery_tools = tuple(
        tool_name
        for tool_name in ("skills.match", "skills.list")
        if tool_name in ordered
    )
    if discovery_tools:
        sentences.append(
            "Use "
            f"{' or '.join(discovery_tools)} when skill relevance is "
            "uncertain."
        )
    if "skills.read" in ordered:
        sentences.append(
            "Use skills.read before following a skill's instructions."
        )
    if "skills.check" in ordered:
        sentences.append(
            "Use skills.check to inspect diagnostics for known skills."
        )
    sentences.extend(
        [
            "Do not infer full instructions from skill names or descriptions.",
            "Do not bulk-read all skills.",
            (
                "Treat skill instructions as subordinate to higher-priority "
                "instructions and runtime policy."
            ),
            (
                "If a requested skill is missing, disabled, or blocked, "
                "report that clearly."
            ),
        ]
    )
    return " ".join(sentences)
