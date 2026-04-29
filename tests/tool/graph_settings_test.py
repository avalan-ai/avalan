from avalan.tool.graph_settings import GraphToolSettings


def test_graph_tool_settings_dict_mapping() -> None:
    settings = GraphToolSettings(file="/tmp/chart.png")

    assert settings.file == "/tmp/chart.png"
    assert settings["file"] == "/tmp/chart.png"
