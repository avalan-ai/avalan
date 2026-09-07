from pathlib import Path


def write_application(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text("""[task]
name="scheduled"
version="1"
[input]
type="string"
[output]
type="text"
[execution]
type="agent"
ref="agent.toml"
[run]
mode="queue"
queue="default"
idempotency="none"
[privacy]
input="encrypt"
files="drop"
raw_retention_days=1
""")
    (root / "agent.toml").write_text("""[agent]
name="old"
instructions="old retained instructions"
[engine]
uri="ai://env:KEY@openai/gpt-4o-mini"
[tool]
enable=[]
""")


def write_flow_application(root: Path) -> None:
    write_application(root)
    task = root / "task.toml"
    task.write_text(
        task.read_text()
        .replace('type="string"', 'type="object"\nschema={type="object"}')
        .replace('type="agent"', 'type="flow"')
        .replace('ref="agent.toml"', 'ref="flow.toml"')
    )
    (root / "flow.toml").write_text("""[flow]
name="scheduled-flow"
version="1"
[[inputs]]
name="scheduled"
type="string"
[[outputs]]
name="answer"
type="json"
[entry]
type="node"
node="echo"
[output_behavior]
type="map"
[output_behavior.outputs]
answer="echo.value"
[nodes.echo]
type="pass-through"
[nodes.echo.mapping.value]
type="select"
source="input.scheduled"
""")
