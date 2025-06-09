from ..flow.connection import Connection
from ..flow.node import Node
from typing import Any, Callable
from re import match


class Flow:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.connections: list[Connection] = []
        self.outgoing: dict[str, list[Connection]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node
        self.outgoing.setdefault(node.name, [])

    def add_connection(
        self,
        src_name: str,
        dest_name: str,
        label: str | None = None,
        conditions: list[Callable[[Any], bool]] | None = None,
        filters: list[Callable[[Any], Any]] | None = None,
    ) -> None:
        if src_name not in self.nodes or dest_name not in self.nodes:
            raise KeyError(f"Unknown node: {src_name} or {dest_name}")
        src = self.nodes[src_name]
        dest = self.nodes[dest_name]
        conn = Connection(src, dest, label, conditions, filters)
        self.connections.append(conn)
        self.outgoing[src_name].append(conn)

    def parse_mermaid(self, mermaid: str) -> None:
        lines = [line.strip() for line in mermaid.splitlines() if line.strip()]
        if lines and lines[0].lower().startswith("graph"):
            lines = lines[1:]
        for line in lines:
            if "-->" not in line:
                continue
            left, right = line.split("-->", 1)
            left = left.strip()
            right = right.strip()
            # strip edge label syntax: A -- text --> B
            label = None
            if "--" in left:
                parts = left.split("--", 1)
                left, label = parts[0].strip(), parts[1].strip()
            src_id, src_lbl, src_shape = self._parse_node(left)
            dst_id, dst_lbl, dst_shape = self._parse_node(right)
            for nid, nlbl, nshape in [
                (src_id, src_lbl, src_shape),
                (dst_id, dst_lbl, dst_shape),
            ]:
                if nid not in self.nodes:
                    self.add_node(Node(nid, label=nlbl or nid, shape=nshape))
                else:
                    node = self.nodes[nid]
                    if nlbl and node.label == node.name:
                        node.label = nlbl
                    if nshape and not node.shape:
                        node.shape = nshape
            self.add_connection(src_id, dst_id, label=label)

    def _parse_node(self, text: str) -> tuple[str, str | None, str | None]:
        m = match(r"^([A-Za-z0-9_]+)", text)
        if not m:
            return text, None, None
        nid = m.group(1)
        rem = text[len(nid) :].strip()
        if not rem:
            return nid, None, None
        shape = None
        label = None
        if rem.startswith("[") and rem.endswith("]"):
            shape = "rect"
            label = rem[1:-1]
        elif rem.startswith("(((") and rem.endswith(")))"):
            shape = "circle"
            label = rem[3:-3]
        elif rem.startswith("(") and rem.endswith(")"):
            shape = "roundrect"
            label = rem[1:-1]
        elif rem.startswith("{") and rem.endswith("}"):
            shape = "diamond"
            label = rem[1:-1]
        else:
            label = rem
        return nid, label, shape

    def execute(
        self,
        initial_node: str | Node | None = None,
        initial_data: Any = None,
    ) -> Any:
        # Determine start nodes
        if initial_node is None:
            indegree = {n: 0 for n in self.nodes}
            for c in self.connections:
                indegree[c.dest.name] += 1
            start = [self.nodes[n] for n, d in indegree.items() if d == 0]
        else:
            start = (
                [self.nodes[initial_node]]
                if isinstance(initial_node, str)
                else [initial_node]
            )  # type: ignore
        indegree_map = {n: 0 for n in self.nodes}
        for c in self.connections:
            indegree_map[c.dest.name] += 1

        buffers: dict[str, dict[str, Any]] = {n: {} for n in self.nodes}
        if initial_data is not None and len(start) == 1:
            buffers[start[0].name] = {"__init__": initial_data}

        ready: list[Node] = []
        for n in start:
            indegree_map[n.name] = 0
            ready.append(n)

        outputs: dict[str, Any] = {}
        while ready:
            node = ready.pop(0)
            inp = buffers[node.name]
            incoming = sum(
                1 for c in self.connections if c.dest.name == node.name
            )
            if incoming > 0 and not inp:
                outputs[node.name] = None
            else:
                outputs[node.name] = node.execute(inp)
            outval = outputs[node.name]
            if outval is not None:
                for c in self.outgoing.get(node.name, []):
                    if c.check_conditions(outval):
                        forwarded = c.apply_filters(outval)
                        buffers[c.dest.name][node.name] = forwarded
            for c in self.outgoing.get(node.name, []):
                indegree_map[c.dest.name] -= 1
                if indegree_map[c.dest.name] == 0:
                    ready.append(c.dest)

        terminal = {
            n: outputs[n] for n, outs in self.outgoing.items() if not outs
        }
        if len(terminal) == 1:
            return next(iter(terminal.values()))
        return terminal
