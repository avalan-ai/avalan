from ..flow.connection import Connection
from ..flow.node import Node

import re
from collections import deque
from typing import Any, Callable


class Flow:
    """Directed graph of nodes and connections."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.connections: list[Connection] = []
        self.outgoing: dict[str, list[Connection]] = {}
        self.incoming: dict[str, list[Connection]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node
        self.outgoing.setdefault(node.name, [])
        self.incoming.setdefault(node.name, [])

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
        self.incoming.setdefault(dest_name, []).append(conn)

    def parse_mermaid(self, mermaid: str) -> None:
        """Populate the flow from a Mermaid diagram.

        Edge labels are accepted in both ``A -- label --> B`` and
        ``A -->|label| B`` forms. Lines without edges are ignored.
        """
        for src_text, label, dest_text in self._iter_mermaid_edges(mermaid):
            src_id, src_label, src_shape = self._parse_node(src_text)
            dest_id, dest_label, dest_shape = self._parse_node(dest_text)
            src_node = self._ensure_node(src_id, src_label, src_shape)
            dest_node = self._ensure_node(dest_id, dest_label, dest_shape)
            self.add_connection(src_node.name, dest_node.name, label=label)

    def _parse_node(self, text: str) -> tuple[str, str | None, str | None]:
        match = re.match(r"^([A-Za-z0-9_]+)", text)
        if not match:
            return text, None, None
        node_id = match.group(1)
        remainder = text[len(node_id) :].strip()
        if not remainder:
            return node_id, None, None
        shape = None
        label = None
        if remainder.startswith("[") and remainder.endswith("]"):
            shape = "rect"
            label = remainder[1:-1]
        elif remainder.startswith("(((") and remainder.endswith(")))"):
            shape = "circle"
            label = remainder[3:-3]
        elif remainder.startswith("(") and remainder.endswith(")"):
            shape = "roundrect"
            label = remainder[1:-1]
        elif remainder.startswith("{") and remainder.endswith("}"):
            shape = "diamond"
            label = remainder[1:-1]
        else:
            label = remainder
        return node_id, label, shape

    def _ensure_node(
        self, name: str, label: str | None, shape: str | None
    ) -> Node:
        node = self.nodes.get(name)
        if node is None:
            node = Node(name, label=label or name, shape=shape)
            self.add_node(node)
            return node
        if label and node.label == node.name:
            node.label = label
        if shape and not node.shape:
            node.shape = shape
        return node

    def _iter_mermaid_edges(
        self, mermaid: str
    ) -> list[tuple[str, str | None, str]]:
        lines = [line.strip() for line in mermaid.splitlines() if line.strip()]
        if lines and lines[0].lower().startswith("graph"):
            lines = lines[1:]
        edges: list[tuple[str, str | None, str]] = []
        for line in lines:
            if "-->" not in line:
                continue
            left, right = [part.strip() for part in line.split("-->", 1)]
            label: str | None = None
            if "--" in left:
                src_text, label_text = left.split("--", 1)
                left = src_text.strip()
                label = label_text.strip()
            elif right.startswith("|") and "|" in right[1:]:
                label_text, dest_text = right[1:].split("|", 1)
                label = label_text.strip()
                right = dest_text.strip()
            edges.append((left, label, right))
        return edges

    def execute(
        self,
        initial_node: str | Node | None = None,
        initial_data: Any = None,
    ) -> Any:
        start_nodes = self._resolve_start_nodes(initial_node)
        if not start_nodes:
            raise ValueError(
                "Flow has no valid starting node; graph may contain a cycle"
            )

        indegree = {
            name: len(self.incoming.get(name, [])) for name in self.nodes
        }
        buffers: dict[str, dict[str, Any]] = {name: {} for name in self.nodes}
        if initial_data is not None and len(start_nodes) == 1:
            buffers[start_nodes[0].name] = {"__init__": initial_data}

        queue: deque[Node] = deque()
        for node in start_nodes:
            indegree[node.name] = 0
            queue.append(node)

        outputs: dict[str, Any] = {}
        processed: set[str] = set()
        reachable = self._collect_reachable(start_nodes)
        while queue:
            node = queue.popleft()
            processed.add(node.name)
            inputs = buffers[node.name]
            if indegree[node.name] > 0 and not inputs:
                outputs[node.name] = None
            else:
                outputs[node.name] = node.execute(inputs)
            out_value = outputs[node.name]
            for connection in self.outgoing.get(node.name, []):
                indegree[connection.dest.name] -= 1
                if out_value is not None and connection.check_conditions(
                    out_value
                ):
                    forwarded = connection.apply_filters(out_value)
                    buffers[connection.dest.name][node.name] = forwarded
                if indegree[connection.dest.name] == 0:
                    queue.append(connection.dest)

        if processed != reachable:
            remaining = sorted(reachable - processed)
            raise ValueError(
                "Flow contains a cycle involving: " + ", ".join(remaining)
            )

        cycle_nodes = self._detect_cycle_nodes(start_nodes)
        if cycle_nodes:
            remaining = sorted(cycle_nodes)
            raise ValueError(
                "Flow contains a cycle involving: " + ", ".join(remaining)
            )

        terminal = {
            name: outputs[name]
            for name, outs in self.outgoing.items()
            if not outs
        }
        if len(terminal) == 1:
            return next(iter(terminal.values()))
        return terminal

    def _resolve_start_nodes(
        self, initial_node: str | Node | None
    ) -> list[Node]:
        if initial_node is None:
            return [
                self.nodes[name]
                for name, inbound in self.incoming.items()
                if not inbound
            ]
        if isinstance(initial_node, Node):
            if initial_node.name not in self.nodes:
                raise KeyError(f"Unknown node: {initial_node.name}")
            return [self.nodes[initial_node.name]]
        if initial_node not in self.nodes:
            raise KeyError(f"Unknown node: {initial_node}")
        return [self.nodes[initial_node]]

    def _collect_reachable(self, start_nodes: list[Node]) -> set[str]:
        reachable: set[str] = set()
        stack = [node.name for node in start_nodes]
        while stack:
            name = stack.pop()
            if name in reachable:
                continue
            reachable.add(name)
            for connection in self.outgoing.get(name, []):
                stack.append(connection.dest.name)
        return reachable

    def _detect_cycle_nodes(self, start_nodes: list[Node]) -> set[str]:
        visited: set[str] = set()
        recursion_stack: set[str] = set()
        cycle_nodes: set[str] = set()

        def dfs(name: str) -> bool:
            if name in recursion_stack:
                cycle_nodes.add(name)
                return True
            if name in visited:
                return False
            visited.add(name)
            recursion_stack.add(name)
            found_cycle = False
            for connection in self.outgoing.get(name, []):
                if dfs(connection.dest.name):
                    cycle_nodes.add(name)
                    found_cycle = True
            recursion_stack.remove(name)
            return found_cycle

        for node in start_nodes:
            dfs(node.name)
        return cycle_nodes
