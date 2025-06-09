from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..flow.flow import Flow


class Node:
    def __init__(
        self,
        name: str,
        label: str | None = None,
        shape: str | None = None,
        input_schema: type | None = None,
        output_schema: type | None = None,
        func: Callable[..., Any] | None = None,
        subgraph: Flow | None = None,
    ) -> None:
        self.name: str = name
        self.label: str = label or name
        self.shape: str | None = shape
        self.input_schema: type | None = input_schema
        self.output_schema: type | None = output_schema
        self.func: Callable[..., Any] | None = func
        self.subgraph: Flow | None = subgraph

    def execute(self, inputs: dict[str, Any]) -> Any:
        # Delegate to subgraph if present
        if self.subgraph is not None:
            initial = (
                next(iter(inputs.values())) if len(inputs) == 1 else inputs
            )
            result = self.subgraph.execute(
                initial_node=None, initial_data=initial
            )
            if self.output_schema and not isinstance(
                result, self.output_schema
            ):
                raise TypeError(
                    f"{self.name} output {result!r} not {self.output_schema}"
                )
            return result

        # Validate input schema
        if self.input_schema:
            if isinstance(self.input_schema, type):
                if isinstance(inputs, dict) and len(inputs) == 1:
                    val = next(iter(inputs.values()))
                    if not isinstance(val, self.input_schema):
                        raise TypeError(
                            f"{self.name} input {val!r} not"
                            f" {self.input_schema}"
                        )
                elif not isinstance(inputs, self.input_schema):
                    raise TypeError(
                        f"{self.name} input {inputs!r} not {self.input_schema}"
                    )

        # Compute output
        if callable(self.func):
            try:
                output = self.func(inputs)
            except TypeError:
                output = self.func(*inputs.values())  # type: ignore
        else:
            if not inputs:
                output = None
            elif len(inputs) == 1:
                output = next(iter(inputs.values()))
            else:
                output = inputs

        # Validate output schema
        if self.output_schema and output is not None:
            if not isinstance(output, self.output_schema):
                raise TypeError(
                    f"{self.name} output {output!r} not {self.output_schema}"
                )

        return output

    def __repr__(self) -> str:
        return f"<Node {self.name}>"
