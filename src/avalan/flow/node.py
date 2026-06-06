from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..flow.flow import Flow

CancellationChecker = Callable[[], Awaitable[None]]


class Node:
    def __init__(
        self,
        name: str,
        label: str | None = None,
        shape: str | None = None,
        input_schema: type | None = None,
        output_schema: type | None = None,
        func: Callable[..., Any] | None = None,
        subgraph: "Flow | None" = None,
        async_only: bool = False,
        receives_cancellation_checker: bool = False,
    ) -> None:
        assert isinstance(async_only, bool)
        assert isinstance(receives_cancellation_checker, bool)
        self.name: str = name
        self.label: str = label or name
        self.shape: str | None = shape
        self.input_schema: type | None = input_schema
        self.output_schema: type | None = output_schema
        self.func: Callable[..., Any] | None = func
        self.subgraph: "Flow | None" = subgraph
        self.async_only: bool = async_only
        self.receives_cancellation_checker: bool = (
            receives_cancellation_checker
        )

    def execute(self, inputs: dict[str, Any]) -> Any:
        if self.subgraph is not None:
            initial = self._subgraph_initial(inputs)
            result = self.subgraph.execute(
                initial_node=None, initial_data=initial
            )
            self._validate_output(result)
            return result

        self._validate_input(inputs)
        output = self._compute_output(inputs)
        if isawaitable(output):
            close = getattr(output, "close", None)
            if callable(close):
                close()
            raise TypeError(
                f"{self.name} produced awaitable output; use execute_async"
            )
        self._validate_output(output)

        return output

    async def execute_async(
        self,
        inputs: dict[str, Any],
        *,
        cancellation_checker: CancellationChecker | None = None,
    ) -> Any:
        await _check_cancelled(cancellation_checker)
        if self.subgraph is not None:
            result = await self.subgraph.execute_async(
                initial_node=None,
                initial_data=self._subgraph_initial(inputs),
                cancellation_checker=cancellation_checker,
            )
            self._validate_output(result)
            await _check_cancelled(cancellation_checker)
            return result

        self._validate_input(inputs)
        await _check_cancelled(cancellation_checker)
        output = self._compute_output_async(
            inputs,
            cancellation_checker=cancellation_checker,
        )
        if isawaitable(output):
            output = await output
        self._validate_output(output)
        await _check_cancelled(cancellation_checker)
        return output

    def _compute_output(self, inputs: dict[str, Any]) -> Any:
        if not callable(self.func):
            if not inputs:
                return None
            if len(inputs) == 1:
                return next(iter(inputs.values()))
            return inputs
        try:
            return self.func(inputs)
        except TypeError:
            return self.func(*inputs.values())

    def _compute_output_async(
        self,
        inputs: dict[str, Any],
        *,
        cancellation_checker: CancellationChecker | None,
    ) -> Any:
        if callable(self.func) and self.receives_cancellation_checker:
            return self.func(
                inputs,
                cancellation_checker=cancellation_checker,
            )
        return self._compute_output(inputs)

    def _subgraph_initial(self, inputs: dict[str, Any]) -> Any:
        return next(iter(inputs.values())) if len(inputs) == 1 else inputs

    def _validate_input(self, inputs: dict[str, Any]) -> None:
        if not self.input_schema:
            return
        if isinstance(self.input_schema, type):
            if isinstance(inputs, dict) and len(inputs) == 1:
                val = next(iter(inputs.values()))
                if not isinstance(val, self.input_schema):
                    raise TypeError(
                        f"{self.name} input {val!r} not {self.input_schema}"
                    )
            elif not isinstance(inputs, self.input_schema):
                raise TypeError(
                    f"{self.name} input {inputs!r} not {self.input_schema}"
                )

    def _validate_output(self, output: Any) -> None:
        if self.output_schema and output is not None:
            if not isinstance(output, self.output_schema):
                raise TypeError(
                    f"{self.name} output {output!r} not {self.output_schema}"
                )

    def __repr__(self) -> str:
        return f"<Node {self.name}>"


async def _check_cancelled(
    cancellation_checker: CancellationChecker | None,
) -> None:
    if cancellation_checker is not None:
        await cancellation_checker()
