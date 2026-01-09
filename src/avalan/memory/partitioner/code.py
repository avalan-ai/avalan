from ...memory.partitioner import Encoding, PartitionerException

from dataclasses import dataclass
from logging import Logger
from typing import Literal

from tree_sitter import Language, Node, Parser
from tree_sitter_python import language as python

LanguageName = Literal["python"]

SymbolType = Literal[
    "class",
    "function",
]

ParameterType = Literal[
    "default_parameter",
    "dictionary_splat_pattern",
    "identifier",
    "keyword_separator",
    "typed_default_parameter",
    "typed_parameter",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class Parameter:
    parameter_type: ParameterType
    name: str
    type: str | None


@dataclass(frozen=True, kw_only=True, slots=True)
class Function:
    id: str
    namespace: str | None
    class_name: str | None
    name: str
    parameters: list[Parameter] | None
    return_type: str | None


@dataclass(frozen=True, kw_only=True, slots=True)
class Symbol:
    symbol_type: SymbolType
    id: str


@dataclass(frozen=True, kw_only=True, slots=True)
class CodePartition:
    data: str
    encoding: Encoding
    symbols: list[Symbol]


class CodePartitioner:
    _logger: Logger
    _parsers: dict[LanguageName, tuple[Parser, Language]] = {}

    def __init__(self, logger: Logger):
        self._logger = logger

    def partition(
        self,
        language_name: LanguageName,
        input: str,
        encoding: Encoding,
        max_chars: int,
        namespace: str | None = None,
    ) -> tuple[list[CodePartition], list[Function] | None]:
        parser, language = self._get_parser(language_name)
        tree = parser.parse(input.encode(encoding), encoding=encoding)
        root_node = tree.root_node

        self._logger.debug("Parsing %s code for functions", language_name)
        functions = self._get_functions(namespace, root_node, encoding)

        if root_node.children and root_node.children[0].is_error:
            error_node = root_node.children[0]
            error_name = error_node.grammar_name
            error_text = error_node.text
            assert error_text is not None, "Error node text cannot be None"
            error_message = error_text.decode(encoding)
            error_row, error_column = error_node.start_point
            raise PartitionerException(
                f'{error_name}: "{error_message}" at'
                f" {error_row},{error_column}"
            )

        self._logger.debug("Partitioning %s code", language_name)
        partitions = self._partition(
            input, encoding, max_chars, root_node, current_namespace=namespace
        )
        self._logger.debug(
            "Partitioned %s into %d partitions and %d functions",
            language_name,
            len(partitions),
            len(functions),
        )
        return partitions, functions

    def _get_parser(
        self, language_name: LanguageName
    ) -> tuple[Parser, Language]:
        if language_name in self._parsers:
            return self._parsers[language_name]

        assert language_name == "python"  # @TODO obviously expand this
        language_impl = python()
        language = Language(language_impl)
        parser = Parser(language)
        self._parsers[language_name] = (parser, language)
        return parser, language

    # Implementation inspiration from Kevin Lu / SweepAI
    @classmethod
    def _partition(
        cls,
        source: str,
        encoding: Encoding,
        max_chars: int,
        node: Node,
        last_end: int = 0,
        current_namespace: str | None = None,
        current_class_name: str | None = None,
        current_symbols: list[Symbol] | None = None,
    ) -> list[CodePartition]:
        if current_symbols is None:
            current_symbols = []

        chunk, chunks = "", []

        for child_node in node.children:
            if child_node.type == "class_definition":
                class_name_node = child_node.child_by_field_name("name")
                if class_name_node and class_name_node.text:
                    current_class_name = class_name_node.text.decode(encoding)
                else:
                    current_class_name = None
                class_parts: list[str] = []
                if current_namespace:
                    class_parts.append(current_namespace)
                if current_class_name:
                    class_parts.append(current_class_name)
                class_id = ".".join(class_parts) if class_parts else ""
                child_symbols = current_symbols + [
                    Symbol(symbol_type="class", id=class_id)
                ]
            elif child_node.type in (
                "function_definition",
                "async_function_definition",
            ):
                function_id, _ = cls._get_function_id_and_name_from_node(
                    current_namespace, current_class_name, child_node, encoding
                )
                child_symbols = current_symbols + [
                    Symbol(symbol_type="function", id=function_id)
                ]
            else:
                child_symbols = current_symbols

            start, end = child_node.start_byte, child_node.end_byte
            length = end - start
            if length > max_chars:
                if chunk:
                    chunks.append(
                        CodePartition(
                            data=chunk,
                            encoding=encoding,
                            symbols=child_symbols or current_symbols,
                        )
                    )
                child_chunks = cls._partition(
                    source,
                    encoding,
                    max_chars,
                    child_node,
                    last_end,
                    current_namespace,
                    current_class_name,
                    child_symbols,
                )
                chunks.extend(child_chunks)
                chunk, last_end = "", end
                continue
            fragment = source[last_end:end]
            if len(chunk) + length > max_chars:
                if chunk:
                    chunks.append(
                        CodePartition(
                            data=chunk,
                            encoding=encoding,
                            symbols=child_symbols or current_symbols,
                        )
                    )
                chunk = fragment
            else:
                chunk += fragment
            last_end = end
        if chunk:
            chunks.append(
                CodePartition(
                    data=chunk,
                    encoding=encoding,
                    symbols=child_symbols or current_symbols,
                )
            )
        return chunks

    @classmethod
    def _get_functions(
        cls,
        current_namespace: str | None,
        node: Node,
        encoding: Encoding,
        current_class_name: str | None = None,
    ) -> list[Function]:
        assert node and encoding
        results = []
        if node.type == "class_definition":
            class_name_node = node.child_by_field_name("name")
            if class_name_node and class_name_node.text:
                class_name = class_name_node.text.decode(encoding)
            else:
                class_name = None
            for child in node.children:
                functs = cls._get_functions(
                    current_namespace,
                    child,
                    encoding,
                    class_name,
                )
                results.extend(functs)
        elif node.type in ("function_definition", "async_function_definition"):
            funct = cls._get_function_from_node(
                current_namespace, current_class_name, node, encoding
            )
            results.append(funct)
        else:
            for child in node.children:
                functs = cls._get_functions(
                    current_namespace,
                    child,
                    encoding,
                    current_class_name,
                )
                results.extend(functs)

        return results

    @classmethod
    def _get_function_from_node(
        cls,
        current_namespace: str | None,
        current_class_name: str | None,
        node: Node,
        encoding: Encoding,
    ) -> Function:
        assert node.type in (
            "function_definition",
            "async_function_definition",
        )
        function_id, function_name = cls._get_function_id_and_name_from_node(
            current_namespace, current_class_name, node, encoding
        )
        params_node = node.child_by_field_name("parameters")
        return_type_node = node.child_by_field_name("return_type")
        return_type: str | None = None
        if return_type_node and return_type_node.text:
            return_type = return_type_node.text.decode(encoding)
        return Function(
            id=function_id,
            namespace=current_namespace,
            class_name=current_class_name,
            name=function_name,
            parameters=(
                cls._get_parameters(params_node, encoding)
                if params_node
                else None
            ),
            return_type=return_type,
        )

    @staticmethod
    def _get_function_id_and_name_from_node(
        current_namespace: str | None,
        current_class_name: str | None,
        node: Node,
        encoding: Encoding,
    ) -> tuple[str, str]:
        assert node.type in (
            "function_definition",
            "async_function_definition",
        )
        function_name_node = node.child_by_field_name("name")
        assert function_name_node is not None
        assert function_name_node.text is not None
        function_name = function_name_node.text.decode(encoding)
        assert function_name
        id_parts: list[str] = []
        if current_namespace:
            id_parts.append(current_namespace)
        if current_class_name:
            id_parts.append(current_class_name)
        id_parts.append(function_name)
        function_id = ".".join(id_parts)
        return function_id, function_name

    @staticmethod
    def _get_parameters(node: Node, encoding: Encoding) -> list[Parameter]:
        assert node
        parameters: list[Parameter] = []
        for child in node.children:
            match child.type:
                case (
                    "default_parameter"
                    | "typed_default_parameter"
                    | "typed_parameter"
                ) as param_kind:
                    type_node = child.child_by_field_name("type")
                    param_type: str | None = None
                    if type_node and type_node.text:
                        param_type = type_node.text.decode(encoding)
                    param_name: str = ""
                    if child.named_child_count:
                        for parameter_child in child.children:
                            if parameter_child.type == "identifier":
                                if parameter_child.text:
                                    param_name = parameter_child.text.decode(
                                        encoding
                                    )
                                break
                    parameters.append(
                        Parameter(
                            parameter_type=param_kind,
                            name=param_name,
                            type=param_type,
                        )
                    )
                case (
                    "dictionary_splat_pattern"
                    | "identifier"
                    | "keyword_separator"
                ) as simple_kind:
                    name: str = ""
                    if child.text:
                        name = child.text.decode(encoding)
                    parameters.append(
                        Parameter(
                            parameter_type=simple_kind,
                            name=name,
                            type=None,
                        )
                    )

        return parameters
