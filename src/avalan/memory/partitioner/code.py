from ...memory.partitioner import Encoding, PartitionerException
from ...utils import _j
from dataclasses import dataclass
from logging import Logger
from tree_sitter_python import language as python
from tree_sitter import Language, Node, Parser
from typing import Literal, Optional

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


@dataclass(frozen=True, kw_only=True)
class Parameter:
    parameter_type: ParameterType
    name: str
    type: Optional[str]


@dataclass(frozen=True, kw_only=True)
class Function:
    id: str
    namespace: Optional[str]
    class_name: Optional[str]
    name: str
    parameters: Optional[list[Parameter]]
    return_type: Optional[str]


@dataclass(frozen=True, kw_only=True)
class Symbol:
    symbol_type: SymbolType
    id: str


@dataclass(frozen=True, kw_only=True)
class CodePartition:
    data: str
    encoding: Encoding
    symbols: list[Symbol]


class CodePartitioner:
    _logger: Logger
    _parsers: dict[LanguageName, Parser] = {}

    def __init__(self, logger: Logger):
        self._logger = logger

    def partition(
        self,
        language_name: LanguageName,
        input: str,
        encoding: Encoding,
        max_chars: int,
        namespace: Optional[str] = None,
    ) -> tuple[list[CodePartition], Optional[list[Function]]]:
        parser, language = self._get_parser(language_name)
        tree = parser.parse(input.encode(encoding), encoding=encoding)
        root_node = tree.root_node

        self._logger.debug(f"Parsing {language_name} code for functions")
        functions = self._get_functions(namespace, root_node, encoding)

        if root_node.children and root_node.children[0].is_error:
            error_node = root_node.children[0]
            error_name = error_node.grammar_name
            error_message = error_node.text.decode(encoding)
            error_row, error_column = error_node.start_point
            raise PartitionerException(
                f'{error_name}: "{error_message}" at {error_row},{error_column}'
            )

        self._logger.debug(f"Partitioning {language_name} code")
        partitions = self._partition(
            input, encoding, max_chars, root_node, current_namespace=namespace
        )
        self._logger.debug(
            f"Partitioned {language_name} into "
            f"{len(partitions)} partitions and "
            f"{len(functions)} functions"
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
        self._parsers[language_name] = parser
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
        current_namespace: Optional[str] = None,
        current_class_name: Optional[str] = None,
        current_symbols: Optional[list[Symbol]] = None,
    ) -> list[CodePartition]:
        if current_symbols is None:
            current_symbols = []

        chunk, chunks = "", []

        for child_node in node.children:
            if child_node.type == "class_definition":
                class_name_node = child_node.child_by_field_name("name")
                current_class_name = (
                    class_name_node.text.decode(encoding)
                    if class_name_node
                    else None
                )
                class_id = _j(".", [current_namespace, current_class_name])
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
        current_namespace: Optional[str],
        node: Node,
        encoding: Encoding,
        current_class_name: Optional[str] = None,
    ) -> list[Function]:
        assert node and encoding
        results = []
        if node.type == "class_definition":
            class_name_node = node.child_by_field_name("name")
            class_name = (
                class_name_node.text.decode(encoding)
                if class_name_node
                else None
            )
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
        current_namespace: Optional[str],
        current_class_name: Optional[str],
        node: Node,
        encoding: Encoding,
    ) -> Function:
        assert node.type in ("function_definition", "async_function_definition")
        function_id, function_name = cls._get_function_id_and_name_from_node(
            current_namespace, current_class_name, node, encoding
        )
        params_node = node.child_by_field_name("parameters")
        return_type_node = node.child_by_field_name("return_type")
        return Function(
            id=function_id,
            namespace=current_namespace,
            class_name=current_class_name,
            name=function_name,
            parameters=cls._get_parameters(params_node, encoding)
            if params_node
            else None,
            return_type=return_type_node.text.decode(encoding)
            if return_type_node
            else None,
        )

    @staticmethod
    def _get_function_id_and_name_from_node(
        current_namespace: Optional[str],
        current_class_name: Optional[str],
        node: Node,
        encoding: Encoding,
    ) -> tuple[str, str]:
        assert node.type in ("function_definition", "async_function_definition")
        function_name_node = node.child_by_field_name("name")
        assert function_name_node
        function_name = function_name_node.text.decode(encoding)
        assert function_name
        function_id = _j(
            ".", [current_namespace, current_class_name, function_name]
        )
        return function_id, function_name

    @staticmethod
    def _get_parameters(
        node: Node, encoding: Encoding
    ) -> list[dict[str, Optional[str]]]:
        assert node
        parameters = []
        for child in node.children:
            if child.type in (
                "default_parameter",
                "typed_default_parameter",
                "typed_parameter",
            ):
                param_type = child.child_by_field_name("type").text.decode(
                    encoding
                )
                param_name = None
                if child.named_child_count:
                    for parameter_child in child.children:
                        if parameter_child.type == "identifier":
                            param_name = parameter_child.text.decode(encoding)
                            break
                parameters.append(
                    Parameter(
                        parameter_type=child.type,
                        name=param_name,
                        type=param_type,
                    )
                )
            elif child.type in (
                "dictionary_splat_pattern",
                "identifier",
                "keyword_separator",
            ):
                parameters.append(
                    Parameter(
                        parameter_type=child.type,
                        name=child.text.decode(encoding),
                        type=None,
                    )
                )

        return parameters
