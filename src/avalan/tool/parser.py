from ast import literal_eval
from ..entities import ToolCall, ToolFormat
from json import JSONDecodeError, loads
from re import DOTALL, search
from typing import Any, Optional, Tuple
from uuid import uuid4
from xml.etree import ElementTree


class ToolCallParser:
    _eos_token: Optional[str]
    _tool_format: Optional[ToolFormat]

    def __init__(
        self,
        tool_format: Optional[ToolFormat] = None,
        eos_token: Optional[str] = None,
    ) -> None:
        self._tool_format = tool_format
        self._eos_token = eos_token

    def __call__(self, text: str) -> Optional[list[ToolCall]]:
        calls = (
            self._parse_json(text)
            if self._tool_format is ToolFormat.JSON
            else self._parse_react(text)
            if self._tool_format is ToolFormat.REACT
            else self._parse_bracket(text)
            if self._tool_format is ToolFormat.BRACKET
            else self._parse_openai_json(text)
            if self._tool_format is ToolFormat.OPENAI
            else None
        )
        if not calls:
            calls = self._parse_tag(text)
        return calls

    def set_eos_token(self, eos_token: str) -> None:
        self._eos_token = eos_token

    def _parse_json(self, text: str) -> Optional[Tuple[str, dict[str, Any]]]:
        try:
            payload = loads(text)
            return payload["tool"], payload.get("arguments", {})
        except Exception:
            return None

    def _parse_react(self, text: str) -> Optional[Tuple[str, dict[str, Any]]]:
        act = search(r"Action:\s*(\w+)", text)
        inp = search(r"Action Input:\s*({.*})", text, DOTALL)
        if act and inp:
            try:
                return act.group(1), loads(inp.group(1))
            except JSONDecodeError:
                pass
        return None

    def _parse_bracket(self, text: str) -> Optional[Tuple[str, dict[str, Any]]]:
        m = search(r"\[(\w+)\]\(([^)]+)\)", text)
        if m:
            return m.group(1), {"input": m.group(2)}
        return None

    def _parse_openai_json(
        self, text: str
    ) -> Optional[Tuple[str, dict[str, Any]]]:
        try:
            payload = loads(text)
            name = payload.get("name")
            args = payload.get("arguments", {})
            if isinstance(name, str) and isinstance(args, dict):
                return name, args
        except JSONDecodeError:
            pass
        return None

    def _parse_tag(self, text: str) -> Optional[Tuple[str, dict[str, Any]]]:
        tool_calls: list[ToolCall] = []

        if self._eos_token:
            text = text.strip().removesuffix(self._eos_token)
        try:
            root = ElementTree.fromstring(f"<root>{text}</root>")
            for element in root.findall(".//tool_call"):
                tool_call = None
                try:
                    json_text = element.text.strip()

                    try:
                        tool_call = loads(json_text)
                    except JSONDecodeError:
                        try:
                            tool_call = literal_eval(json_text)
                        except (SyntaxError, ValueError):
                            continue
                except Exception:
                    pass

                if (
                    tool_call is not None
                    and "name" in tool_call
                    and "arguments" in tool_call
                ):
                    tool_calls.append(
                        ToolCall(
                            id=uuid4(),
                            name=tool_call["name"],
                            arguments=tool_call["arguments"],
                        )
                    )
        except ElementTree.ParseError:
            pass

        if tool_calls:
            return tool_calls

        m = search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=DOTALL
        )
        if m:
            tool_call_payload = m.group(1)
            try:
                tool_call = loads(tool_call_payload)
                if (
                    tool_call
                    and "name" in tool_call
                    and "arguments" in tool_call
                ):
                    tool_calls.append(
                        ToolCall(
                            id=uuid4(),
                            name=tool_call["name"],
                            arguments=tool_call["arguments"],
                        )
                    )
            except JSONDecodeError:
                pass

        m = search(
            r"<tool_call\s+name=\"([^\"]+)\"\s*>(\{.*?\})</tool_call>",
            text,
            DOTALL,
        )
        if m:
            try:
                tool_calls.append(
                    ToolCall(
                        id=uuid4(),
                        name=m.group(1),
                        arguments=loads(m.group(2))
                    )
                )
            except JSONDecodeError:
                pass

        m = search(
            r"<tool\s+name=\"([^\"]+)\"\s*>(\{.*?\})</tool>",
            text,
            DOTALL,
        )
        if m:
            try:
                tool_calls.append(
                    ToolCall(
                        id=uuid4(),
                        name=m.group(1),
                        arguments=loads(m.group(2))
                    )
                )
            except JSONDecodeError:
                pass

        m = search(
            r"<tool_call\s+name=\"([^\"]+)\"\s+arguments='(\{.*?\})'\s*/>",
            text,
        )
        if m:
            try:
                tool_calls.append(
                    ToolCall(
                        id=uuid4(),
                        name=m.group(1),
                        arguments=loads(m.group(2))
                    )
                )
            except JSONDecodeError:
                pass

        return tool_calls if tool_calls else None
