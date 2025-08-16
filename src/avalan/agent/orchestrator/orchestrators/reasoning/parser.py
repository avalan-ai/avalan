from dataclasses import dataclass
from re import DOTALL, IGNORECASE, escape, search


@dataclass(slots=True)
class ReasoningOutputParser:
    """Parse reasoning and answers from text responses.

    The parser looks for reasoning enclosed in a tag, followed by an answer
    prefixed by a specific keyword. Both the tag name and the prefixes are
    configurable.
    """

    reasoning_tag: str | None = "think"
    reasoning_prefixes: list[str] | None = None
    answer_prefix: str = "answer"

    def __post_init__(self) -> None:
        if self.reasoning_prefixes is None:
            self.reasoning_prefixes = ["reasoning", "thought"]

    def parse(self, text: str) -> tuple[str | None, str]:
        """Extract reasoning and answer from *text*.

        Args:
            text: Full model response.

        Returns:
            Tuple with optional reasoning and the final answer.
        """
        if self.reasoning_tag:
            tag = escape(self.reasoning_tag)
            match = search(rf"<{tag}>(.*?)</{tag}>", text, DOTALL | IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                remaining = text[match.end() :].strip()
                ans_match = search(
                    self._answer_regex(), remaining, DOTALL | IGNORECASE
                )
                answer = ans_match.group(1).strip() if ans_match else remaining
                return reasoning, answer

        if self.reasoning_prefixes:
            prefixes = "|".join(escape(p) for p in self.reasoning_prefixes)
            prefix = search(rf"^(?:{prefixes}):", text, IGNORECASE)
            if prefix:
                after = text[prefix.end() :].strip()
                ans_match = search(
                    self._answer_regex(), after, DOTALL | IGNORECASE
                )
                if ans_match:
                    reasoning = after[: ans_match.start()].strip()
                    answer = ans_match.group(1).strip()
                else:
                    reasoning = after
                    answer = reasoning
                return reasoning or None, answer

        ans_match = search(self._answer_regex(), text, DOTALL | IGNORECASE)
        if ans_match:
            answer = ans_match.group(1).strip()
            reasoning = text[: ans_match.start()].strip() or None
            return reasoning, answer

        return None, text.strip()

    def _answer_regex(self) -> str:
        prefix = escape(self.answer_prefix)
        return rf"{prefix}:\s*(.*)"
