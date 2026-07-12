from argparse import Namespace
from dataclasses import dataclass
from typing import Literal

DiagnosticChannel = Literal["live", "stderr", "none"]


@dataclass(frozen=True, slots=True)
class CliStreamDisplayConfig:
    """Represent normalized CLI streaming display options."""

    quiet: bool
    stats: bool
    display_tools: bool
    display_events: bool
    display_tools_events: int | None
    record: bool
    interactive: bool
    refresh_per_second: int
    answer_height: int
    answer_height_expand: bool
    display_tokens: int
    display_pause: int
    display_probabilities: bool
    display_probabilities_maximum: float
    display_probabilities_sample_minimum: float
    display_time_to_n_token: int | None
    display_reasoning_time: bool
    display_reasoning: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.quiet, bool)
        assert isinstance(self.stats, bool)
        assert isinstance(self.display_tools, bool)
        assert isinstance(self.display_events, bool)
        assert self.display_tools_events is None or (
            isinstance(self.display_tools_events, int)
            and self.display_tools_events >= 0
        )
        assert isinstance(self.record, bool)
        assert isinstance(self.interactive, bool)
        assert isinstance(self.refresh_per_second, int)
        assert self.refresh_per_second > 0
        assert isinstance(self.answer_height, int)
        assert self.answer_height >= 0
        assert isinstance(self.answer_height_expand, bool)
        assert isinstance(self.display_tokens, int)
        assert self.display_tokens >= 0
        assert isinstance(self.display_pause, int)
        assert self.display_pause >= 0
        assert isinstance(self.display_probabilities, bool)
        assert isinstance(self.display_probabilities_maximum, float)
        assert isinstance(self.display_probabilities_sample_minimum, float)
        assert self.display_time_to_n_token is None or (
            isinstance(self.display_time_to_n_token, int)
            and self.display_time_to_n_token > 0
        )
        assert isinstance(self.display_reasoning_time, bool)
        assert isinstance(self.display_reasoning, bool)

    @property
    def diagnostic_channel(self) -> DiagnosticChannel:
        """Return where optional streaming diagnostics should render."""
        if self.quiet:
            return "none"
        if (
            self.stats
            or self.display_tools
            or self.display_events
            or self.display_reasoning
        ):
            return "live" if self.interactive else "stderr"
        return "none"

    @property
    def show_reasoning(self) -> bool:
        """Return whether reasoning text should render."""
        return self.display_reasoning and self.diagnostic_channel != "none"

    @property
    def show_stats(self) -> bool:
        """Return whether token statistics should render."""
        return self.stats and self.diagnostic_channel != "none"

    @property
    def show_tools(self) -> bool:
        """Return whether tool diagnostics should render."""
        return self.display_tools and self.diagnostic_channel != "none"

    @property
    def show_events(self) -> bool:
        """Return whether non-tool events should render."""
        return self.display_events and self.diagnostic_channel != "none"

    @property
    def show_token_details(self) -> bool:
        """Return whether per-token detail panels should render."""
        return (
            self.show_stats and self.live_enabled and self.display_tokens > 0
        )

    @property
    def show_probabilities(self) -> bool:
        """Return whether token probability panels should render."""
        return self.show_token_details and self.display_probabilities

    @property
    def show_timing(self) -> bool:
        """Return whether timing details should render."""
        return self.show_stats and (
            self.display_reasoning_time
            or self.display_time_to_n_token is not None
        )

    @property
    def live_enabled(self) -> bool:
        """Return whether live diagnostic rendering is enabled."""
        return self.diagnostic_channel == "live"

    @property
    def record_enabled(self) -> bool:
        """Return whether live frames should be saved to SVG."""
        return self.record and self.live_enabled

    @property
    def answer_stdout_only(self) -> bool:
        """Return whether stdout should contain only answer text."""
        return self.quiet or not self.interactive


def cli_stream_display_config(
    args: Namespace,
    *,
    refresh_per_second: int,
    interactive: bool = True,
) -> CliStreamDisplayConfig:
    """Return normalized stream display configuration from parsed args."""
    assert isinstance(args, Namespace)
    assert isinstance(refresh_per_second, int)
    assert refresh_per_second > 0
    assert isinstance(interactive, bool)

    quiet = bool(getattr(args, "quiet", False))
    if quiet:
        return CliStreamDisplayConfig(
            quiet=True,
            stats=False,
            display_tools=False,
            display_events=False,
            display_tools_events=0,
            record=False,
            interactive=interactive,
            refresh_per_second=refresh_per_second,
            answer_height=0,
            answer_height_expand=False,
            display_tokens=0,
            display_pause=0,
            display_probabilities=False,
            display_probabilities_maximum=float(
                getattr(args, "display_probabilities_maximum", 0.0)
            ),
            display_probabilities_sample_minimum=float(
                getattr(args, "display_probabilities_sample_minimum", 0.0)
            ),
            display_time_to_n_token=None,
            display_reasoning_time=False,
        )

    display_tools_events = getattr(args, "display_tools_events", None)
    if display_tools_events is not None:
        display_tools_events = int(display_tools_events)

    display_tokens = getattr(args, "display_tokens", None) or 0
    display_pause = getattr(args, "display_pause", None) or 0
    display_time_to_n_token = getattr(args, "display_time_to_n_token", None)
    if display_time_to_n_token is not None:
        display_time_to_n_token = int(display_time_to_n_token)

    return CliStreamDisplayConfig(
        quiet=False,
        stats=bool(getattr(args, "stats", False)),
        display_tools=bool(getattr(args, "display_tools", False)),
        display_events=bool(getattr(args, "display_events", False)),
        display_tools_events=display_tools_events,
        record=bool(getattr(args, "record", False)),
        interactive=interactive,
        refresh_per_second=refresh_per_second,
        answer_height=int(getattr(args, "display_answer_height", 12)),
        answer_height_expand=bool(
            getattr(args, "display_answer_height_expand", False)
        ),
        display_tokens=int(display_tokens),
        display_pause=int(display_pause),
        display_probabilities=bool(
            getattr(args, "display_probabilities", False)
        ),
        display_probabilities_maximum=float(
            getattr(args, "display_probabilities_maximum", 0.8)
        ),
        display_probabilities_sample_minimum=float(
            getattr(args, "display_probabilities_sample_minimum", 0.1)
        ),
        display_time_to_n_token=display_time_to_n_token,
        display_reasoning_time=not bool(
            getattr(args, "skip_display_reasoning_time", False)
        ),
        display_reasoning=bool(getattr(args, "display_reasoning", False)),
    )
