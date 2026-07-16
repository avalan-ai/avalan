from argparse import Namespace
from typing import Any, cast
from unittest import TestCase
from unittest.mock import MagicMock

from avalan.cli.__main__ import CLI
from avalan.cli.display import (
    CliStreamDisplayConfig,
    cli_stream_display_config,
)


def _args(**overrides: object) -> Namespace:
    values = {
        "quiet": False,
        "stats": False,
        "display_tools": False,
        "display_events": False,
        "display_tools_events": None,
        "record": False,
        "display_answer_height": 12,
        "display_answer_height_expand": False,
        "display_tokens": None,
        "display_pause": None,
        "display_probabilities": False,
        "display_probabilities_maximum": 0.8,
        "display_probabilities_sample_minimum": 0.1,
        "display_time_to_n_token": None,
        "skip_display_reasoning_time": False,
        "display_reasoning": False,
        "display_reasoning_raw": False,
    }
    values.update(overrides)
    return Namespace(**values)


class CliStreamDisplayConfigTestCase(TestCase):
    def test_matrix_from_args(self) -> None:
        cases = [
            (
                "default",
                _args(),
                True,
                {
                    "show_stats": False,
                    "show_tools": False,
                    "show_events": False,
                    "show_reasoning": False,
                    "display_tools_events": None,
                    "display_reasoning": False,
                    "display_reasoning_raw": False,
                    "diagnostic_channel": "none",
                    "answer_stdout_only": False,
                },
            ),
            (
                "display-reasoning",
                _args(display_reasoning=True),
                True,
                {
                    "display_reasoning": True,
                    "show_reasoning": True,
                    "diagnostic_channel": "live",
                    "answer_stdout_only": False,
                },
            ),
            (
                "display-reasoning-raw-without-display-reasoning",
                _args(display_reasoning_raw=True),
                True,
                {
                    "display_reasoning": False,
                    "display_reasoning_raw": True,
                    "show_reasoning": False,
                    "diagnostic_channel": "none",
                    "answer_stdout_only": False,
                },
            ),
            (
                "stats",
                _args(stats=True),
                True,
                {
                    "show_stats": True,
                    "show_tools": False,
                    "show_events": False,
                    "show_reasoning": False,
                    "diagnostic_channel": "live",
                    "answer_stdout_only": False,
                },
            ),
            (
                "display-tools",
                _args(display_tools=True),
                True,
                {
                    "show_stats": False,
                    "show_tools": True,
                    "show_events": False,
                    "diagnostic_channel": "live",
                    "answer_stdout_only": False,
                },
            ),
            (
                "display-events",
                _args(display_events=True),
                True,
                {
                    "show_stats": False,
                    "show_tools": False,
                    "show_events": True,
                    "diagnostic_channel": "live",
                    "answer_stdout_only": False,
                },
            ),
            (
                "combined",
                _args(display_tools=True, display_events=True),
                True,
                {
                    "show_stats": False,
                    "show_tools": True,
                    "show_events": True,
                    "diagnostic_channel": "live",
                    "answer_stdout_only": False,
                },
            ),
            (
                "display-tools-events-zero",
                _args(
                    display_tools=True,
                    display_events=True,
                    display_tools_events=0,
                ),
                True,
                {
                    "show_stats": False,
                    "show_tools": True,
                    "show_events": True,
                    "diagnostic_channel": "live",
                    "answer_stdout_only": False,
                },
            ),
            (
                "record",
                _args(record=True),
                True,
                {
                    "record": True,
                    "record_enabled": False,
                    "live_enabled": False,
                    "diagnostic_channel": "none",
                },
            ),
            (
                "non-tty",
                _args(
                    stats=True,
                    display_tools=True,
                    display_events=True,
                    display_tokens=4,
                    display_probabilities=True,
                ),
                False,
                {
                    "show_stats": True,
                    "show_tools": True,
                    "show_events": True,
                    "show_token_details": False,
                    "show_probabilities": False,
                    "diagnostic_channel": "stderr",
                    "live_enabled": False,
                    "record_enabled": False,
                    "answer_stdout_only": True,
                },
            ),
            (
                "quiet-all-flags",
                _args(
                    quiet=True,
                    stats=True,
                    display_tools=True,
                    display_events=True,
                    display_tools_events=3,
                    record=True,
                    display_tokens=8,
                    display_pause=100,
                    display_probabilities=True,
                    display_time_to_n_token=256,
                    display_reasoning=True,
                    display_reasoning_raw=True,
                ),
                True,
                {
                    "stats": False,
                    "display_tools": False,
                    "display_events": False,
                    "display_tools_events": 0,
                    "record": False,
                    "display_tokens": 0,
                    "display_pause": 0,
                    "display_probabilities": False,
                    "display_time_to_n_token": None,
                    "display_reasoning_time": False,
                    "display_reasoning": False,
                    "display_reasoning_raw": False,
                    "show_stats": False,
                    "show_tools": False,
                    "show_events": False,
                    "show_token_details": False,
                    "show_probabilities": False,
                    "show_timing": False,
                    "record_enabled": False,
                    "live_enabled": False,
                    "answer_stdout_only": True,
                    "diagnostic_channel": "none",
                },
            ),
        ]

        for label, args, interactive, expected in cases:
            with self.subTest(label=label):
                config = cli_stream_display_config(
                    args,
                    refresh_per_second=7,
                    interactive=interactive,
                )
                self.assertIsInstance(config, CliStreamDisplayConfig)
                self.assertEqual(config.refresh_per_second, 7)
                for attribute, value in expected.items():
                    self.assertEqual(getattr(config, attribute), value)

    def test_token_detail_probability_and_timing_flags(self) -> None:
        config = cli_stream_display_config(
            _args(
                stats=True,
                display_tokens=5,
                display_probabilities=True,
                display_probabilities_maximum=0.5,
                display_probabilities_sample_minimum=0.2,
                display_time_to_n_token=32,
                display_pause=250,
                display_answer_height=20,
                display_answer_height_expand=True,
            ),
            refresh_per_second=3,
            interactive=True,
        )

        self.assertTrue(config.show_token_details)
        self.assertTrue(config.show_probabilities)
        self.assertTrue(config.show_timing)
        self.assertEqual(config.display_probabilities_maximum, 0.5)
        self.assertEqual(config.display_probabilities_sample_minimum, 0.2)
        self.assertEqual(config.display_pause, 250)
        self.assertEqual(config.answer_height, 20)
        self.assertTrue(config.answer_height_expand)

    def test_direct_config_accepts_unbounded_tool_event_history(self) -> None:
        config = CliStreamDisplayConfig(
            quiet=False,
            stats=True,
            display_tools=True,
            display_events=True,
            display_tools_events=None,
            record=True,
            interactive=True,
            refresh_per_second=1,
            answer_height=12,
            answer_height_expand=False,
            display_tokens=1,
            display_pause=0,
            display_probabilities=False,
            display_probabilities_maximum=0.8,
            display_probabilities_sample_minimum=0.1,
            display_time_to_n_token=None,
            display_reasoning_time=True,
        )

        self.assertTrue(config.show_tools)
        self.assertTrue(config.record_enabled)
        self.assertFalse(config.show_reasoning)

    def test_noninteractive_reasoning_uses_stderr_diagnostics(self) -> None:
        config = cli_stream_display_config(
            _args(display_reasoning=True),
            refresh_per_second=3,
            interactive=False,
        )

        self.assertTrue(config.show_reasoning)
        self.assertEqual(config.diagnostic_channel, "stderr")
        self.assertTrue(config.answer_stdout_only)

    def test_direct_config_rejects_invalid_values(self) -> None:
        base = {
            "quiet": False,
            "stats": False,
            "display_tools": False,
            "display_events": False,
            "display_tools_events": 2,
            "record": False,
            "interactive": True,
            "refresh_per_second": 1,
            "answer_height": 12,
            "answer_height_expand": False,
            "display_tokens": 0,
            "display_pause": 0,
            "display_probabilities": False,
            "display_probabilities_maximum": 0.8,
            "display_probabilities_sample_minimum": 0.1,
            "display_time_to_n_token": None,
            "display_reasoning_time": True,
        }
        cases = [
            ("display_tools_events", -1),
            ("refresh_per_second", 0),
            ("answer_height", -1),
            ("display_tokens", -1),
            ("display_pause", -1),
            ("display_time_to_n_token", 0),
            ("display_reasoning_raw", "yes"),
        ]

        for attribute, value in cases:
            with self.subTest(attribute=attribute):
                kwargs = base | {attribute: value}
                with self.assertRaises(AssertionError):
                    CliStreamDisplayConfig(**cast(Any, kwargs))

    def test_model_and_agent_run_parse_same_display_surface(self) -> None:
        cli = CLI(MagicMock())
        model_args = cli._parser.parse_args(
            [
                "model",
                "run",
                "model-id",
                "--stats",
                "--display-tools",
                "--display-events",
                "--display-tools-events",
                "0",
                "--display-reasoning-raw",
            ]
        )
        agent_args = cli._parser.parse_args(
            [
                "agent",
                "run",
                "agent.toml",
                "--stats",
                "--display-tools",
                "--display-events",
                "--display-tools-events",
                "0",
                "--display-reasoning-raw",
            ]
        )

        for parsed in (model_args, agent_args):
            self.assertTrue(parsed.stats)
            self.assertTrue(parsed.display_tools)
            self.assertTrue(parsed.display_events)
            self.assertEqual(parsed.display_tools_events, 0)
            self.assertTrue(parsed.display_reasoning_raw)

    def test_model_and_agent_run_default_to_unbounded_tool_history(
        self,
    ) -> None:
        cli = CLI(MagicMock())
        model_args = cli._parser.parse_args(["model", "run", "model-id"])
        agent_args = cli._parser.parse_args(["agent", "run", "agent.toml"])

        for parsed in (model_args, agent_args):
            self.assertIsNone(parsed.display_tools_events)
            self.assertFalse(parsed.display_reasoning_raw)
            config = cli_stream_display_config(
                parsed,
                refresh_per_second=1,
                interactive=True,
            )
            self.assertIsNone(config.display_tools_events)

    def test_model_and_agent_run_parse_quiet_record_display_surface(
        self,
    ) -> None:
        cli = CLI(MagicMock())
        cases = [
            [
                "model",
                "run",
                "model-id",
                "--quiet",
                "--record",
                "--stats",
                "--display-tools",
                "--display-events",
                "--display-tools-events",
                "0",
            ],
            [
                "agent",
                "run",
                "agent.toml",
                "--quiet",
                "--record",
                "--stats",
                "--display-tools",
                "--display-events",
                "--display-tools-events",
                "0",
            ],
        ]

        for argv in cases:
            with self.subTest(command=argv[:2]):
                parsed = cli._parser.parse_args(argv)
                config = cli_stream_display_config(
                    parsed,
                    refresh_per_second=1,
                    interactive=True,
                )

            self.assertTrue(parsed.quiet)
            self.assertTrue(parsed.record)
            self.assertTrue(parsed.stats)
            self.assertTrue(parsed.display_tools)
            self.assertTrue(parsed.display_events)
            self.assertEqual(parsed.display_tools_events, 0)
            self.assertFalse(config.record)
            self.assertFalse(config.show_stats)
            self.assertFalse(config.show_tools)
            self.assertFalse(config.show_events)
            self.assertTrue(config.answer_stdout_only)
