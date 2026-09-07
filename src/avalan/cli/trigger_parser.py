"""Declare bounded trigger commands without importing runtime extras."""

from .task_store import add_task_store_arguments

from argparse import ArgumentParser, _SubParsersAction


def add_deployment_arguments(
    parser: ArgumentParser, *, required: bool
) -> None:
    parser.add_argument(
        "--deployment-root",
        required=required,
        default=None,
        help="Persistent private immutable deployment catalog directory.",
    )
    add_encrypted_artifact_arguments(parser)


def add_encrypted_artifact_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--raw-storage-allowed",
        action="store_true",
        help="Authorize encrypted registered input and artifact retention.",
    )
    parser.add_argument("--retention-days", type=int, default=30)
    parser.add_argument("--max-artifact-bytes", type=int, default=16777216)


def add_trigger_commands(
    commands: "_SubParsersAction[ArgumentParser]",
    global_parser: ArgumentParser,
) -> None:
    group = commands.add_parser(
        "trigger",
        description="Register and operate durable time triggers",
        parents=[global_parser],
    )
    children = group.add_subparsers(dest="trigger_command", required=True)
    for name in (
        "validate",
        "preview",
        "apply",
        "list",
        "inspect",
        "occurrences",
        "events",
        "pause",
        "resume",
        "serve",
    ):
        parser = children.add_parser(name, parents=[global_parser])
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit one safe structured JSON result.",
        )
        if name in {"validate", "preview", "apply"}:
            parser.add_argument("configuration", help="Trigger TOML file.")
            parser.add_argument(
                "--root",
                default=None,
                help=(
                    "Trusted application root; defaults to the trigger file"
                    " directory."
                ),
            )
        if name not in {"validate", "preview"}:
            add_task_store_arguments(parser)
            parser.add_argument(
                "--owner-scope",
                default=None,
                help="Trusted owner; defaults to AVALAN_TASK_OWNER_SCOPE.",
            )
        if name in {"apply", "serve"}:
            add_deployment_arguments(parser, required=True)
        if name == "preview":
            parser.add_argument("--count", type=int, default=5)
            parser.add_argument(
                "--reference-time",
                default=None,
                help="Explicit aware ISO timestamp for reproducible preview.",
            )
        if name in {"inspect", "occurrences", "events", "pause", "resume"}:
            parser.add_argument("name")
        if name in {"list", "occurrences", "events"}:
            parser.add_argument("--limit", type=int, default=50)
            parser.add_argument("--cursor", type=int, default=0)
        if name == "occurrences":
            parser.add_argument(
                "--coverage",
                action="store_true",
                help=(
                    "Page compressed decision spans instead of individual"
                    " occurrences."
                ),
            )
        if name == "events":
            parser.add_argument(
                "--event-file",
                default=None,
                help=(
                    "Append this bounded committed event page for downstream"
                    " replay by event_id."
                ),
            )
        if name in {"apply", "pause", "resume"}:
            parser.add_argument(
                "--expected-generation", type=int, required=name != "apply"
            )
        if name == "serve":
            parser.add_argument(
                "--once",
                action="store_true",
                help="Admit one bounded batch; does not execute task runs.",
            )
            parser.add_argument(
                "--metrics-file",
                default=None,
                help=(
                    "Optional Prometheus textfile for aggregate scheduler"
                    " observations."
                ),
            )
            for option, default in (
                ("discovery-limit", 100),
                ("decisions-per-tick", 100),
                ("decisions-per-trigger", 10),
                ("admissions-per-tick", 100),
                ("admissions-per-trigger", 10),
                ("candidate-evaluations", 10000),
                ("poll-interval-seconds", 1),
                ("tick-timeout-seconds", 5),
                ("shutdown-timeout-seconds", 10),
                ("orphan-grace-seconds", 86400),
            ):
                parser.add_argument("--" + option, type=int, default=default)
