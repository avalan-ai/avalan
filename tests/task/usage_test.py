from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType, SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

import avalan.task.usage as usage_module
from avalan.task import (
    USAGE_COUNTER_NAMES,
    USAGE_METADATA_KEYS,
    TaskAttempt,
    TaskAttemptState,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskStoreConflictError,
    TaskStoreNotFoundError,
    UsageCounterPresence,
    UsageProviderFamily,
    UsageRecord,
    UsageSource,
    UsageTotals,
    attach_response_usage_recorder,
    freeze_usage_metadata,
    freeze_usage_value,
    stable_usage_id,
    stable_usage_id_for_response,
    usage_counter_presence,
    usage_observation_from_response,
    usage_observations_from_response,
    usage_smoke_summary,
    usage_totals_from_response,
)
from avalan.task.observability import record_response_usage
from avalan.task.stores import InMemoryTaskStore


class SequenceClock:
    def __init__(self) -> None:
        self._next = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        value = self._next
        self._next = self._next + timedelta(seconds=1)
        return value


class SequenceIds:
    def __init__(self) -> None:
        self._next = 1

    def __call__(self) -> str:
        value = f"id-{self._next}"
        self._next = self._next + 1
        return value


class FixedIds:
    def __init__(self, *values: str) -> None:
        self._values = iter(values)

    def __call__(self) -> str:
        return next(self._values)


class FakeConsumedResponse:
    input_token_count: int | None
    output_token_count: int | None

    def __init__(
        self,
        *,
        input_token_count: int | None = 3,
        output_token_count: int | None = 4,
    ) -> None:
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count
        self._callbacks: list[Callable[[], Awaitable[None] | None]] = []

    def add_done_callback(
        self,
        callback: Callable[[], Awaitable[None] | None],
    ) -> None:
        self._callbacks.append(callback)

    async def consume(self) -> None:
        for callback in self._callbacks:
            result = callback()
            if isinstance(result, Awaitable):
                await result


class FakeProviderResponse(FakeConsumedResponse):
    def __init__(self) -> None:
        super().__init__()
        self.usage = SimpleNamespace(
            input_tokens=3,
            provider_family=UsageProviderFamily.OPENAI,
            input_tokens_details=SimpleNamespace(cached_tokens=1),
            cache_creation_input_tokens=2,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=6,
                ephemeral_1h_input_tokens=8,
            ),
            output_tokens=4,
            output_tokens_details={"reasoning_tokens": 5},
            total_tokens=99,
        )


class FakeUsageWrapperResponse(FakeConsumedResponse):
    def __init__(self, *responses: object) -> None:
        super().__init__(
            input_token_count=None,
            output_token_count=None,
        )
        self.usage_responses = responses


class HostileResponse:
    input_token_count = -1
    output_token_count = True
    cached_input_token_count = "raw"


class MultiCallUsageResponse:
    def __init__(
        self, *responses: object, usage: object | None = None
    ) -> None:
        self.usage_responses = responses
        self.usage = usage


class WeakUsageResponse:
    __slots__ = ("__weakref__",)


class CallableUsageResponses:
    def __init__(self, responses: object) -> None:
        self._responses = responses

    def usage_responses(self) -> object:
        return self._responses


class RaisingUsageProperties:
    @property
    def usage_responses(self) -> object:
        raise RuntimeError("private usage responses failure")

    @property
    def usage(self) -> object:
        raise RuntimeError("private usage failure")

    @property
    def usage_metadata(self) -> object:
        raise RuntimeError("private usage metadata failure")

    @property
    def usageMetadata(self) -> object:
        raise RuntimeError("private camel usage metadata failure")

    @property
    def input_token_count(self) -> int:
        raise RuntimeError("private input counter failure")


class RaisingUsageResponsesCallable:
    def usage_responses(self) -> object:
        raise RuntimeError("private usage responses callable failure")


class RaisingStoredUsageKey:
    @property
    def _avalan_usage_call_key(self) -> object:
        raise RuntimeError("private stored usage key failure")


def definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="summarize", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/summarize.toml"),
    )


class UsageTotalsTest(TestCase):
    def test_usage_counter_and_metadata_contracts_are_stable(self) -> None:
        self.assertEqual(
            USAGE_COUNTER_NAMES,
            (
                "input_tokens",
                "cached_input_tokens",
                "cache_creation_input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
            ),
        )
        self.assertEqual(
            USAGE_METADATA_KEYS,
            (
                "provider_family",
                "cache_creation_ephemeral_5m_input_tokens",
                "cache_creation_ephemeral_1h_input_tokens",
                "cache_read_ephemeral_5m_input_tokens",
                "cache_read_ephemeral_1h_input_tokens",
            ),
        )

    def test_usage_counter_presence_distinguishes_missing_zero_and_positive(
        self,
    ) -> None:
        presence = usage_counter_presence(
            UsageTotals(
                input_tokens=None,
                cached_input_tokens=0,
                cache_creation_input_tokens=2,
                output_tokens=5,
            )
        )

        self.assertEqual(
            presence["input_tokens"],
            UsageCounterPresence.MISSING,
        )
        self.assertEqual(
            presence["cached_input_tokens"],
            UsageCounterPresence.REPORTED_ZERO,
        )
        self.assertEqual(
            presence["cache_creation_input_tokens"],
            UsageCounterPresence.REPORTED_POSITIVE,
        )
        self.assertEqual(
            presence["output_tokens"],
            UsageCounterPresence.REPORTED_POSITIVE,
        )
        with self.assertRaises(TypeError):
            cast(dict[str, object], presence)["input_tokens"] = "private"

    def test_usage_smoke_summary_is_redacted_and_numeric(self) -> None:
        summary = usage_smoke_summary(
            task_variant="native_pdf",
            success=True,
            schema_valid=True,
            expected_output_match=False,
            totals=UsageTotals(
                input_tokens=12,
                cached_input_tokens=0,
                output_tokens=8,
                reasoning_tokens=3,
                total_tokens=20,
            ),
            required_counters=(
                "cached_input_tokens",
                "reasoning_tokens",
            ),
        )

        self.assertEqual(summary["task_variant"], "native_pdf")
        self.assertTrue(summary["success"])
        self.assertTrue(summary["schema_valid"])
        self.assertFalse(summary["expected_output_match"])
        self.assertTrue(summary["required_usage_present"])
        presence = summary["usage_field_presence"]
        assert isinstance(presence, Mapping)
        self.assertEqual(
            presence["cached_input_tokens"],
            "reported_zero",
        )
        self.assertEqual(
            presence["reasoning_tokens"],
            "reported_positive",
        )
        totals = summary["usage_totals"]
        assert isinstance(totals, Mapping)
        self.assertEqual(totals["input_tokens"], 12)
        self.assertEqual(totals["cached_input_tokens"], 0)
        self.assertIsNone(totals["cache_creation_input_tokens"])
        self.assertNotIn("private", str(summary))

    def test_usage_smoke_summary_reports_missing_required_counters(
        self,
    ) -> None:
        summary = usage_smoke_summary(
            task_variant="image_flow",
            success=True,
            schema_valid=True,
            expected_output_match=True,
            totals=UsageTotals(input_tokens=0),
            required_counters=("reasoning_tokens",),
        )

        self.assertFalse(summary["required_usage_present"])
        presence = summary["usage_field_presence"]
        assert isinstance(presence, Mapping)
        self.assertEqual(presence["input_tokens"], "reported_zero")
        self.assertEqual(presence["reasoning_tokens"], "missing")

    def test_usage_smoke_summary_rejects_unsafe_values(self) -> None:
        hostile_variants = (
            "",
            "../native_pdf",
            "NativePDF",
            "native pdf",
            "native_pdf/private",
        )
        for task_variant in hostile_variants:
            with self.subTest(task_variant=task_variant):
                with self.assertRaises(AssertionError):
                    usage_smoke_summary(
                        task_variant=task_variant,
                        success=True,
                        schema_valid=True,
                        expected_output_match=True,
                        totals=UsageTotals(input_tokens=1),
                    )
        with self.assertRaises(AssertionError):
            usage_smoke_summary(
                task_variant="native_pdf",
                success=True,
                schema_valid=True,
                expected_output_match=True,
                totals=UsageTotals(input_tokens=1),
                required_counters=("private_counter",),
            )
        with self.assertRaises(AssertionError):
            usage_smoke_summary(
                task_variant="native_pdf",
                success=cast(bool, 1),
                schema_valid=True,
                expected_output_match=True,
                totals=UsageTotals(input_tokens=1),
            )

    def test_response_usage_preserves_unavailable_counters_as_none(
        self,
    ) -> None:
        totals = usage_totals_from_response(
            FakeConsumedResponse(input_token_count=None, output_token_count=2)
        )

        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertIsNone(totals.input_tokens)
        self.assertEqual(totals.output_tokens, 2)
        self.assertIsNone(totals.cached_input_tokens)
        self.assertIsNone(totals.cache_creation_input_tokens)
        self.assertIsNone(totals.reasoning_tokens)
        self.assertIsNone(totals.total_tokens)

    def test_response_usage_keeps_unreported_total_unavailable(
        self,
    ) -> None:
        totals = usage_totals_from_response(FakeConsumedResponse())

        self.assertEqual(
            totals,
            UsageTotals(
                input_tokens=3,
                output_tokens=4,
            ),
        )

    def test_provider_reported_fields_are_preserved(self) -> None:
        observation = usage_observation_from_response(FakeProviderResponse())

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.source, UsageSource.EXACT)
        self.assertEqual(
            observation.metadata["provider_family"],
            UsageProviderFamily.OPENAI.value,
        )
        self.assertEqual(
            observation.totals,
            UsageTotals(
                input_tokens=3,
                cached_input_tokens=1,
                cache_creation_input_tokens=2,
                output_tokens=4,
                reasoning_tokens=5,
                total_tokens=99,
            ),
        )

    def test_openai_responses_usage_fields_are_preserved(self) -> None:
        response = SimpleNamespace(
            provider_family="azure_openai",
            usage={
                "input_tokens": 12,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens": 8,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 20,
                "model": "private-deployment-name",
                "response_id": "private-response-id",
            },
        )

        observation = usage_observation_from_response(response)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.source, UsageSource.EXACT)
        self.assertEqual(
            observation.metadata,
            {"provider_family": UsageProviderFamily.AZURE_OPENAI.value},
        )
        self.assertEqual(
            observation.totals,
            UsageTotals(
                input_tokens=12,
                cached_input_tokens=5,
                output_tokens=8,
                reasoning_tokens=3,
                total_tokens=20,
            ),
        )
        self.assertIsNone(observation.totals.cache_creation_input_tokens)
        self.assertNotIn("private-deployment-name", str(observation))
        self.assertNotIn("private-response-id", str(observation))

    def test_openai_chat_compatible_usage_fields_are_preserved(
        self,
    ) -> None:
        response = SimpleNamespace(
            provider_family="openai",
            usage={
                "prompt_tokens": 9,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 2},
                "total_tokens": 16,
            },
        )

        totals = usage_totals_from_response(response)

        self.assertEqual(
            totals,
            UsageTotals(
                input_tokens=9,
                cached_input_tokens=4,
                output_tokens=7,
                reasoning_tokens=2,
                total_tokens=16,
            ),
        )
        assert totals is not None
        self.assertIsNone(totals.cache_creation_input_tokens)

    def test_openai_usage_drops_malformed_counters_without_inference(
        self,
    ) -> None:
        response = SimpleNamespace(
            provider_family="azure_openai",
            usage={
                "input_tokens": True,
                "input_tokens_details": {"cached_tokens": -1},
                "prompt_tokens_details": {"cached_tokens": "4"},
                "output_tokens": 6,
                "output_tokens_details": {"reasoning_tokens": 1.5},
                "completion_tokens_details": {"reasoning_tokens": False},
                "total_tokens": "6",
                "model": "private-deployment-name",
            },
        )

        observation = usage_observation_from_response(response)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.totals,
            UsageTotals(output_tokens=6),
        )
        self.assertIsNone(observation.totals.cache_creation_input_tokens)
        self.assertEqual(
            observation.metadata,
            {"provider_family": UsageProviderFamily.AZURE_OPENAI.value},
        )
        self.assertNotIn("private-deployment-name", str(observation))

    def test_provider_usage_preserves_zero_and_drops_hostile_values(
        self,
    ) -> None:
        response = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": None,
                "promptTokenCount": float("nan"),
                "cached_input_tokens": False,
                "cache_creation_input_tokens": -1,
                "output_tokens": 0,
                "reasoning_tokens": 1.5,
                "total_tokens": "0",
                "input_tokens_details": {
                    "cached_tokens": object(),
                    "unknown": {"nested": "private nested body"},
                },
                "output_tokens_details": {
                    "reasoning_tokens": float("inf"),
                    "signature": "private reasoning signature",
                },
                "headers": {"authorization": "Bearer private-provider-key"},
                "response_metadata": {"id": "private-response-id"},
            },
        )

        observation = usage_observation_from_response(response)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.source, UsageSource.EXACT)
        self.assertEqual(
            observation.totals,
            UsageTotals(output_tokens=0),
        )
        self.assertEqual(
            observation.metadata,
            {"provider_family": UsageProviderFamily.OPENAI.value},
        )
        rendered = str(observation)
        self.assertNotIn("private nested body", rendered)
        self.assertNotIn("private reasoning signature", rendered)
        self.assertNotIn("private-provider-key", rendered)
        self.assertNotIn("private-response-id", rendered)

    def test_usage_metadata_drops_adversarial_privacy_sentinels(
        self,
    ) -> None:
        sentinels = (
            "private system instructions",
            "private user prompt",
            "private token text",
            "private reasoning text",
            "private reasoning signature",
            "private redacted blob",
            "private PDF bytes",
            "private image bytes",
            "data:application/pdf;base64,private",
            "data:image/png;base64,private",
            "private-file.pdf",
            "/private/tmp/private-file.pdf",
            "../private/path",
            "private-cache-key",
            "private-cache-handle",
            "private-request-id",
            "private-response-id",
            "private-azure-deployment",
            "private-region",
            "https://private-endpoint.example/openai/v1/",
            "Bearer private-provider-key",
            "private/raw-model-id",
            "private provider error body",
        )
        response = SimpleNamespace(
            provider_family="azure_openai",
            usage={
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "prompt": sentinels[0],
                "user_prompt": sentinels[1],
                "token_text": sentinels[2],
                "reasoning": {
                    "text": sentinels[3],
                    "signature": sentinels[4],
                    "redacted": sentinels[5],
                },
                "file_data": sentinels[8],
                "image_url": sentinels[9],
                "filename": sentinels[10],
                "path": sentinels[11],
                "cache_key": sentinels[13],
                "cache_handle": sentinels[14],
                "request_id": sentinels[15],
                "response_id": sentinels[16],
                "deployment": sentinels[17],
                "region": sentinels[18],
                "endpoint": sentinels[19],
                "headers": {"authorization": sentinels[20]},
                "model": sentinels[21],
                "error": {"body": sentinels[22]},
            },
        )

        observation = usage_observation_from_response(response)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.metadata,
            {"provider_family": UsageProviderFamily.AZURE_OPENAI.value},
        )
        rendered = str(observation)
        for sentinel in sentinels:
            with self.subTest(sentinel=sentinel):
                self.assertNotIn(sentinel, rendered)

    def test_streaming_vendor_usage_aliases_are_preserved(self) -> None:
        google = usage_totals_from_response(
            SimpleNamespace(
                usageMetadata={
                    "promptTokenCount": 4,
                    "cachedContentTokenCount": 1,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 2,
                    "totalTokenCount": 9,
                }
            )
        )
        bedrock = usage_totals_from_response(
            SimpleNamespace(
                usage={
                    "inputTokens": 5,
                    "cacheReadInputTokens": 1,
                    "cacheWriteInputTokens": 2,
                    "outputTokens": 7,
                    "totalTokens": 12,
                }
            )
        )

        self.assertEqual(
            google,
            UsageTotals(
                input_tokens=4,
                cached_input_tokens=1,
                output_tokens=3,
                reasoning_tokens=2,
                total_tokens=9,
            ),
        )
        self.assertEqual(
            bedrock,
            UsageTotals(
                input_tokens=5,
                cached_input_tokens=1,
                cache_creation_input_tokens=2,
                output_tokens=7,
                total_tokens=12,
            ),
        )

    def test_nested_provider_usage_containers_are_preserved(self) -> None:
        openai = SimpleNamespace(
            type="response.completed",
            provider_family="openai",
            response={
                "usage": {
                    "input_tokens": 7,
                    "input_tokens_details": {"cached_tokens": 2},
                    "output_tokens": 5,
                    "output_tokens_details": {"reasoning_tokens": 3},
                    "total_tokens": 12,
                    "response_id": "private-response-id",
                }
            },
        )
        anthropic = {
            "type": "message_delta",
            "provider_family": "anthropic",
            "delta": {
                "usage": {
                    "input_tokens": 11,
                    "cache_read_input_tokens": 4,
                    "cache_creation_input_tokens": 6,
                    "output_tokens": 8,
                    "output_tokens_details": {"thinking_tokens": 5},
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 9,
                        "cache_key": "private-cache-key",
                    },
                }
            },
        }
        bedrock = {
            "metadata": {
                "usage": {
                    "inputTokens": 13,
                    "cacheReadInputTokens": 1,
                    "cacheWriteInputTokens": 2,
                    "outputTokens": 17,
                    "totalTokens": 30,
                },
                "endpoint": "https://private-endpoint.example",
            },
            "provider_family": "bedrock",
        }

        openai_observation = usage_observation_from_response(openai)
        anthropic_observation = usage_observation_from_response(anthropic)
        bedrock_observation = usage_observation_from_response(bedrock)

        self.assertIsNotNone(openai_observation)
        self.assertIsNotNone(anthropic_observation)
        self.assertIsNotNone(bedrock_observation)
        assert openai_observation is not None
        assert anthropic_observation is not None
        assert bedrock_observation is not None
        self.assertEqual(
            openai_observation.totals,
            UsageTotals(
                input_tokens=7,
                cached_input_tokens=2,
                output_tokens=5,
                reasoning_tokens=3,
                total_tokens=12,
            ),
        )
        self.assertEqual(
            openai_observation.metadata,
            {"provider_family": UsageProviderFamily.OPENAI.value},
        )
        self.assertEqual(
            anthropic_observation.totals,
            UsageTotals(
                input_tokens=11,
                cached_input_tokens=4,
                cache_creation_input_tokens=6,
                output_tokens=8,
                reasoning_tokens=5,
            ),
        )
        self.assertEqual(
            anthropic_observation.metadata,
            {
                "provider_family": UsageProviderFamily.ANTHROPIC.value,
                "cache_creation_ephemeral_5m_input_tokens": 9,
            },
        )
        self.assertEqual(
            bedrock_observation.totals,
            UsageTotals(
                input_tokens=13,
                cached_input_tokens=1,
                cache_creation_input_tokens=2,
                output_tokens=17,
                total_tokens=30,
            ),
        )
        self.assertEqual(
            bedrock_observation.metadata,
            {"provider_family": UsageProviderFamily.BEDROCK.value},
        )
        rendered = (
            str(openai_observation)
            + str(anthropic_observation)
            + str(bedrock_observation)
        )
        self.assertNotIn("private-response-id", rendered)
        self.assertNotIn("private-cache-key", rendered)
        self.assertNotIn("private-endpoint.example", rendered)

    def test_malformed_nested_provider_usage_is_ignored_safely(
        self,
    ) -> None:
        response = SimpleNamespace(
            provider_family="azure_openai",
            response={
                "usage": {
                    "input_tokens": "private prompt token text",
                    "cached_input_tokens": True,
                    "output_tokens": -1,
                    "reasoning_tokens": 1.5,
                    "total_tokens": float("nan"),
                    "response_id": "private-response-id",
                }
            },
        )

        self.assertIsNone(usage_observation_from_response(response))
        self.assertNotIn(
            "private-response-id",
            str(usage_observations_from_response(response)),
        )

    def test_anthropic_usage_preserves_cache_and_thinking_details(
        self,
    ) -> None:
        response = SimpleNamespace(
            provider_family="anthropic",
            usage={
                "input_tokens": 7,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 3,
                "output_tokens": 5,
                "output_tokens_details": {"thinking_tokens": 4},
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 11,
                    "ephemeral_1h_input_tokens": 13,
                    "cache_key": "private-cache-key",
                },
            },
        )

        observation = usage_observation_from_response(response)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.source, UsageSource.EXACT)
        self.assertEqual(
            observation.totals,
            UsageTotals(
                input_tokens=7,
                cached_input_tokens=2,
                cache_creation_input_tokens=3,
                output_tokens=5,
                reasoning_tokens=4,
            ),
        )
        self.assertEqual(
            observation.metadata,
            {
                "provider_family": UsageProviderFamily.ANTHROPIC.value,
                "cache_creation_ephemeral_5m_input_tokens": 11,
                "cache_creation_ephemeral_1h_input_tokens": 13,
            },
        )
        self.assertNotIn("private-cache-key", str(observation))

    def test_bedrock_usage_preserves_cache_write_and_details_metadata(
        self,
    ) -> None:
        response = SimpleNamespace(
            provider_family="bedrock",
            usage={
                "inputTokens": 10,
                "cacheReadInputTokens": 1,
                "cache_write_input_tokens": 2,
                "outputTokens": 8,
                "totalTokens": 18,
                "reasoning": {"text": "private visible reasoning"},
                "cacheDetails": {
                    "cacheRead": {
                        "ephemeral5mInputTokens": 4,
                        "ephemeral1hInputTokens": 5,
                    },
                    "cacheWrite": {
                        "ephemeral5mInputTokens": 6,
                        "ephemeral1hInputTokens": 7,
                    },
                    "cacheHandle": "private-cache-handle",
                },
            },
        )

        observation = usage_observation_from_response(response)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.totals,
            UsageTotals(
                input_tokens=10,
                cached_input_tokens=1,
                cache_creation_input_tokens=2,
                output_tokens=8,
                total_tokens=18,
            ),
        )
        self.assertIsNone(observation.totals.reasoning_tokens)
        self.assertEqual(
            observation.metadata,
            {
                "provider_family": UsageProviderFamily.BEDROCK.value,
                "cache_creation_ephemeral_5m_input_tokens": 6,
                "cache_creation_ephemeral_1h_input_tokens": 7,
                "cache_read_ephemeral_5m_input_tokens": 4,
                "cache_read_ephemeral_1h_input_tokens": 5,
            },
        )
        self.assertNotIn("private visible reasoning", str(observation))
        self.assertNotIn("private-cache-handle", str(observation))

    def test_google_usage_preserves_snake_and_camel_metadata(self) -> None:
        camel = usage_totals_from_response(
            SimpleNamespace(
                provider_family="google",
                usageMetadata={
                    "promptTokenCount": 4,
                    "cachedContentTokenCount": 1,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 2,
                    "totalTokenCount": 9,
                },
            )
        )
        snake = usage_totals_from_response(
            SimpleNamespace(
                provider_family="google",
                usage_metadata={
                    "prompt_token_count": 5,
                    "cached_content_token_count": 2,
                    "candidates_token_count": 4,
                    "thoughts_token_count": 1,
                    "total_token_count": 11,
                },
            )
        )

        self.assertEqual(
            camel,
            UsageTotals(
                input_tokens=4,
                cached_input_tokens=1,
                output_tokens=3,
                reasoning_tokens=2,
                total_tokens=9,
            ),
        )
        self.assertEqual(
            snake,
            UsageTotals(
                input_tokens=5,
                cached_input_tokens=2,
                output_tokens=4,
                reasoning_tokens=1,
                total_tokens=11,
            ),
        )

    def test_provider_usage_does_not_infer_from_hints(
        self,
    ) -> None:
        response = SimpleNamespace(
            provider_family="anthropic",
            usage={
                "input_tokens": 1,
                "output_tokens": 2,
                "output_tokens_details": {"thinking_tokens": "private text"},
                "thinking": {"budget_tokens": 9000},
                "reasoning_effort": "high",
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 3,
                    "ephemeral_1h_input_tokens": True,
                },
                "latency_ms": 1,
            },
        )

        observation = usage_observation_from_response(response)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.totals,
            UsageTotals(input_tokens=1, output_tokens=2),
        )
        self.assertEqual(
            observation.metadata,
            {
                "provider_family": UsageProviderFamily.ANTHROPIC.value,
                "cache_creation_ephemeral_5m_input_tokens": 3,
            },
        )
        self.assertNotIn("private text", str(observation))

    def test_text_response_counts_are_estimated(self) -> None:
        observation = usage_observation_from_response(FakeConsumedResponse())

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.source, UsageSource.ESTIMATED)
        self.assertIsNone(observation.totals.total_tokens)

    def test_direct_response_source_marker_does_not_make_counts_exact(
        self,
    ) -> None:
        marked_enum = usage_observation_from_response(
            SimpleNamespace(
                input_token_count=1, usage_source=UsageSource.EXACT
            )
        )
        marked_string = usage_observation_from_response(
            SimpleNamespace(input_token_count=1, usage_source="exact")
        )
        invalid_string = usage_observation_from_response(
            SimpleNamespace(input_token_count=1, usage_source="unknown")
        )

        self.assertIsNotNone(marked_enum)
        self.assertIsNotNone(marked_string)
        self.assertIsNotNone(invalid_string)
        assert marked_enum is not None
        assert marked_string is not None
        assert invalid_string is not None
        self.assertEqual(marked_enum.source, UsageSource.ESTIMATED)
        self.assertEqual(marked_string.source, UsageSource.ESTIMATED)
        self.assertEqual(invalid_string.source, UsageSource.ESTIMATED)

    def test_invalid_response_counters_are_unavailable(self) -> None:
        self.assertIsNone(usage_totals_from_response(object()))
        self.assertIsNone(usage_totals_from_response(HostileResponse()))

    def test_usage_observations_preserve_per_call_provider_usage(
        self,
    ) -> None:
        response = MultiCallUsageResponse(
            SimpleNamespace(
                usage={
                    "input_tokens": 2,
                    "cached_input_tokens": 1,
                    "output_tokens": 3,
                    "total_tokens": 5,
                    "provider_family": "openai",
                }
            ),
            SimpleNamespace(
                usage={
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 2,
                    "output_tokens": 6,
                    "reasoning_tokens": 1,
                    "total_tokens": 10,
                    "provider_family": "openai",
                    "raw_response_id": "private-response-id",
                }
            ),
        )

        observations = usage_observations_from_response(response)
        aggregate = usage_observation_from_response(response)

        self.assertEqual(len(observations), 2)
        self.assertTrue(
            all(
                observation.source == UsageSource.EXACT
                for observation in observations
            )
        )
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.source, UsageSource.EXACT)
        self.assertEqual(aggregate.totals.input_tokens, 6)
        self.assertEqual(aggregate.totals.cached_input_tokens, 1)
        self.assertEqual(aggregate.totals.cache_creation_input_tokens, 2)
        self.assertEqual(aggregate.totals.output_tokens, 9)
        self.assertEqual(aggregate.totals.reasoning_tokens, 1)
        self.assertEqual(aggregate.totals.total_tokens, 15)
        self.assertEqual(
            aggregate.metadata["provider_family"],
            UsageProviderFamily.OPENAI.value,
        )
        self.assertNotIn("raw_response_id", aggregate.metadata)

    def test_usage_observations_drop_malformed_nested_provider_usage(
        self,
    ) -> None:
        response = MultiCallUsageResponse(
            SimpleNamespace(
                usage={
                    "input_tokens": -1,
                    "cached_input_tokens": True,
                    "output_tokens": "private-token-count",
                    "total_tokens": 1.5,
                    "provider_family": "private-provider",
                }
            ),
            SimpleNamespace(
                usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            ),
        )

        observations = usage_observations_from_response(response)
        aggregate = usage_observation_from_response(response)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].totals.input_tokens, 0)
        self.assertEqual(observations[0].totals.output_tokens, 0)
        self.assertEqual(observations[0].totals.total_tokens, 0)
        self.assertEqual(observations[0].metadata, {})
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.totals.total_tokens, 0)

    def test_public_usage_tuple_is_split_into_per_call_observations(
        self,
    ) -> None:
        response = MultiCallUsageResponse(
            usage=(
                {"input_tokens": 1, "total_tokens": 2},
                {"input_tokens": 3, "total_tokens": 4},
                {"input_tokens": "private malformed"},
            )
        )

        observations = usage_observations_from_response(response)
        aggregate = usage_observation_from_response(response)

        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0].totals.input_tokens, 1)
        self.assertEqual(observations[1].totals.input_tokens, 3)
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.totals.input_tokens, 4)
        self.assertEqual(aggregate.totals.total_tokens, 6)

    def test_public_usage_items_use_explicit_mapping_identity(
        self,
    ) -> None:
        response = MultiCallUsageResponse(
            usage=(
                {
                    "usage_call_key": "private-provider-call",
                    "input_tokens": 1,
                    "provider_family": "openai",
                },
                {"input_tokens": 2, "provider_family": "openai"},
            )
        )
        replayed = MultiCallUsageResponse(
            usage=(
                {
                    "usage_call_key": "private-provider-call",
                    "input_tokens": 1,
                    "provider_family": "openai",
                },
                {"input_tokens": 2, "provider_family": "openai"},
            )
        )

        entries = usage_module.usage_observation_entries_from_response(
            response
        )
        replayed_entries = (
            usage_module.usage_observation_entries_from_response(replayed)
        )

        self.assertEqual(len(entries), 2)
        self.assertEqual(len(replayed_entries), 2)
        usage_id = stable_usage_id_for_response(
            entries[0].response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=entries[0].sequence,
        )
        replayed_usage_id = stable_usage_id_for_response(
            replayed_entries[0].response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=replayed_entries[0].sequence,
        )
        unkeyed_usage_id = stable_usage_id_for_response(
            entries[1].response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=entries[1].sequence,
        )
        replayed_unkeyed_usage_id = stable_usage_id_for_response(
            replayed_entries[1].response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=replayed_entries[1].sequence,
        )

        self.assertEqual(usage_id, replayed_usage_id)
        self.assertNotEqual(unkeyed_usage_id, replayed_unkeyed_usage_id)
        self.assertNotIn("private-provider-call", usage_id)

    def test_callable_usage_responses_are_supported(self) -> None:
        response = CallableUsageResponses(
            (SimpleNamespace(input_token_count=1, output_token_count=2),)
        )

        observations = usage_observations_from_response(response)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source, UsageSource.ESTIMATED)
        self.assertEqual(observations[0].totals.input_tokens, 1)
        self.assertEqual(observations[0].totals.output_tokens, 2)

    def test_invalid_usage_responses_shape_is_ignored(self) -> None:
        response = CallableUsageResponses("private invalid shape")

        self.assertEqual(usage_observations_from_response(response), ())

    def test_raising_usage_accessors_are_ignored_safely(self) -> None:
        properties = RaisingUsageProperties()
        callable_response = RaisingUsageResponsesCallable()

        self.assertIsNone(usage_observation_from_response(properties))
        self.assertEqual(usage_observations_from_response(properties), ())
        self.assertEqual(
            usage_observations_from_response(callable_response),
            (),
        )

    def test_raising_stored_usage_key_falls_back_safely(self) -> None:
        response = RaisingStoredUsageKey()

        usage_id = stable_usage_id_for_response(
            response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        same_usage_id = stable_usage_id_for_response(
            response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )

        self.assertEqual(usage_id, same_usage_id)
        self.assertTrue(usage_id.startswith("usage-"))
        self.assertNotIn("private stored usage key failure", usage_id)

    def test_usage_responses_skip_self_references(self) -> None:
        response = MultiCallUsageResponse()
        response.usage_responses = (
            response,
            SimpleNamespace(usage={"input_tokens": 1}),
        )

        observations = usage_observations_from_response(response)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].totals.input_tokens, 1)

    def test_child_usage_tuple_is_split_into_observations(self) -> None:
        response = MultiCallUsageResponse(
            SimpleNamespace(
                usage=(
                    {"input_tokens": 1, "provider_family": "openai"},
                    {"input_tokens": 2, "provider_family": "openai"},
                )
            )
        )

        observations = usage_observations_from_response(response)
        aggregate = usage_observation_from_response(response)

        self.assertEqual(len(observations), 2)
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.totals.input_tokens, 3)
        self.assertEqual(
            aggregate.metadata["provider_family"],
            UsageProviderFamily.OPENAI.value,
        )

    def test_malformed_child_usage_does_not_hide_nested_usage(
        self,
    ) -> None:
        nested = MultiCallUsageResponse(
            SimpleNamespace(
                usage={
                    "input_tokens": 6,
                    "output_tokens": 4,
                    "total_tokens": 10,
                    "provider_family": "openai",
                }
            )
        )
        wrapper = MultiCallUsageResponse(nested)
        wrapper.usage = {
            "input_tokens": "private prompt token text",
            "output_tokens": -1,
            "total_tokens": True,
            "provider_family": "private-provider",
        }
        response = MultiCallUsageResponse(wrapper)

        observations = usage_observations_from_response(response)
        aggregate = usage_observation_from_response(response)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source, UsageSource.EXACT)
        self.assertEqual(
            observations[0].totals,
            UsageTotals(input_tokens=6, output_tokens=4, total_tokens=10),
        )
        self.assertEqual(
            observations[0].metadata,
            {"provider_family": UsageProviderFamily.OPENAI.value},
        )
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.totals.input_tokens, 6)
        self.assertNotIn("private prompt token text", str(observations))
        self.assertNotIn("private-provider", str(observations))

    def test_malformed_child_usage_falls_back_to_estimated_counts(
        self,
    ) -> None:
        response = MultiCallUsageResponse(
            SimpleNamespace(
                input_token_count=7,
                output_token_count=3,
                provider_family="openai",
                usage={
                    "input_tokens": "private prompt token text",
                    "output_tokens": -1,
                    "total_tokens": True,
                    "raw_response_body": "private provider body",
                },
            ),
            SimpleNamespace(
                input_token_count=5,
                output_token_count=2,
                provider_family="openai",
                usage=(
                    {
                        "input_tokens": "private nested prompt text",
                        "total_tokens": False,
                    },
                ),
            ),
        )

        observations = usage_observations_from_response(response)
        aggregate = usage_observation_from_response(response)

        self.assertEqual(len(observations), 2)
        self.assertTrue(
            all(
                observation.source == UsageSource.ESTIMATED
                for observation in observations
            )
        )
        self.assertEqual(observations[0].totals.input_tokens, 7)
        self.assertEqual(observations[0].totals.output_tokens, 3)
        self.assertEqual(observations[1].totals.input_tokens, 5)
        self.assertEqual(observations[1].totals.output_tokens, 2)
        self.assertEqual(
            observations[0].metadata,
            {"provider_family": UsageProviderFamily.OPENAI.value},
        )
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.source, UsageSource.ESTIMATED)
        self.assertEqual(aggregate.totals.input_tokens, 12)
        self.assertEqual(aggregate.totals.output_tokens, 5)
        self.assertEqual(
            aggregate.metadata["provider_family"],
            UsageProviderFamily.OPENAI.value,
        )
        rendered = str(observations) + str(aggregate)
        self.assertNotIn("private prompt token text", rendered)
        self.assertNotIn("private provider body", rendered)
        self.assertNotIn("private nested prompt text", rendered)

    def test_nested_usage_responses_are_flattened_per_call(self) -> None:
        nested = MultiCallUsageResponse(
            SimpleNamespace(
                usage={
                    "input_tokens": 1,
                    "total_tokens": 2,
                    "provider_family": "openai",
                }
            ),
            SimpleNamespace(
                usage={
                    "input_tokens": 3,
                    "reasoning_tokens": 1,
                    "total_tokens": 4,
                    "provider_family": "openai",
                }
            ),
        )
        response = MultiCallUsageResponse(
            nested,
            SimpleNamespace(
                usage={
                    "input_tokens": 5,
                    "cached_input_tokens": 2,
                    "total_tokens": 6,
                    "provider_family": "openai",
                }
            ),
        )

        observations = usage_observations_from_response(response)
        aggregate = usage_observation_from_response(response)

        self.assertEqual(len(observations), 3)
        self.assertEqual(
            [observation.totals.input_tokens for observation in observations],
            [1, 3, 5],
        )
        self.assertEqual(observations[1].totals.reasoning_tokens, 1)
        self.assertEqual(observations[2].totals.cached_input_tokens, 2)
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.totals.input_tokens, 9)
        self.assertEqual(aggregate.totals.cached_input_tokens, 2)
        self.assertEqual(aggregate.totals.reasoning_tokens, 1)
        self.assertEqual(aggregate.totals.total_tokens, 12)
        self.assertEqual(
            aggregate.metadata["provider_family"],
            UsageProviderFamily.OPENAI.value,
        )

    def test_cyclic_usage_responses_are_ignored(self) -> None:
        parent = MultiCallUsageResponse()
        child = MultiCallUsageResponse(parent)
        parent.usage_responses = (
            child,
            SimpleNamespace(
                usage={
                    "input_tokens": 7,
                    "total_tokens": 8,
                    "provider_family": "openai",
                }
            ),
        )

        observations = usage_observations_from_response(parent)
        aggregate = usage_observation_from_response(parent)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].totals.input_tokens, 7)
        self.assertEqual(observations[0].totals.total_tokens, 8)
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.totals.input_tokens, 7)
        self.assertEqual(aggregate.totals.total_tokens, 8)

    def test_seen_usage_response_is_ignored(self) -> None:
        response = MultiCallUsageResponse(
            SimpleNamespace(usage={"input_tokens": 1})
        )

        observations = usage_module._child_usage_observations(
            response,
            seen={id(response)},
        )

        self.assertEqual(observations, ())

    def test_mixed_source_aggregate_is_estimated_without_shared_metadata(
        self,
    ) -> None:
        response = MultiCallUsageResponse(
            SimpleNamespace(input_token_count=1),
            SimpleNamespace(
                usage={
                    "input_tokens": 2,
                    "provider_family": "openai",
                }
            ),
        )

        aggregate = usage_observation_from_response(response)

        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.source, UsageSource.ESTIMATED)
        self.assertEqual(aggregate.totals.input_tokens, 3)
        self.assertEqual(aggregate.metadata, {})

    def test_usage_entities_are_frozen_and_reject_invalid_counters(
        self,
    ) -> None:
        record = UsageRecord(
            usage_id="usage-1",
            run_id="run-1",
            attempt_id=None,
            sequence=1,
            source=UsageSource.UNAVAILABLE,
            totals=UsageTotals(),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            metadata={"provider_family": UsageProviderFamily.LOCAL},
        )

        self.assertFalse(record.totals.has_observations)
        self.assertIsInstance(record.metadata, MappingProxyType)
        self.assertEqual(
            record.metadata["provider_family"],
            UsageProviderFamily.LOCAL.value,
        )
        with self.assertRaises(TypeError):
            cast(dict[str, object], record.metadata)["raw"] = "leak"
        with self.assertRaises(AssertionError):
            UsageTotals(input_tokens=-1)
        with self.assertRaises(AssertionError):
            UsageTotals(total_tokens=True)
        with self.assertRaises(AssertionError):
            UsageRecord(
                usage_id="usage-2",
                run_id="run-1",
                attempt_id=None,
                sequence=0,
                source=UsageSource.UNAVAILABLE,
                totals=UsageTotals(),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_usage_metadata_rejects_unsafe_values(self) -> None:
        empty = freeze_usage_metadata(None)
        metadata = freeze_usage_metadata(
            {
                "provider_family": "azure_openai",
                "cache_creation_ephemeral_5m_input_tokens": 3,
                "cache_creation_ephemeral_1h_input_tokens": 4,
                "cache_read_ephemeral_5m_input_tokens": 5,
                "cache_read_ephemeral_1h_input_tokens": 6,
                "cacheCreationEphemeral5mInputTokens": 99,
                "cache_read_ephemeral_5m_input_tokens_bad": 7,
                "cache_read_ephemeral_1h_input_tokens_bad": -1,
                "raw_model_id": "private-deployment",
                "headers": {"authorization": "private-token"},
                "request_id": "private-response-id",
                "unknown": object(),
            }
        )
        invalid_metadata = freeze_usage_metadata(
            {"provider_family": "private-provider-name"}
        )
        frozen = freeze_usage_value({"counts": [1, 2]})
        finite_float = freeze_usage_value(1.5)

        self.assertEqual(empty, {})
        self.assertEqual(
            metadata,
            {
                "provider_family": UsageProviderFamily.AZURE_OPENAI.value,
                "cache_creation_ephemeral_5m_input_tokens": 3,
                "cache_creation_ephemeral_1h_input_tokens": 4,
                "cache_read_ephemeral_5m_input_tokens": 5,
                "cache_read_ephemeral_1h_input_tokens": 6,
            },
        )
        self.assertEqual(invalid_metadata, {})
        self.assertIsNone(freeze_usage_value(None))
        self.assertTrue(freeze_usage_value(True))
        self.assertEqual(freeze_usage_value("safe"), "safe")
        self.assertEqual(cast(Mapping[str, object], frozen)["counts"], (1, 2))
        self.assertEqual(finite_float, 1.5)
        with self.assertRaises(AssertionError):
            freeze_usage_metadata(cast(Mapping[str, object], []))
        with self.assertRaises(AssertionError):
            freeze_usage_value({"raw": object()})
        with self.assertRaises(AssertionError):
            freeze_usage_value(float("inf"))
        with self.assertRaises(AssertionError):
            freeze_usage_value({"": "empty"})

    def test_unknown_usage_counter_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            usage_module._counter_value(  # noqa: SLF001
                UsageTotals(),
                "unknown",
            )

    def test_stable_usage_id_hashes_call_key(self) -> None:
        usage_id = stable_usage_id(
            run_id="run-1",
            attempt_id="attempt-1",
            call_key="private-provider-response-id",
        )
        same_usage_id = stable_usage_id(
            run_id="run-1",
            attempt_id="attempt-1",
            call_key="private-provider-response-id",
        )
        other_call_usage_id = stable_usage_id(
            run_id="run-1",
            attempt_id="attempt-1",
            call_key="other-call",
        )
        other_attempt_usage_id = stable_usage_id(
            run_id="run-1",
            attempt_id="attempt-2",
            call_key="private-provider-response-id",
        )

        self.assertEqual(usage_id, same_usage_id)
        self.assertNotEqual(usage_id, other_call_usage_id)
        self.assertNotEqual(usage_id, other_attempt_usage_id)
        self.assertTrue(usage_id.startswith("usage-"))
        self.assertNotIn("private-provider-response-id", usage_id)
        with self.assertRaises(AssertionError):
            stable_usage_id(run_id="run-1", attempt_id=None, call_key="")

    def test_stable_usage_id_for_response_hashes_explicit_key(self) -> None:
        response = SimpleNamespace(
            usage_call_key="private-provider-response-id",
            usage={"input_tokens": 1},
        )

        usage_id = stable_usage_id_for_response(
            response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        same_usage_id = stable_usage_id_for_response(
            response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        next_usage_id = stable_usage_id_for_response(
            response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=2,
        )

        self.assertEqual(usage_id, same_usage_id)
        self.assertNotEqual(usage_id, next_usage_id)
        self.assertTrue(usage_id.startswith("usage-"))
        self.assertNotIn("private-provider-response-id", usage_id)
        with self.assertRaises(AssertionError):
            stable_usage_id_for_response(
                response,
                run_id="run-1",
                attempt_id="attempt-1",
                sequence=0,
            )

    def test_stable_usage_id_for_response_hashes_mapping_key(self) -> None:
        response = {
            "usage_call_key": "private-provider-response-id",
            "usage": {"input_tokens": 1},
        }
        replayed = {
            "usage_call_key": "private-provider-response-id",
            "usage": {"input_tokens": 1},
        }

        usage_id = stable_usage_id_for_response(
            response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        replayed_usage_id = stable_usage_id_for_response(
            replayed,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )

        self.assertEqual(usage_id, replayed_usage_id)
        self.assertNotIn("private-provider-response-id", usage_id)

    def test_stable_usage_id_for_response_keeps_exact_calls_distinct(
        self,
    ) -> None:
        first = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
                "raw_response_id": "private-response-one",
            },
        )
        replayed = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
                "raw_response_id": "private-response-two",
            },
        )
        different = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": 3,
                "output_tokens": 6,
                "total_tokens": 9,
                "raw_response_id": "private-response-three",
            },
        )
        malformed = SimpleNamespace(usage={"input_tokens": "private prompt"})

        usage_id = stable_usage_id_for_response(
            first,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        replayed_usage_id = stable_usage_id_for_response(
            replayed,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        different_usage_id = stable_usage_id_for_response(
            different,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        malformed_usage_id = stable_usage_id_for_response(
            malformed,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        same_malformed_usage_id = stable_usage_id_for_response(
            malformed,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )

        self.assertNotEqual(usage_id, replayed_usage_id)
        self.assertNotEqual(usage_id, different_usage_id)
        self.assertEqual(malformed_usage_id, same_malformed_usage_id)
        self.assertNotEqual(usage_id, malformed_usage_id)
        self.assertNotIn("private-response-one", usage_id)
        self.assertNotIn("private-response-two", usage_id)

    def test_stable_usage_id_for_response_deduplicates_explicit_replay(
        self,
    ) -> None:
        first = SimpleNamespace(
            usage_call_key="private-provider-call",
            provider_family="openai",
            usage={"input_tokens": 3, "output_tokens": 5},
        )
        replayed = SimpleNamespace(
            usage_call_key="private-provider-call",
            provider_family="openai",
            usage={"input_tokens": 3, "output_tokens": 5},
        )

        usage_id = stable_usage_id_for_response(
            first,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        replayed_usage_id = stable_usage_id_for_response(
            replayed,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )

        self.assertEqual(usage_id, replayed_usage_id)
        self.assertNotIn("private-provider-call", usage_id)

    def test_stable_usage_id_for_response_keeps_estimated_calls_distinct(
        self,
    ) -> None:
        first = SimpleNamespace(input_token_count=3, output_token_count=5)
        second = SimpleNamespace(input_token_count=3, output_token_count=5)

        first_usage_id = stable_usage_id_for_response(
            first,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        second_usage_id = stable_usage_id_for_response(
            second,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        same_first_usage_id = stable_usage_id_for_response(
            first,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )

        self.assertNotEqual(first_usage_id, second_usage_id)
        self.assertEqual(first_usage_id, same_first_usage_id)

    def test_stable_usage_id_for_response_tracks_weak_objects(
        self,
    ) -> None:
        first = WeakUsageResponse()
        second = WeakUsageResponse()

        first_usage_id = stable_usage_id_for_response(
            first,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        same_first_usage_id = stable_usage_id_for_response(
            first,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        second_usage_id = stable_usage_id_for_response(
            second,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )

        self.assertEqual(first_usage_id, same_first_usage_id)
        self.assertNotEqual(first_usage_id, second_usage_id)

    def test_stable_usage_id_for_response_handles_untracked_objects(
        self,
    ) -> None:
        response = ("private", "tuple")
        usage_id = stable_usage_id_for_response(
            response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )

        self.assertTrue(usage_id.startswith("usage-"))
        self.assertNotIn("private", usage_id)

    def test_stable_usage_id_for_response_deduplicates_untracked_replay(
        self,
    ) -> None:
        response = {
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "raw_response_id": "private-response-id",
            }
        }
        other_response = {
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "raw_response_id": "private-response-id",
            }
        }

        usage_id = stable_usage_id_for_response(
            response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        replayed_usage_id = stable_usage_id_for_response(
            response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )
        other_usage_id = stable_usage_id_for_response(
            other_response,
            run_id="run-1",
            attempt_id="attempt-1",
            sequence=1,
        )

        self.assertEqual(usage_id, replayed_usage_id)
        self.assertNotEqual(usage_id, other_usage_id)
        self.assertTrue(usage_id.startswith("usage-"))
        self.assertNotIn("private-response-id", usage_id)

    def test_stable_usage_id_for_response_handles_untracked_id_collision(
        self,
    ) -> None:
        first_response = {
            "usage": {
                "input_tokens": 3,
                "raw_response_id": "private-first-response",
            }
        }
        second_response = {
            "usage": {
                "input_tokens": 5,
                "raw_response_id": "private-second-response",
            }
        }

        with patch.object(usage_module, "id", return_value=7, create=True):
            first_usage_id = stable_usage_id_for_response(
                first_response,
                run_id="run-1",
                attempt_id="attempt-1",
                sequence=1,
            )
            replayed_first_usage_id = stable_usage_id_for_response(
                first_response,
                run_id="run-1",
                attempt_id="attempt-1",
                sequence=1,
            )
            second_usage_id = stable_usage_id_for_response(
                second_response,
                run_id="run-1",
                attempt_id="attempt-1",
                sequence=1,
            )

        self.assertEqual(first_usage_id, replayed_first_usage_id)
        self.assertNotEqual(first_usage_id, second_usage_id)
        self.assertNotIn("private-first-response", first_usage_id)
        self.assertNotIn("private-second-response", second_usage_id)


class UsageStoreTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryTaskStore(
            clock=SequenceClock(),
            id_factory=SequenceIds(),
        )
        await self.store.register_definition(
            definition(),
            definition_hash="hash-a",
        )
        self.run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        self.attempt = await self.store.create_attempt(self.run.run_id)

    async def test_response_callback_records_usage_after_consumption(
        self,
    ) -> None:
        response = FakeConsumedResponse()
        attached = attach_response_usage_recorder(
            response,
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            metadata={"provider_family": UsageProviderFamily.LOCAL},
        )

        self.assertTrue(attached)
        self.assertEqual(await self.store.list_usage(self.run.run_id), ())

        await response.consume()
        await response.consume()

        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].sequence, 1)
        self.assertEqual(records[0].source, UsageSource.ESTIMATED)
        self.assertEqual(records[0].attempt_id, self.attempt.attempt_id)
        self.assertEqual(records[0].totals.input_tokens, 3)
        self.assertEqual(records[0].totals.output_tokens, 4)
        self.assertIsNone(records[0].totals.total_tokens)
        self.assertEqual(
            records[0].metadata["provider_family"],
            UsageProviderFamily.LOCAL.value,
        )

    async def test_response_callback_uses_explicit_single_usage_id(
        self,
    ) -> None:
        response = FakeConsumedResponse()

        self.assertTrue(
            attach_response_usage_recorder(
                response,
                store=self.store,
                run_id=self.run.run_id,
                attempt_id=self.attempt.attempt_id,
                usage_id="usage-explicit",
            )
        )
        await response.consume()

        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].usage_id, "usage-explicit")
        self.assertEqual(records[0].totals.input_tokens, 3)

    async def test_response_callback_splits_explicit_multi_usage_id(
        self,
    ) -> None:
        response = FakeConsumedResponse(input_token_count=None)
        response.usage = (
            {"input_tokens": 1},
            {"output_tokens": 2},
        )

        self.assertTrue(
            attach_response_usage_recorder(
                response,
                store=self.store,
                run_id=self.run.run_id,
                attempt_id=self.attempt.attempt_id,
                usage_id="private-explicit-id",
            )
        )
        await response.consume()

        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0].usage_id, records[1].usage_id)
        self.assertNotIn("private-explicit-id", records[0].usage_id)
        self.assertNotIn("private-explicit-id", records[1].usage_id)
        self.assertEqual(records[0].totals.input_tokens, 1)
        self.assertEqual(records[1].totals.output_tokens, 2)

    async def test_usage_records_are_filtered_and_aggregated(self) -> None:
        other_attempt = await self._failed_attempt_then_new_attempt()
        await self.store.append_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
            ),
        )
        await self.store.append_usage(
            self.run.run_id,
            attempt_id=other_attempt.attempt_id,
            source=UsageSource.ESTIMATED,
            totals=UsageTotals(output_tokens=4),
        )

        first_attempt_records = await self.store.list_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )
        totals = await self.store.usage_totals(self.run.run_id)

        self.assertEqual(len(first_attempt_records), 1)
        self.assertEqual(totals.input_tokens, 2)
        self.assertEqual(totals.output_tokens, 7)
        self.assertEqual(totals.total_tokens, 5)
        self.assertIsNone(totals.cached_input_tokens)

    async def test_provider_response_callback_records_exact_usage(
        self,
    ) -> None:
        response = FakeProviderResponse()

        self.assertTrue(
            attach_response_usage_recorder(
                response,
                store=self.store,
                run_id=self.run.run_id,
                attempt_id=self.attempt.attempt_id,
            )
        )
        await response.consume()

        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, UsageSource.EXACT)
        self.assertEqual(records[0].totals.cached_input_tokens, 1)
        self.assertEqual(records[0].totals.reasoning_tokens, 5)
        self.assertEqual(records[0].totals.total_tokens, 99)
        self.assertEqual(
            records[0].metadata["provider_family"],
            UsageProviderFamily.OPENAI.value,
        )
        self.assertEqual(
            records[0].metadata["cache_creation_ephemeral_5m_input_tokens"],
            6,
        )
        self.assertEqual(
            records[0].metadata["cache_creation_ephemeral_1h_input_tokens"],
            8,
        )

    async def test_callback_records_nested_usage_without_private_replay(
        self,
    ) -> None:
        first_response = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": 2,
                "cached_input_tokens": 1,
                "output_tokens": 3,
                "total_tokens": 5,
                "raw_response_id": "private-first-response",
            },
        )
        second_response = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": 4,
                "cache_creation_input_tokens": 2,
                "output_tokens": 6,
                "reasoning_tokens": 1,
                "total_tokens": 10,
                "raw_response_id": "private-second-response",
            },
        )
        malformed_response = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": "private prompt",
                "output_tokens": -1,
                "total_tokens": True,
                "raw_response_body": "private provider body",
            },
        )
        response = FakeUsageWrapperResponse(
            first_response,
            second_response,
            malformed_response,
        )

        self.assertTrue(
            attach_response_usage_recorder(
                response,
                store=self.store,
                run_id=self.run.run_id,
                attempt_id=self.attempt.attempt_id,
            )
        )
        await response.consume()
        await response.consume()

        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)
        self.assertEqual(len(records), 2)
        self.assertEqual([record.sequence for record in records], [1, 2])
        self.assertEqual(records[0].totals.input_tokens, 2)
        self.assertEqual(records[0].totals.cached_input_tokens, 1)
        self.assertEqual(records[1].totals.input_tokens, 4)
        self.assertEqual(records[1].totals.cache_creation_input_tokens, 2)
        self.assertEqual(records[1].totals.reasoning_tokens, 1)
        self.assertEqual(totals.input_tokens, 6)
        self.assertEqual(totals.output_tokens, 9)
        self.assertEqual(totals.total_tokens, 15)
        self.assertNotIn("private-first-response", str(records))
        self.assertNotIn("private-second-response", str(records))
        self.assertNotIn("private provider body", str(records))

    async def test_record_response_usage_deduplicates_mapping_replay(
        self,
    ) -> None:
        response = MultiCallUsageResponse(
            usage=(
                {
                    "usage_call_key": "private-provider-call",
                    "input_tokens": 3,
                    "output_tokens": 5,
                    "total_tokens": 8,
                },
            )
        )
        replayed = MultiCallUsageResponse(
            usage=(
                {
                    "usage_call_key": "private-provider-call",
                    "input_tokens": 3,
                    "output_tokens": 5,
                    "total_tokens": 8,
                },
            )
        )

        await record_response_usage(
            None,
            store=self.store,
            response=response,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )
        await record_response_usage(
            None,
            store=self.store,
            response=replayed,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )

        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, UsageSource.EXACT)
        self.assertEqual(records[0].totals.input_tokens, 3)
        self.assertEqual(totals.total_tokens, 8)
        self.assertNotIn("private-provider-call", records[0].usage_id)

    async def test_stable_usage_id_prevents_duplicate_records(self) -> None:
        usage_id = stable_usage_id(
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            call_key="model-call-1",
        )

        first = await self.store.append_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            usage_id=usage_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=1),
        )
        duplicate = await self.store.append_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            usage_id=usage_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=99, total_tokens=99),
        )
        distinct = await self.store.append_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            usage_id=stable_usage_id(
                run_id=self.run.run_id,
                attempt_id=self.attempt.attempt_id,
                call_key="model-call-2",
            ),
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=2),
        )

        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)

        self.assertEqual(first, duplicate)
        self.assertNotEqual(first.usage_id, distinct.usage_id)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].sequence, 1)
        self.assertEqual(records[1].sequence, 2)
        self.assertEqual(totals.input_tokens, 3)
        self.assertIsNone(totals.total_tokens)

    async def test_stable_usage_id_conflict_rejects_other_run(
        self,
    ) -> None:
        usage_id = stable_usage_id(
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            call_key="model-call-1",
        )
        other_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )

        await self.store.append_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            usage_id=usage_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=1),
        )

        with self.assertRaises(TaskStoreConflictError):
            await self.store.append_usage(
                other_run.run_id,
                usage_id=usage_id,
                source=UsageSource.EXACT,
                totals=UsageTotals(input_tokens=1),
            )

    async def test_generated_usage_id_collision_is_rejected(self) -> None:
        store = InMemoryTaskStore(
            clock=SequenceClock(),
            id_factory=FixedIds("run-1", "attempt-1", "usage-1", "usage-1"),
        )
        await store.register_definition(
            definition(),
            definition_hash="hash-generated-collision",
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-generated-collision")
        )
        attempt = await store.create_attempt(run.run_id)
        await store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=1),
        )

        with self.assertRaises(TaskStoreConflictError):
            await store.append_usage(
                run.run_id,
                attempt_id=attempt.attempt_id,
                source=UsageSource.EXACT,
                totals=UsageTotals(input_tokens=2),
            )

    async def test_callback_is_not_attached_without_response_support(
        self,
    ) -> None:
        self.assertFalse(
            attach_response_usage_recorder(
                object(),
                store=self.store,
                run_id=self.run.run_id,
            )
        )

    async def test_callback_with_unavailable_counters_records_nothing(
        self,
    ) -> None:
        response = FakeConsumedResponse(
            input_token_count=None,
            output_token_count=None,
        )

        self.assertTrue(
            attach_response_usage_recorder(
                response,
                store=self.store,
                run_id=self.run.run_id,
                attempt_id=self.attempt.attempt_id,
            )
        )
        await response.consume()

        self.assertEqual(await self.store.list_usage(self.run.run_id), ())

    async def test_usage_attempt_must_belong_to_run(self) -> None:
        other_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        other_attempt = await self.store.create_attempt(other_run.run_id)

        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.append_usage(
                self.run.run_id,
                attempt_id=other_attempt.attempt_id,
                source=UsageSource.EXACT,
                totals=UsageTotals(input_tokens=1),
            )
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_usage(
                self.run.run_id,
                attempt_id=other_attempt.attempt_id,
            )

    async def _failed_attempt_then_new_attempt(self) -> TaskAttempt:
        await self.store.transition_attempt(
            self.attempt.attempt_id,
            from_states={TaskAttemptState.CREATED},
            to_state=TaskAttemptState.RUNNING,
            reason="started",
        )
        await self.store.transition_attempt(
            self.attempt.attempt_id,
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="failed",
        )
        return await self.store.create_attempt(self.run.run_id)


if __name__ == "__main__":
    main()
