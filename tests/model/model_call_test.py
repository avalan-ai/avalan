from typing import cast
from unittest import TestCase

from avalan.agent import Specification
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Modality,
    Operation,
    OperationParameters,
)
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.capability import ModelCapabilityCatalog
from avalan.model.engine import Engine


class ModelCallCapabilityIdentityTestCase(TestCase):
    def _call(
        self,
        *,
        capability: ModelCapabilityCatalog | None,
        context: ModelCallContext,
    ) -> ModelCall:
        return ModelCall(
            engine_uri=EngineUri(
                host=None,
                port=None,
                user=None,
                password=None,
                vendor=None,
                model_id="model",
                params={},
            ),
            model=cast(Engine, object()),
            operation=Operation(
                generation_settings=GenerationSettings(),
                input=None,
                modality=Modality.TEXT_GENERATION,
                parameters=OperationParameters(),
            ),
            capability=capability,
            context=context,
        )

    @staticmethod
    def _context(
        capability: ModelCapabilityCatalog | None = None,
    ) -> ModelCallContext:
        return ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=None,
            capability=capability,
        )

    def test_derives_call_capability_from_context(self) -> None:
        capability = ModelCapabilityCatalog.create()
        context = self._context(capability)

        call = self._call(capability=None, context=context)

        self.assertIs(call.capability, capability)
        self.assertIs(call.context, context)

    def test_normalizes_context_to_call_capability(self) -> None:
        capability = ModelCapabilityCatalog.create()
        context = self._context()

        call = self._call(capability=capability, context=context)

        self.assertIs(call.capability, capability)
        self.assertIs(call.context.capability, capability)
        self.assertEqual(call.context.specification, context.specification)

    def test_rejects_distinct_catalog_instances(self) -> None:
        with self.assertRaisesRegex(AssertionError, "must be identical"):
            self._call(
                capability=ModelCapabilityCatalog.create(),
                context=self._context(ModelCapabilityCatalog.create()),
            )
