from argparse import Namespace

import pytest

from avalan.entities import Modality, ReasoningEffort, ReasoningTag
from avalan.model.modalities import ModalityRegistry


def test_registry_contains_all_handlers():
    for modality in Modality:
        if modality is Modality.EMBEDDING:
            with pytest.raises(NotImplementedError):
                ModalityRegistry.get(modality)
        else:
            handler = ModalityRegistry.get(modality)
            assert callable(handler)


def test_get_operation_from_arguments_maps_reasoning_effort():
    args = Namespace(
        cache_strategy=None,
        chat_disable_thinking=True,
        do_sample=False,
        enable_gradient_calculation=False,
        max_new_tokens=32,
        min_p=None,
        no_reasoning=False,
        reasoning_effort="xhigh",
        reasoning_max_new_tokens=12,
        reasoning_stop_on_max_new_tokens=True,
        reasoning_tag="think",
        repetition_penalty=1.0,
        temperature=0.25,
        top_k=4,
        top_p=0.9,
        use_cache=False,
    )

    operation = ModalityRegistry.get_operation_from_arguments(
        Modality.EMBEDDING, args, "hi"
    )

    reasoning = operation.generation_settings.reasoning
    assert reasoning is not None
    assert reasoning.effort == ReasoningEffort.XHIGH
    assert reasoning.max_new_tokens == 12
    assert reasoning.stop_on_max_new_tokens is True
    assert reasoning.tag == ReasoningTag.THINK
    assert operation.generation_settings.chat_settings.enable_thinking is False
    assert operation.input == "hi"
