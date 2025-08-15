import pytest

from avalan.entities import Modality
from avalan.model.modalities import ModalityRegistry


def test_registry_contains_all_handlers():
    for modality in Modality:
        if modality is Modality.EMBEDDING:
            with pytest.raises(NotImplementedError):
                ModalityRegistry.get(modality)
        else:
            handler = ModalityRegistry.get(modality)
            assert callable(handler)
