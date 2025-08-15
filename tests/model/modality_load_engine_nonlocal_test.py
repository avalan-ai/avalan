from logging import Logger
from unittest.mock import MagicMock

import pytest

from avalan.entities import EngineUri, TransformerEngineSettings
from avalan.model.modalities.audio import (
    AudioSpeechRecognitionModality,
    AudioTextToSpeechModality,
)
from avalan.model.modalities.text import (
    TextGenerationModality,
    TextSequenceClassificationModality,
    TextTokenClassificationModality,
)
from avalan.model.modalities.vision import (
    VisionEncoderDecoderModality,
    VisionImageClassificationModality,
    VisionImageTextToTextModality,
    VisionImageToTextModality,
    VisionObjectDetectionModality,
    VisionSemanticSegmentationModality,
    VisionTextToAnimationModality,
    VisionTextToImageModality,
    VisionTextToVideoModality,
)


@pytest.mark.parametrize(
    "modality",
    [
        AudioSpeechRecognitionModality(),
        AudioTextToSpeechModality(),
        TextGenerationModality(),
        TextSequenceClassificationModality(),
        TextTokenClassificationModality(),
        VisionEncoderDecoderModality(),
        VisionImageClassificationModality(),
        VisionImageToTextModality(),
        VisionImageTextToTextModality(),
        VisionObjectDetectionModality(),
        VisionTextToImageModality(),
        VisionTextToAnimationModality(),
        VisionTextToVideoModality(),
        VisionSemanticSegmentationModality(),
    ],
)
def test_load_engine_non_local_raises(modality):
    engine_uri = EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor="google",
        model_id="model",
        params={},
    )
    settings = TransformerEngineSettings()
    logger = MagicMock(spec=Logger)

    with pytest.raises(NotImplementedError):
        modality.load_engine(engine_uri, settings, logger)
