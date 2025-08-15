from .registry import ModalityRegistry
from ..vision.classification import ImageClassificationModel
from ..vision.decoder import VisionEncoderDecoderModel
from ..vision.detection import ObjectDetectionModel
from ..vision.diffusion import (
    TextToAnimationModel,
    TextToImageModel,
    TextToVideoModel,
)
from ..vision.segmentation import SemanticSegmentationModel
from ..vision.text import ImageTextToTextModel, ImageToTextModel
from ...entities import EngineUri, Modality, Operation
from ...tool.manager import ToolManager

from typing import Any


@ModalityRegistry.register(Modality.VISION_ENCODER_DECODER)
class VisionEncoderDecoderModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: VisionEncoderDecoderModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.parameters["vision"]
            and operation.parameters["vision"].path
        )

        return await model(
            operation.parameters["vision"].path,
            prompt=operation.input,
            skip_special_tokens=operation.parameters[
                "vision"
            ].skip_special_tokens,
        )


@ModalityRegistry.register(Modality.VISION_IMAGE_CLASSIFICATION)
class VisionImageClassificationModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: ImageClassificationModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.parameters["vision"]
            and operation.parameters["vision"].path
        )

        return await model(operation.parameters["vision"].path)


@ModalityRegistry.register(Modality.VISION_IMAGE_TO_TEXT)
class VisionImageToTextModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: ImageToTextModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.parameters["vision"]
            and operation.parameters["vision"].path
        )

        return await model(
            operation.parameters["vision"].path,
            skip_special_tokens=operation.parameters[
                "vision"
            ].skip_special_tokens,
        )


@ModalityRegistry.register(Modality.VISION_IMAGE_TEXT_TO_TEXT)
class VisionImageTextToTextModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: ImageTextToTextModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.parameters["vision"]
            and operation.parameters["vision"].path
        )

        return await model(
            operation.parameters["vision"].path,
            operation.input,
            system_prompt=operation.parameters["vision"].system_prompt,
            settings=operation.generation_settings,
            width=operation.parameters["vision"].width,
        )


@ModalityRegistry.register(Modality.VISION_OBJECT_DETECTION)
class VisionObjectDetectionModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: ObjectDetectionModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.parameters["vision"]
            and operation.parameters["vision"].path
            and operation.parameters["vision"].threshold is not None
        )

        return await model(
            operation.parameters["vision"].path,
            threshold=operation.parameters["vision"].threshold,
        )


@ModalityRegistry.register(Modality.VISION_TEXT_TO_IMAGE)
class VisionTextToImageModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: TextToImageModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.input
            and operation.parameters["vision"]
            and operation.parameters["vision"].path
            and operation.parameters["vision"].color_model
            and operation.parameters["vision"].high_noise_frac is not None
            and operation.parameters["vision"].image_format
            and operation.parameters["vision"].n_steps is not None
        )

        return await model(
            operation.input,
            operation.parameters["vision"].path,
            color_model=operation.parameters["vision"].color_model,
            high_noise_frac=operation.parameters["vision"].high_noise_frac,
            image_format=operation.parameters["vision"].image_format,
            n_steps=operation.parameters["vision"].n_steps,
        )


@ModalityRegistry.register(Modality.VISION_TEXT_TO_ANIMATION)
class VisionTextToAnimationModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: TextToAnimationModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.input
            and operation.parameters["vision"]
            and operation.parameters["vision"].path
            and operation.parameters["vision"].n_steps is not None
            and operation.parameters["vision"].timestep_spacing
            and operation.parameters["vision"].beta_schedule
            and operation.parameters["vision"].guidance_scale is not None
        )

        return await model(
            operation.input,
            operation.parameters["vision"].path,
            beta_schedule=operation.parameters["vision"].beta_schedule,
            guidance_scale=operation.parameters["vision"].guidance_scale,
            steps=operation.parameters["vision"].n_steps,
            timestep_spacing=operation.parameters["vision"].timestep_spacing,
        )


@ModalityRegistry.register(Modality.VISION_TEXT_TO_VIDEO)
class VisionTextToVideoModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: TextToVideoModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.input
            and operation.parameters["vision"]
            and operation.parameters["vision"].path
        )
        vision = operation.parameters["vision"]
        kwargs = {
            "reference_path": vision.reference_path,
            "negative_prompt": vision.negative_prompt,
            "height": vision.height,
            "downscale": vision.downscale,
            "frames": vision.frames,
            "denoise_strength": vision.denoise_strength,
            "inference_steps": vision.inference_steps,
            "decode_timestep": vision.decode_timestep,
            "noise_scale": vision.noise_scale,
            "frames_per_second": vision.frames_per_second,
        }
        if vision.width is not None:
            kwargs["width"] = vision.width
        if vision.n_steps is not None:
            kwargs["steps"] = vision.n_steps

        return await model(operation.input, vision.path, **kwargs)


@ModalityRegistry.register(Modality.VISION_SEMANTIC_SEGMENTATION)
class VisionSemanticSegmentationModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: SemanticSegmentationModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.parameters["vision"]
            and operation.parameters["vision"].path
        )

        return await model(operation.parameters["vision"].path)
