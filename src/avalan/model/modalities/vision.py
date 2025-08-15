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
from ...entities import (
    EngineUri,
    GenerationSettings,
    Input,
    Modality,
    Operation,
    OperationParameters,
    OperationVisionParameters,
    TransformerEngineSettings,
)
from ...tool.manager import ToolManager

from argparse import Namespace
from logging import Logger
from typing import Any


@ModalityRegistry.register(Modality.VISION_ENCODER_DECODER)
class VisionEncoderDecoderModality:
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> VisionEncoderDecoderModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return VisionEncoderDecoderModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            vision=OperationVisionParameters(
                path=args.path,
                skip_special_tokens=args.skip_special_tokens,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.VISION_ENCODER_DECODER,
            parameters=parameters,
            requires_input=False,
        )

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
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> ImageClassificationModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return ImageClassificationModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            vision=OperationVisionParameters(
                path=args.path,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.VISION_IMAGE_CLASSIFICATION,
            parameters=parameters,
            requires_input=False,
        )

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
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> ImageToTextModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return ImageToTextModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            vision=OperationVisionParameters(
                path=args.path,
                skip_special_tokens=args.skip_special_tokens,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.VISION_IMAGE_TO_TEXT,
            parameters=parameters,
            requires_input=False,
        )

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
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> ImageTextToTextModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return ImageTextToTextModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            vision=OperationVisionParameters(
                path=args.path,
                system_prompt=args.system or None,
                width=getattr(
                    args,
                    "vision_width",
                    getattr(args, "image_width", None),
                ),
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.VISION_IMAGE_TEXT_TO_TEXT,
            parameters=parameters,
            requires_input=True,
        )

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
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> ObjectDetectionModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return ObjectDetectionModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            vision=OperationVisionParameters(
                path=args.path,
                threshold=getattr(
                    args,
                    "vision_threshold",
                    getattr(args, "image_threshold", None),
                ),
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.VISION_OBJECT_DETECTION,
            parameters=parameters,
            requires_input=False,
        )

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
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> TextToImageModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return TextToImageModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            vision=OperationVisionParameters(
                path=args.path,
                color_model=args.vision_color_model,
                high_noise_frac=args.vision_high_noise_frac,
                image_format=args.vision_image_format,
                n_steps=args.vision_steps,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.VISION_TEXT_TO_IMAGE,
            parameters=parameters,
            requires_input=True,
        )

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
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> TextToAnimationModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return TextToAnimationModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            vision=OperationVisionParameters(
                path=args.path,
                n_steps=args.vision_steps,
                timestep_spacing=args.vision_timestep_spacing,
                beta_schedule=args.vision_beta_schedule,
                guidance_scale=args.vision_guidance_scale,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.VISION_TEXT_TO_ANIMATION,
            parameters=parameters,
            requires_input=True,
        )

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
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> TextToVideoModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return TextToVideoModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            vision=OperationVisionParameters(
                path=args.path,
                reference_path=getattr(args, "vision_reference_path", None),
                negative_prompt=getattr(args, "vision_negative_prompt", None),
                width=getattr(args, "vision_width", None),
                height=getattr(args, "vision_height", None),
                downscale=getattr(args, "vision_downscale", None),
                frames=getattr(args, "vision_frames", None),
                denoise_strength=getattr(
                    args, "vision_denoise_strength", None
                ),
                n_steps=getattr(args, "vision_steps", None),
                inference_steps=getattr(args, "vision_inference_steps", None),
                decode_timestep=getattr(args, "vision_decode_timestep", None),
                noise_scale=getattr(args, "vision_noise_scale", None),
                frames_per_second=getattr(args, "vision_fps", None),
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.VISION_TEXT_TO_VIDEO,
            parameters=parameters,
            requires_input=True,
        )

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
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> SemanticSegmentationModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return SemanticSegmentationModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            vision=OperationVisionParameters(
                path=args.path,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.VISION_SEMANTIC_SEGMENTATION,
            parameters=parameters,
            requires_input=False,
        )

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
