from contextlib import nullcontext
from logging import Logger
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import ANY, MagicMock, call, patch

from avalan.entities import EngineSettings
from avalan.model.engine import Engine
from avalan.model.vision.diffusion import TextToVideoModel
from avalan.model.vision.diffusion import video as diffusion_video


class DummyDiffusionPipeline:
    def to(self, *args: object, **kwargs: object) -> object:
        return self


class TextToVideoModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"
    upsampler_id = "up/model"

    def test_lazy_diffusers_helpers_import_targets(self) -> None:
        pipeline = object()
        condition = object()

        def export_to_video(*args: object, **kwargs: object) -> object:
            return ("export", args, kwargs)

        def load_image(*args: object, **kwargs: object) -> object:
            return ("image", args, kwargs)

        def load_video(*args: object, **kwargs: object) -> object:
            return ("video", args, kwargs)

        modules = {
            "diffusers": SimpleNamespace(DiffusionPipeline=pipeline),
            "diffusers.pipelines.ltx.pipeline_ltx_condition": SimpleNamespace(
                LTXVideoCondition=condition
            ),
            "diffusers.utils": SimpleNamespace(
                export_to_video=export_to_video,
                load_image=load_image,
                load_video=load_video,
            ),
        }

        with patch(
            "avalan.model.vision.diffusion.video.import_module",
            side_effect=modules.__getitem__,
        ):
            self.assertIs(diffusion_video._diffusion_pipeline(), pipeline)
            self.assertIs(diffusion_video._ltx_video_condition(), condition)
            self.assertEqual(
                diffusion_video.export_to_video("frame", fps=12),
                ("export", ("frame",), {"fps": 12}),
            )
            self.assertEqual(
                diffusion_video.load_image("image.png"),
                ("image", ("image.png",), {}),
            )
            self.assertEqual(
                diffusion_video.load_video("video.mp4"),
                ("video", ("video.mp4",), {}),
            )

    def test_instantiation_with_load_model(self) -> None:
        logger_mock = MagicMock(spec=Logger)
        pipeline = MagicMock()
        with (
            patch(
                "avalan.model.vision.diffusion.video._diffusion_pipeline",
                return_value=pipeline,
            ),
            patch(
                "avalan.model.engine.DiffusionPipeline",
                DummyDiffusionPipeline,
            ),
            patch.object(Engine, "weight", return_value="dtype"),
            patch.object(Engine, "get_default_device", return_value="cpu"),
        ):
            pipeline_mock = pipeline.from_pretrained
            base_instance = MagicMock(spec=DummyDiffusionPipeline)
            base_instance.to.return_value = base_instance
            base_instance.vae = MagicMock()
            base_instance.vae.enable_tiling = MagicMock()
            upsampler_instance = MagicMock(spec=DummyDiffusionPipeline)
            upsampler_instance.to.return_value = upsampler_instance
            pipeline_mock.side_effect = [base_instance, upsampler_instance]

            settings = EngineSettings(upsampler_model_id=self.upsampler_id)
            model = TextToVideoModel(
                self.model_id, settings, logger=logger_mock
            )

            self.assertIs(model.model, base_instance)
            pipeline_mock.assert_has_calls(
                [
                    call(self.model_id, torch_dtype="dtype"),
                    call(
                        self.upsampler_id,
                        vae=base_instance.vae,
                        torch_dtype="dtype",
                    ),
                ]
            )
            base_instance.to.assert_called_once_with("cpu")
            upsampler_instance.to.assert_called_once_with("cpu")
            base_instance.vae.enable_tiling.assert_called_once_with()


class TextToVideoModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"
    upsampler_id = "up/model"

    async def test_call(self) -> None:
        logger_mock = MagicMock(spec=Logger)
        pipeline = MagicMock()
        with (
            patch(
                "avalan.model.vision.diffusion.video._diffusion_pipeline",
                return_value=pipeline,
            ),
            patch(
                "avalan.model.engine.DiffusionPipeline",
                DummyDiffusionPipeline,
            ),
            patch.object(Engine, "weight", return_value="dtype"),
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch(
                "avalan.model.vision.diffusion.video.load_image"
            ) as load_image_mock,
            patch(
                "avalan.model.vision.diffusion.video.load_video",
                return_value="vid",
            ),
            patch(
                "avalan.model.vision.diffusion.video.export_to_video"
            ) as export_mock,
            patch(
                "avalan.model.vision.diffusion.video.inference_mode",
                return_value=nullcontext(),
            ),
            patch(
                "avalan.model.vision.diffusion.video._ltx_video_condition"
            ) as cond_cls,
        ):
            pipeline_mock = pipeline.from_pretrained
            base_instance = MagicMock(spec=DummyDiffusionPipeline)
            base_instance.to.return_value = base_instance
            base_instance.vae = MagicMock()
            base_instance.vae_spatial_compression_ratio = 2
            first_out = MagicMock()
            first_out.frames = ["latents"]
            second_out = MagicMock()
            frame = MagicMock()
            second_out.frames = [[frame]]
            base_instance.side_effect = [first_out, second_out]
            upsampler_instance = MagicMock(spec=DummyDiffusionPipeline)
            upsampler_instance.to.return_value = upsampler_instance
            upsampler_instance.return_value = MagicMock(frames=["upscaled"])
            pipeline_mock.side_effect = [base_instance, upsampler_instance]

            condition_class = MagicMock()
            cond_instance = MagicMock()
            condition_class.return_value = cond_instance
            cond_cls.return_value = condition_class
            load_image_mock.return_value = "img"

            settings = EngineSettings(upsampler_model_id=self.upsampler_id)
            model = TextToVideoModel(
                self.model_id, settings, logger=logger_mock
            )

            result = await model(
                "prompt",
                "neg",
                "ref.png",
                "out.mp4",
                width=512,
                height=480,
            )

            self.assertEqual(result, "out.mp4")
            load_image_mock.assert_called_once_with("ref.png")
            export_mock.assert_called_with(
                [frame.resize.return_value], "out.mp4", fps=24
            )
            base_instance.assert_has_calls(
                [
                    call(
                        conditions=[cond_instance],
                        prompt="prompt",
                        negative_prompt="neg",
                        width=340,
                        height=320,
                        num_frames=96,
                        num_inference_steps=30,
                        generator=ANY,
                        output_type="latent",
                    ),
                    call(
                        conditions=[cond_instance],
                        prompt="prompt",
                        negative_prompt="neg",
                        width=680,
                        height=640,
                        num_frames=96,
                        denoise_strength=0.4,
                        num_inference_steps=10,
                        latents=upsampler_instance.return_value.frames,
                        decode_timestep=0.05,
                        image_cond_noise_scale=0.025,
                        generator=ANY,
                        output_type="pil",
                    ),
                ]
            )
            upsampler_instance.assert_called_once_with(
                latents=first_out.frames, output_type="latent"
            )


class TextToVideoModelRoundResolutionTestCase(TestCase):
    def test_round_to_nearest_resolution(self) -> None:
        self.assertEqual(
            TextToVideoModel._round_to_nearest_resolution_acceptable_by_vae(
                5, 7, 2
            ),
            (4, 6),
        )


if __name__ == "__main__":
    main()
