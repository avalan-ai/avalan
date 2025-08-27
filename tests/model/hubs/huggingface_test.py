from avalan.entities import HubCache, HubCacheDeletion, Model, User
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.hubs import HubAccessDeniedException
from datetime import datetime
from huggingface_hub.errors import GatedRepoError
from logging import Logger
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch, call


class HuggingfaceHubTestCase(TestCase):
    def setUp(self):
        self.hf_patch = patch("avalan.model.hubs.huggingface.HfApi")
        self.login_patch = patch("avalan.model.hubs.huggingface.login")
        self.mock_HfApi = self.hf_patch.start()
        self.mock_login = self.login_patch.start()
        self.addCleanup(self.hf_patch.stop)
        self.addCleanup(self.login_patch.stop)
        self.hf_instance = MagicMock()
        self.mock_HfApi.return_value = self.hf_instance
        self.logger = MagicMock(spec=Logger)
        self.hub = HuggingfaceHub(
            access_token="token",
            cache_dir="/cache",
            logger=self.logger,
            endpoint="https://hf.co",
        )

    def test_cache_dir(self):
        self.assertEqual(self.hub.cache_dir, "/cache")

    def test_domain(self):
        self.assertEqual(self.hub.domain, "hf.co")

    def test_cache_delete(self):
        revision = SimpleNamespace(commit_hash="abc")
        info = SimpleNamespace(repo_id="model", revisions=[revision])
        strategy = SimpleNamespace(
            expected_freed_size=1.0,
            blobs=["b"],
            refs=["r"],
            repos=["p"],
            snapshots=["s"],
            execute=MagicMock(),
        )
        scan_result = SimpleNamespace(
            repos=[info],
            delete_revisions=MagicMock(return_value=strategy),
        )
        with patch(
            "avalan.model.hubs.huggingface.scan_cache_dir",
            return_value=scan_result,
        ) as scan_mock:
            deletion, execute = self.hub.cache_delete("model")
        scan_mock.assert_called_once_with("/cache")
        self.assertIsInstance(deletion, HubCacheDeletion)
        self.assertEqual(deletion.revisions, ["abc"])
        execute()
        strategy.execute.assert_called_once()

    def test_cache_delete_no_match(self):
        scan_result = SimpleNamespace(repos=[])
        with patch(
            "avalan.model.hubs.huggingface.scan_cache_dir",
            return_value=scan_result,
        ) as scan_mock:
            result = self.hub.cache_delete("model")
        scan_mock.assert_called_once_with("/cache")
        self.assertEqual(result, (None, None))

    def test_cache_scan(self):
        now = datetime.now().timestamp()
        files = [
            SimpleNamespace(
                file_name="f1",
                file_path="p1",
                size_on_disk=2,
                blob_last_accessed=now,
                blob_last_modified=now,
            ),
            SimpleNamespace(
                file_name="f2",
                file_path="p2",
                size_on_disk=1,
                blob_last_accessed=now,
                blob_last_modified=now,
            ),
        ]
        revision = SimpleNamespace(commit_hash="rev", files=files)
        info = SimpleNamespace(
            repo_id="model",
            repo_path="/cache/model",
            size_on_disk=3,
            revisions=[revision],
            nb_files=2,
        )
        scan_result = SimpleNamespace(repos=[info])
        with patch(
            "avalan.model.hubs.huggingface.scan_cache_dir",
            return_value=scan_result,
        ):
            caches = self.hub.cache_scan()
        self.assertEqual(len(caches), 1)
        cache = caches[0]
        self.assertIsInstance(cache, HubCache)
        self.assertEqual(cache.model_id, "model")
        self.assertEqual(cache.total_revisions, 1)
        self.assertEqual(cache.total_files, 2)
        self.assertEqual([f.name for f in cache.files["rev"]], ["f1", "f2"])

    def test_can_access(self):
        self.assertTrue(self.hub.can_access("model"))
        self.hf_instance.auth_check.assert_called_with("model")
        self.hf_instance.auth_check.side_effect = GatedRepoError("denied")
        self.assertFalse(self.hub.can_access("model"))

    def test_download(self):
        self.hf_instance.snapshot_download.return_value = "/path"
        path = self.hub.download("model")
        self.assertEqual(path, "/path")
        self.hf_instance.snapshot_download.assert_called_with(
            "model",
            cache_dir="/cache",
            tqdm_class=None,
            force_download=False,
            max_workers=8,
            local_dir=None,
            local_dir_use_symlinks=False,
        )
        self.hf_instance.snapshot_download.side_effect = GatedRepoError(
            "denied"
        )
        with self.assertRaises(HubAccessDeniedException):
            self.hub.download("model")

    def test_download_all(self):
        self.hf_instance.list_repo_files.return_value = ["a", "b"]
        files = self.hub.download_all("model")
        self.assertEqual(files, ["a", "b"])
        calls = [
            call(
                "model",
                "a",
                cache_dir="/cache",
                force_download=False,
            ),
            call(
                "model",
                "b",
                cache_dir="/cache",
                force_download=False,
            ),
        ]
        self.hf_instance.hf_hub_download.assert_has_calls(calls)

    def test_model(self):
        info = SimpleNamespace(
            id="model",
            safetensors=SimpleNamespace(total=1, parameters={"w": 0}),
            inference="inf",
            library_name=None,
            card_data={"library_name": "lib", "license": "mit"},
            pipeline_tag="tag",
            tags=["x"],
            config={"architectures": ["arc"], "model_type": "mt"},
            transformers_info={"auto_model": "am", "processor": "proc"},
            gated=False,
            private=True,
            disabled=False,
            downloads=5,
            downloads_all_time=None,
            likes=2,
            trending_score=1,
            author="me",
            created_at=datetime(2020, 1, 1),
            last_modified=datetime(2020, 1, 2),
        )
        self.hf_instance.model_info.return_value = info
        model = self.hub.model("model")
        self.assertIsInstance(model, Model)
        self.assertEqual(model.id, "model")
        self.assertEqual(model.parameters, 1)
        self.assertEqual(model.parameter_types, ["w"])
        self.assertEqual(model.library_name, "lib")
        self.assertEqual(model.license, "mit")

    def test_model_url(self):
        self.assertEqual(self.hub.model_url("m"), "https://huggingface.co/m")

    def test_models(self):
        info = SimpleNamespace(id="m")
        self.hf_instance.list_models.return_value = [info]
        with patch.object(HuggingfaceHub, "_model", return_value="x") as m:
            models = list(
                self.hub.models(
                    filter="f",
                    name=["n1", "n2"],
                    search=["s"],
                    library=["lib"],
                    author="a",
                    gated=True,
                    language=["en"],
                    task=["t"],
                    tags=["tag"],
                    limit=1,
                )
            )
        self.hf_instance.list_models.assert_called_once_with(
            model_name=["n1", "n2"],
            filter="f",
            search=["s"],
            library=["lib"],
            author="a",
            gated=True,
            language=["en"],
            task=["t"],
            tags=["tag"],
            limit=1,
            full=True,
        )
        m.assert_called_once_with(info)
        self.assertEqual(models, ["x"])

    def test_login(self):
        self.hub.login()
        self.mock_login.assert_called_once_with("token")

    def test_user(self):
        self.hf_instance.whoami.return_value = {
            "name": "n",
            "fullname": "full",
            "auth": {"accessToken": {"displayName": "token"}},
        }
        user = self.hub.user()
        self.assertIsInstance(user, User)
        self.assertEqual(user.name, "n")
        self.assertEqual(user.full_name, "full")
        self.assertEqual(user.access_token_name, "token")


if __name__ == "__main__":
    main()
