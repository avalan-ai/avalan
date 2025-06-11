import sys
from types import ModuleType
from tempfile import NamedTemporaryFile
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

# Stub boto3 and botocore if missing
try:
    import boto3  # noqa: F401
    from botocore.exceptions import ClientError  # noqa: F401
except ModuleNotFoundError:
    boto3 = ModuleType("boto3")
    boto3.client = MagicMock()
    boto3_session = ModuleType("boto3.session")

    class DummySession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def client(self, name: str):
            return MagicMock(name=f"{name}_client")

    boto3_session.Session = DummySession
    boto3.session = boto3_session
    botocore_exceptions = ModuleType("botocore.exceptions")

    class ClientError(Exception):
        def __init__(self, response, op):
            super().__init__(response, op)
            self.response = response
            self.operation_name = op

    botocore_exceptions.ClientError = ClientError
    sys.modules.setdefault("boto3", boto3)
    sys.modules.setdefault("boto3.session", boto3_session)
    sys.modules.setdefault("botocore.exceptions", botocore_exceptions)

from avalan.cli.commands import deploy as deploy_cmds


class CliDeployRunTestCase(IsolatedAsyncioTestCase):
    async def test_deploy_run(self):
        config = """
[agents]
publish = "agent.toml"
port = 8000

[aws]
vpc = "vpc"
instance = "t2"
pgsql = "cls"
database = "db"
"""
        with NamedTemporaryFile("w", delete=False) as fh:
            fh.write(config)
            path = fh.name
        args = MagicMock(deployment=path)
        logger = MagicMock()
        aws = MagicMock()
        aws.get_vpc_id = AsyncMock(return_value="v")
        aws.get_security_group = AsyncMock(return_value="sg")
        aws.configure_security_group = AsyncMock()
        aws.create_instance_if_missing = AsyncMock()
        with patch.object(deploy_cmds, "Aws", return_value=aws) as aws_cls:
            await deploy_cmds.deploy_run(args, logger)
        aws_cls.assert_called_once()
        aws.get_vpc_id.assert_awaited_once_with("vpc")
        aws.get_security_group.assert_awaited_once_with("avalan-sg-vpc", "v")
        aws.configure_security_group.assert_awaited_once_with("sg", 8000)
        aws.create_instance_if_missing.assert_awaited_once_with(
            "v",
            "sg",
            "ami-0c02fb55956c7d316",
            "t2",
            "avalan-t2",
            "agent.toml",
            8000,
        )

    async def test_deploy_run_with_persistent_memory(self):
        config = """
[agents]
publish = "agent.toml"
port = 8000
[agents.memory]
permanent = "postgresql://db"

[aws]
vpc = "vpc"
instance = "t2"
pgsql = "cls"
database = "db"
"""
        with NamedTemporaryFile("w", delete=False) as fh:
            fh.write(config)
            path = fh.name
        args = MagicMock(deployment=path)
        logger = MagicMock()
        aws = MagicMock()
        aws.get_vpc_id = AsyncMock(return_value="v")
        aws.get_security_group = AsyncMock(return_value="sg")
        aws.configure_security_group = AsyncMock()
        aws.create_instance_if_missing = AsyncMock()
        aws.create_rds_if_missing = AsyncMock()
        with patch.object(deploy_cmds, "Aws", return_value=aws):
            await deploy_cmds.deploy_run(args, logger)
        aws.create_rds_if_missing.assert_awaited_once_with(
            "db", "cls", "sg", 20
        )
