import sys
from types import ModuleType
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

# Provide boto3/botocore stubs when missing
try:
    import boto3  # noqa: F401
    from botocore.exceptions import (
        ClientError as DummyClientError,
    )  # noqa: F401
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

    class DummyClientError(Exception):
        def __init__(self, error_response, operation_name):
            super().__init__(error_response, operation_name)
            self.response = error_response
            self.operation_name = operation_name

    botocore_exceptions.ClientError = DummyClientError
    sys.modules.setdefault("boto3", boto3)
    sys.modules.setdefault("boto3.session", boto3_session)
    sys.modules.setdefault("botocore.exceptions", botocore_exceptions)

from avalan.deploy.aws import AsyncClient, Aws, DeployError


class AsyncClientTestCase(IsolatedAsyncioTestCase):
    async def test_callable_attribute_runs_in_executor(self):
        cli = MagicMock()
        cli.call = MagicMock(return_value="ok")
        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=lambda exec, func: func())
        ac = AsyncClient(cli, loop=loop, executor="E")

        result = await ac.call(1, a=2)

        self.assertEqual(result, "ok")
        loop.run_in_executor.assert_awaited_once()
        cli.call.assert_called_once_with(1, a=2)

    async def test_non_callable_attribute_passthrough(self):
        cli = MagicMock()
        cli.value = 5
        loop = MagicMock()
        ac = AsyncClient(cli, loop=loop)
        self.assertEqual(ac.value, 5)


class AwsTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        session = MagicMock()
        session.client.side_effect = lambda name: MagicMock(
            name=f"{name}_client"
        )
        self.session_patch = patch(
            "avalan.deploy.aws.Session", return_value=session
        )
        self.ac_patch = patch(
            "avalan.deploy.aws.AsyncClient", side_effect=lambda c: MagicMock()
        )
        self.session = session
        self.session_patch.start()
        self.async_client_class = self.ac_patch.start()
        self.addCleanup(self.session_patch.stop)
        self.addCleanup(self.ac_patch.stop)
        self.aws = Aws({"zone": "z"}, token_pair="A:B")
        self.aws._ec2 = MagicMock()
        self.aws._rds = MagicMock()
        exc = type("NotFound", (Exception,), {})
        self.aws._rds.exceptions = MagicMock(DBInstanceNotFoundFault=exc)

    async def test_init_creates_clients(self):
        self.session_patch.stop()
        self.ac_patch.stop()
        session = MagicMock()
        session.client.side_effect = lambda name: f"{name}-client"
        with (
            patch("avalan.deploy.aws.Session", return_value=session) as ses,
            patch(
                "avalan.deploy.aws.AsyncClient",
                side_effect=lambda c: f"async-{c}",
            ) as ac,
        ):
            aws = Aws({"zone": "r"}, token_pair="X:Y")
        ses.assert_called_once_with(
            aws_access_key_id="X", aws_secret_access_key="Y", region_name="r"
        )
        self.assertEqual(aws._ec2, "async-ec2-client")
        self.assertEqual(aws._rds, "async-rds-client")
        ac.assert_any_call("ec2-client")
        ac.assert_any_call("rds-client")

    async def test_get_vpc_id_found_and_missing(self):
        self.aws._ec2.describe_vpcs = AsyncMock(
            return_value={"Vpcs": [{"VpcId": "v"}]}
        )
        vpc = await self.aws.get_vpc_id("name")
        self.assertEqual(vpc, "v")
        self.aws._ec2.describe_vpcs.assert_awaited_once()

        self.aws._ec2.describe_vpcs = AsyncMock(return_value={"Vpcs": []})
        with self.assertRaises(DeployError):
            await self.aws.get_vpc_id("name")

    async def test_get_security_group(self):
        self.aws._ec2.describe_security_groups = AsyncMock(
            return_value={"SecurityGroups": [{"GroupId": "g"}]}
        )
        gid = await self.aws.get_security_group("n", "v")
        self.assertEqual(gid, "g")

        self.aws._ec2.describe_security_groups = AsyncMock(
            return_value={"SecurityGroups": []}
        )
        self.aws._ec2.create_security_group = AsyncMock(
            return_value={"GroupId": "n"}
        )
        gid = await self.aws.get_security_group("n", "v")
        self.assertEqual(gid, "n")
        self.aws._ec2.create_security_group.assert_awaited_once()

    async def test_configure_security_group(self):
        self.aws._ec2.authorize_security_group_ingress = AsyncMock()
        await self.aws.configure_security_group("g", 1)
        self.aws._ec2.authorize_security_group_ingress.assert_awaited_once_with(
            GroupId="g",
            IpProtocol="tcp",
            FromPort=1,
            ToPort=1,
            CidrIp="0.0.0.0/0",
        )

        self.aws._ec2.authorize_security_group_ingress.reset_mock()
        err = DummyClientError(
            {"Error": {"Code": "InvalidPermission.Duplicate"}}, "op"
        )
        self.aws._ec2.authorize_security_group_ingress.side_effect = err
        await self.aws.configure_security_group("g", 1)

    async def test_create_rds_if_missing(self):
        self.aws._rds.describe_db_instances = AsyncMock(
            side_effect=self.aws._rds.exceptions.DBInstanceNotFoundFault
        )
        self.aws._rds.create_db_instance = AsyncMock()
        waiter = MagicMock()
        waiter.wait = MagicMock()
        self.aws._rds.get_waiter = AsyncMock(return_value=waiter)
        db = await self.aws.create_rds_if_missing("db", "cls", "sg", 10)
        self.assertEqual(db, "db")
        self.aws._rds.create_db_instance.assert_awaited_once()
        waiter.wait.assert_called_once_with(DBInstanceIdentifier="db")

        self.aws._rds.describe_db_instances = AsyncMock(return_value=None)
        self.aws._rds.create_db_instance.reset_mock()
        db = await self.aws.create_rds_if_missing("db", "cls", "sg", 10)
        self.assertEqual(db, "db")
        self.aws._rds.create_db_instance.assert_not_called()

    async def test_create_instance_if_missing(self):
        self.aws._ec2.describe_instances = AsyncMock(
            return_value={
                "Reservations": [{"Instances": [{"InstanceId": "i"}]}]
            }
        )
        iid = await self.aws.create_instance_if_missing(
            "v", "sg", "a", "t", "n", "p", 1
        )
        self.assertEqual(iid, "i")

        self.aws._ec2.describe_instances = AsyncMock(
            return_value={"Reservations": []}
        )
        self.aws._ec2.describe_subnets = AsyncMock(
            return_value={"Subnets": [{"SubnetId": "s"}]}
        )
        self.aws._ec2.run_instances = AsyncMock(
            return_value={"Instances": [{"InstanceId": "n"}]}
        )
        iid = await self.aws.create_instance_if_missing(
            "v", "sg", "a", "t", "n", "p", 1
        )
        self.assertEqual(iid, "n")
        self.aws._ec2.run_instances.assert_awaited_once()

    def test_create_user_data(self):
        data = self.aws._create_user_data("agent.toml", 9)
        self.assertIn("pip3 install avalan", data)
        self.assertIn("--port 9", data)
