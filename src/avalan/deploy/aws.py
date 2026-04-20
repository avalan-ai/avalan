from asyncio import AbstractEventLoop, get_running_loop
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, cast

from boto3.session import Session
from botocore.exceptions import ClientError


class DeployError(Exception):
    """Deployment failed."""


class AsyncWaiter:
    """Run waiter operations without blocking the event loop."""

    def __init__(
        self,
        waiter: Any,
        loop: AbstractEventLoop,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._waiter = waiter
        self._loop = loop
        self._executor = executor

    async def wait(self, *args: Any, **kwargs: Any) -> Any:
        return await self._loop.run_in_executor(
            self._executor, lambda: self._waiter.wait(*args, **kwargs)
        )


class AsyncClient:
    def __init__(
        self,
        client: Any,
        loop: AbstractEventLoop | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._client = client
        self._loop = loop or get_running_loop()
        self._executor = executor or ThreadPoolExecutor()

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]] | Any:
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        async def fn(*args: Any, **kwargs: Any) -> Any:
            result = await self._loop.run_in_executor(
                self._executor, lambda: attr(*args, **kwargs)
            )
            if name == "get_waiter":
                return AsyncWaiter(result, self._loop, self._executor)
            return result

        return fn


class Aws:
    _ec2: AsyncClient
    _rds: AsyncClient
    _session: Session

    def __init__(
        self,
        settings: dict[str, str] | None = None,
        token_pair: str | None = None,
    ) -> None:
        if settings and "token_pair" in settings and not token_pair:
            token_pair = settings.pop("token_pair")

        aws_settings: dict[str, str] = {}

        if token_pair:
            access_key, secret_key = token_pair.split(":", 1)
            aws_settings.update(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )

        if settings and "zone" in settings:
            aws_settings["region_name"] = settings["zone"]

        self._session = Session(**aws_settings)
        self._ec2 = AsyncClient(self._session.client("ec2"))
        self._rds = AsyncClient(self._session.client("rds"))

    async def get_vpc_id(self, name: str) -> str:
        response = cast(
            dict[str, Any],
            await self._ec2.describe_vpcs(
                Filters=[{"Name": "tag:Name", "Values": [name]}]
            ),
        )
        vpcs = cast(list[dict[str, Any]], response.get("Vpcs", []))
        if not vpcs:
            raise DeployError(f"VPC {name!r} not found")
        return str(vpcs[0]["VpcId"])

    async def create_vpc_if_missing(self, name: str, cidr: str) -> str:
        """Return an existing VPC id or create a new VPC."""
        try:
            return await self.get_vpc_id(name)
        except DeployError:
            response = cast(
                dict[str, Any], await self._ec2.create_vpc(CidrBlock=cidr)
            )
            vpc = cast(dict[str, Any], response["Vpc"])
            vpc_id = str(vpc["VpcId"])
            await self._ec2.create_tags(
                Resources=[vpc_id], Tags=[{"Key": "Name", "Value": name}]
            )
            waiter = cast(
                AsyncWaiter, await self._ec2.get_waiter("vpc_available")
            )
            await waiter.wait(VpcIds=[vpc_id])
            return vpc_id

    async def get_security_group(self, name: str, vpc_id: str) -> str:
        response = cast(
            dict[str, Any],
            await self._ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [name]}]
            ),
        )
        groups = cast(list[dict[str, Any]], response.get("SecurityGroups", []))
        if groups:
            return str(groups[0]["GroupId"])

        response = cast(
            dict[str, Any],
            await self._ec2.create_security_group(
                GroupName=name,
                Description="avalan deployment",
                VpcId=vpc_id,
            ),
        )
        return str(response["GroupId"])

    async def configure_security_group(self, group_id: str, port: int) -> None:
        try:
            await self._ec2.authorize_security_group_ingress(
                GroupId=group_id,
                IpProtocol="tcp",
                FromPort=port,
                ToPort=port,
                CidrIp="0.0.0.0/0",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                raise

    async def create_rds_if_missing(
        self, db_id: str, instance_class: str, sg_id: str, storage: int
    ) -> str:
        not_found_error = getattr(
            getattr(self._rds, "exceptions", object()),
            "DBInstanceNotFoundFault",
            None,
        )

        try:
            await self._rds.describe_db_instances(DBInstanceIdentifier=db_id)
        except Exception as exc:
            is_not_found = bool(
                not_found_error
                and isinstance(exc, cast(type[BaseException], not_found_error))
            )
            if isinstance(exc, ClientError):
                is_not_found = (
                    is_not_found
                    or cast(str, exc.response["Error"]["Code"])
                    == "DBInstanceNotFound"
                )
            if not is_not_found:
                raise

            await self._rds.create_db_instance(
                DBInstanceIdentifier=db_id,
                DBInstanceClass=instance_class,
                Engine="postgres",
                MasterUsername="postgres",
                MasterUserPassword="postgres",
                AllocatedStorage=storage,
                VpcSecurityGroupIds=[sg_id],
                Tags=[{"Key": "Name", "Value": db_id}],
            )
            waiter = cast(
                AsyncWaiter,
                await self._rds.get_waiter("db_instance_available"),
            )
            await waiter.wait(DBInstanceIdentifier=db_id)
        return db_id

    async def create_instance_if_missing(
        self,
        vpc_id: str,
        sg_id: str,
        ami_id: str,
        instance_type: str,
        instance_name: str,
        agent_path: str,
        port: int,
    ) -> str:
        user_data = self._create_user_data(agent_path, port)
        response = cast(
            dict[str, Any],
            await self._ec2.describe_instances(
                Filters=[{"Name": "tag:Name", "Values": [instance_name]}]
            ),
        )
        reservations = cast(
            list[dict[str, Any]], response.get("Reservations", [])
        )
        if reservations:
            instances = cast(
                list[dict[str, Any]], reservations[0].get("Instances", [])
            )
            if instances:
                return str(instances[0]["InstanceId"])

        response = cast(
            dict[str, Any],
            await self._ec2.describe_subnets(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            ),
        )
        subnets = cast(list[dict[str, Any]], response.get("Subnets", []))
        assert subnets, "No subnet found for VPC"
        subnet = str(subnets[0]["SubnetId"])

        response = cast(
            dict[str, Any],
            await self._ec2.run_instances(
                ImageId=ami_id,
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                SecurityGroupIds=[sg_id],
                SubnetId=subnet,
                UserData=user_data,
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [{"Key": "Name", "Value": instance_name}],
                    }
                ],
            ),
        )
        instances = cast(list[dict[str, Any]], response.get("Instances", []))
        assert instances, "No instance returned by EC2"
        return str(instances[0]["InstanceId"])

    def _create_user_data(self, agent_path: str, port: int) -> str:
        cmd = f"avalan agent serve {agent_path} --host 0.0.0.0 --port {port}\n"
        service = f"""
    [Unit]
    Description=Avalan Agent
    After=network.target

    [Service]
    Type=simple
    ExecStart=/usr/local/bin/{cmd}
    Restart=always

    [Install]
    WantedBy=multi-user.target
    """
        script = (
            "#!/bin/bash\n"
            "apt-get update -y\n"
            "apt-get install -y python3-pip\n"
            "pip3 install avalan\n"
            f"echo '{service}' > /etc/systemd/system/avalan.service\n"
            "systemctl daemon-reload\n"
            "systemctl enable avalan.service\n"
            "systemctl start avalan.service\n"
        )
        return script
