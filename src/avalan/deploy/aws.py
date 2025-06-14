from asyncio import AbstractEventLoop, get_running_loop
from botocore.exceptions import ClientError
from boto3 import client
from boto3.session import Session
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


class DeployError(Exception):
    """Deployment failed."""


class AsyncClient:
    def __init__(
        self,
        client: client,
        loop: AbstractEventLoop | None = None,
        executor: ThreadPoolExecutor | None = None,
    ):
        self._client = client
        self._loop = loop or get_running_loop()
        self._executor = executor or ThreadPoolExecutor()

    def __getattr__(self, name: str) -> Callable[..., any]:
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        async def fn(*args, **kwargs):
            return await self._loop.run_in_executor(
                self._executor, lambda: attr(*args, **kwargs)
            )

        return fn


class Aws:
    _ec2: AsyncClient
    _rds: AsyncClient
    _session: Session

    def __init__(
        self, settings: dict | None = None, token_pair: str | None = None
    ):
        if settings and "token_pair" in settings and not token_pair:
            token_pair = settings.pop("token_pair")

        aws_settings = {}

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
        response = await self._ec2.describe_vpcs(
            Filters=[{"Name": "tag:Name", "Values": [name]}]
        )
        vpcs = response.get("Vpcs", [])
        if not vpcs:
            raise DeployError(f"VPC {name!r} not found")
        return vpcs[0]["VpcId"]

    async def create_vpc_if_missing(self, name: str, cidr: str) -> str:
        """Return an existing VPC id or create a new VPC."""
        try:
            return await self.get_vpc_id(name)
        except DeployError:
            response = await self._ec2.create_vpc(CidrBlock=cidr)
            vpc_id = response["Vpc"]["VpcId"]
            await self._ec2.create_tags(
                Resources=[vpc_id], Tags=[{"Key": "Name", "Value": name}]
            )
            waiter = await self._ec2.get_waiter("vpc_available")
            waiter.wait(VpcIds=[vpc_id])
            return vpc_id

    async def get_security_group(self, name: str, vpc_id: str) -> str:
        response = await self._ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [name]}]
        )
        groups = response.get("SecurityGroups", [])
        if groups:
            return groups[0]["GroupId"]

        response = await self._ec2.create_security_group(
            GroupName=name,
            Description="avalan deployment",
            VpcId=vpc_id,
        )
        return response["GroupId"]

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
        try:
            await self._rds.describe_db_instances(DBInstanceIdentifier=db_id)
        except self._rds.exceptions.DBInstanceNotFoundFault:
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
            waiter = await self._rds.get_waiter("db_instance_available")
            waiter.wait(DBInstanceIdentifier=db_id)
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
        response = await self._ec2.describe_instances(
            Filters=[{"Name": "tag:Name", "Values": [instance_name]}]
        )
        reservations = response.get("Reservations", [])
        if reservations:
            return reservations[0]["Instances"][0]["InstanceId"]

        response = await self._ec2.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )

        subnet = response["Subnets"][0]["SubnetId"]

        response = await self._ec2.run_instances(
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
        )
        return response["Instances"][0]["InstanceId"]

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
