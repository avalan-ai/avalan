from ...deploy.aws import Aws
from argparse import Namespace
from logging import Logger
from pathlib import Path
from tomllib import load


class DeployError(Exception):
    """Deployment failed."""


async def deploy_run(args: Namespace, logger: Logger) -> None:
    """Deploy agents using a configuration file."""
    path = Path(args.deployment)
    with open(path, "rb") as file:
        config = load(file)
        agents_cfg = config.get("agents", {})
        aws_cfg = config.get("aws", {})
        agent_path = agents_cfg.get("publish", None)
        port = agents_cfg.get("port", 9001)
        memory = agents_cfg.get("memory", {})
        dsn = memory.get("permanent")
        db_storage = 20
        has_persistent_memory = isinstance(dsn, str) and dsn.startswith(
            "postgresql"
        )
        namespaces = {k: v for k, v in agents_cfg.items() if k != "port"}

        assert namespaces, "No agents defined"
        assert aws_cfg and "vpc" in aws_cfg and port and agent_path

        vpc_name = aws_cfg["vpc"]
        sg_name = f"avalan-sg-{vpc_name}"
        instance_name = f"avalan-{aws_cfg['instance']}"
        ami_id = "ami-0c02fb55956c7d316"

        logger.info("Preparing AWS deployment")
        aws = Aws(aws_cfg)

        cidr = aws_cfg.get("cidr", "10.0.0.0/16")
        logger.info('Fetching VPC "%s"', vpc_name)
        try:
            vpc_id = await aws.get_vpc_id(aws_cfg["vpc"])
        except DeployError:
            logger.info('Creating VPC "%s" with CIDR %s', vpc_name, cidr)
            vpc_id = await aws.create_vpc_if_missing(vpc_name, cidr)

        logger.info(
            'Getting or creating security group "%s" on VPC "%s"',
            sg_name,
            vpc_id,
        )
        sg_id = await aws.get_security_group(sg_name, vpc_id)

        logger.info(
            'Configuring access policies for security group "%s" on VPC "%s"',
            sg_name,
            vpc_id,
        )
        await aws.configure_security_group(sg_id, port)

        if has_persistent_memory:
            logger.info('Creating RDS on VPC "%s"', vpc_id)
            await aws.create_rds_if_missing(
                aws_cfg["database"], aws_cfg["pgsql"], sg_id, db_storage
            )

        for namespace, agent_path in namespaces.items():
            logger.info(
                'Creating EC2 instance "%s" on VPC "%s"',
                instance_name,
                vpc_id,
            )
            await aws.create_instance_if_missing(
                vpc_id,
                sg_id,
                ami_id,
                aws_cfg["instance"],
                instance_name,
                agent_path,
                port,
            )
            logger.info(
                "Deployed %s as %s on port %s", agent_path, namespace, port
            )
