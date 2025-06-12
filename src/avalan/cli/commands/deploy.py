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

        logger.info(f'Fetching VPC "{vpc_name}"')
        vpc_id = await aws.get_vpc_id(aws_cfg["vpc"])

        logger.info(
            f'Getting or creating security group "{sg_name}" on VPC "{vpc_id}"'
        )
        sg_id = await aws.get_security_group(sg_name, vpc_id)

        logger.info(
            f"Configuring access policies for security group"
            f' "{sg_name}" on VPC "{vpc_id}"'
        )
        await aws.configure_security_group(sg_id, port)

        if has_persistent_memory:
            logger.info(f'Creating RDS on VPC "{vpc_id}"')
            await aws.create_rds_if_missing(
                aws_cfg["database"], aws_cfg["pgsql"], sg_id, db_storage
            )

        for namespace, agent_path in namespaces.items():
            logger.info(
                f'Creating EC2 instance "{instance_name}" on VPC "{vpc_id}"'
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
            logger.info(f"Deployed {agent_path} as {namespace} on port {port}")
