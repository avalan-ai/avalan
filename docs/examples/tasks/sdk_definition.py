from avalan.task import (
    IdempotencyMode,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskOutputContract,
    TaskRunPolicy,
)


def build_definition() -> TaskDefinition:
    """Return the structured JSON task definition."""
    return TaskDefinition(
        task=TaskMetadata(
            name="structured_json",
            version="1",
            description="Answer a structured request with JSON.",
            labels=("example", "json"),
        ),
        input=TaskInputContract.object(
            schema={
                "additionalProperties": False,
                "properties": {
                    "priority": {
                        "maximum": 5,
                        "minimum": 1,
                        "type": "integer",
                    },
                    "question": {
                        "minLength": 1,
                        "type": "string",
                    },
                },
                "required": ["question"],
                "type": "object",
            },
            description="Question object.",
        ),
        output=TaskOutputContract.json(
            schema={
                "additionalProperties": False,
                "properties": {
                    "answer": {
                        "minLength": 1,
                        "type": "string",
                    },
                    "confidence": {
                        "maximum": 1,
                        "minimum": 0,
                        "type": "number",
                    },
                },
                "required": ["answer"],
                "type": "object",
            },
            description="Structured answer.",
        ),
        execution=TaskExecutionTarget.agent(
            "agents/basic_answer.toml",
            variables={"style": "concise"},
        ),
        run=TaskRunPolicy.direct(
            timeout_seconds=120,
            idempotency=IdempotencyMode.INPUT_HASH,
        ),
        limits=TaskLimitsPolicy(
            input_bytes=8192,
            output_bytes=8192,
            total_tokens=1500,
        ),
    )


if __name__ == "__main__":
    print(build_definition())
