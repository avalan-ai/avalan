from logging import getLogger, Logger


def _lf(items: list) -> list:
    return list(filter(None, items))


def _j(sep: str, items: list[str], *args, empty: str = "") -> str:
    real_items = _lf(items)
    return sep.join(real_items) if real_items else empty


def logger_replace(logger: Logger, logger_names: list[str]) -> None:
    assert logger and logger_names
    for logger_name in logger_names:
        updated_logger = getLogger(logger_name)
        assert updated_logger
        updated_logger.handlers = []
        for handler in logger.handlers:
            updated_logger.addHandler(handler)
        updated_logger.setLevel(logger.level)
        updated_logger.propagate = logger.propagate
