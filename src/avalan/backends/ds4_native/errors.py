class Ds4Error(RuntimeError):
    """Base error for Avalan's native DS4 backend."""


class Ds4ApiVersionError(Ds4Error):
    """Raise when the DS4 binding does not match Avalan's expected API."""


class Ds4BackendUnavailable(Ds4Error):
    """Raise when the requested DS4 native backend cannot be used."""


class Ds4LoadError(Ds4Error):
    """Raise when a DS4 engine cannot load a model."""


class Ds4InvalidModel(Ds4Error):
    """Raise when a DS4 model path or file is invalid."""


class Ds4ContextError(Ds4Error):
    """Raise when a DS4 context size or state is invalid."""


class Ds4GenerationError(Ds4Error):
    """Raise when DS4 generation fails."""


class Ds4Cancelled(Ds4Error):
    """Raise when DS4 generation is cancelled."""
