from ..entities import ImageEntity
from .call import ModelCall as ModelCall
from .call import ModelCallContext as ModelCallContext
from .capability import (
    CapabilityBatchAccepted as CapabilityBatchAccepted,
)
from .capability import (
    CapabilityBatchClassification as CapabilityBatchClassification,
)
from .capability import (
    CapabilityBatchRejected as CapabilityBatchRejected,
)
from .capability import (
    CapabilityBatchRejectionCode as CapabilityBatchRejectionCode,
)
from .capability import (
    ContinuationSnapshotCodecRegistry as ContinuationSnapshotCodecRegistry,
)
from .capability import (
    CorrelatedCapabilityResult as CorrelatedCapabilityResult,
)
from .capability import DomainCapabilityCall as DomainCapabilityCall
from .capability import DomainCapabilitySeed as DomainCapabilitySeed
from .capability import ModelCapabilityCatalog as ModelCapabilityCatalog
from .capability import (
    ModelCapabilityDescriptor as ModelCapabilityDescriptor,
)
from .capability import ModelCapabilityError as ModelCapabilityError
from .capability import ModelCapabilityKind as ModelCapabilityKind
from .capability import (
    ModelCapabilityValidationError as ModelCapabilityValidationError,
)
from .capability import (
    ProviderCapabilityCall as ProviderCapabilityCall,
)
from .capability import (
    ProviderCapabilityProjection as ProviderCapabilityProjection,
)
from .capability import (
    ProviderCapabilitySupport as ProviderCapabilitySupport,
)
from .capability import (
    RegisteredContinuationSnapshotCodec as RegisteredContinuationSnapshotCodec,
)
from .capability import (
    TaskInputCapabilityAdvertisement as TaskInputCapabilityAdvertisement,
)
from .capability import (
    TaskInputCapabilityCall as TaskInputCapabilityCall,
)
from .file_delivery import FileDeliveryDecision as FileDeliveryDecision
from .file_delivery import FileDeliveryDiagnostic as FileDeliveryDiagnostic
from .file_delivery import FileDeliveryLimit as FileDeliveryLimit
from .file_delivery import FileDeliveryMode as FileDeliveryMode
from .file_delivery import FileDeliveryProfile as FileDeliveryProfile
from .file_delivery import FileDeliveryRequest as FileDeliveryRequest
from .file_delivery import (
    LocalFileDeliveryProfile as LocalFileDeliveryProfile,
)
from .file_delivery import plan_file_delivery as plan_file_delivery
from .file_delivery import (
    resolve_file_delivery_profile as resolve_file_delivery_profile,
)
from .input import input_files as input_files
from .reasoning import (
    ReasoningSummaryCapabilityError as ReasoningSummaryCapabilityError,
)
from .reasoning import (
    ReasoningSummaryRequestCapability as ReasoningSummaryRequestCapability,
)
from .reasoning import (
    validate_reasoning_summary_request as validate_reasoning_summary_request,
)
from .response.text import TextGenerationResponse
from .stream import CanonicalStreamItem
from .vendor import TextGenerationVendorStream

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    TypeAlias,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:

    class NDArray:  # noqa: D101
        def __class_getitem__(cls, _: Any) -> Any:
            return Any


OutputGenerator = AsyncGenerator[CanonicalStreamItem, None]
OutputFunction = Callable[..., object]

EngineResponse: TypeAlias = (
    TextGenerationResponse
    | TextGenerationVendorStream
    | ImageEntity
    | list[ImageEntity]
    | list[str]
    | dict[str, str]
    | NDArray[Any]
    | str
)


class ModelAlreadyLoadedException(Exception):
    pass


class TokenizerAlreadyLoadedException(Exception):
    pass


class TokenizerNotSupportedException(Exception):
    pass
