"""Provide inert fake data and canonical deterministic test effects."""

from ..types import JsonValue
from .binding import (
    CapabilityEvidence,
    CapabilityEvidenceState,
    ConversationCapability,
    ConversationCapabilityProfile,
    ProviderLaneBinding,
)
from .contract import (
    AuthorityScope,
    ConversationModelCallId,
    UpstreamResponseId,
)
from .errors import (
    ConversationConflictError,
    ConversationValidationError,
)
from .items import (
    PROVIDER_ITEM_NORMALIZATION_VERSION,
    ProviderItem,
    ProviderItemCaller,
    ProviderItemKind,
    ProviderItemPhase,
)
from .observability import ConversationObservation
from .protocols import (
    CoordinatorBoundaryHook,
    FirstStoredProviderPlan,
    ProviderPlan,
    ProviderResult,
    StandaloneCompactProviderPlan,
    StatelessProviderPlan,
    StoredProviderPlan,
)
from .runtime import CoordinatorAwaitBoundary, PublicationIntent
from .settings import (
    EffectiveReasoningContext,
    EffectiveReasoningMetadata,
    ProviderUsage,
    ReasoningContext,
)
from .store import StoreAwaitBoundary, StoreBoundaryHook
from .value import (
    CapabilityProfileId,
    OpaqueProviderState,
    ProviderItemId,
    ProviderItemIndex,
    ProviderItemOrder,
)

from asyncio import sleep as _asyncio_sleep
from collections.abc import Awaitable, Callable, ItemsView, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import Protocol, TypeAlias, TypeVar, cast, final

_ASYNCIO_SLEEP = _asyncio_sleep
_ATTRIBUTE_ERROR_TYPE = AttributeError
_BASE_EXCEPTION_TYPE = BaseException
_BUILTIN_BOOL_TYPE = bool
_BUILTIN_DICT_TYPE = dict
_BUILTIN_FROZENSET_TYPE = frozenset
_BUILTIN_LIST_TYPE = list
_BUILTIN_SET_TYPE = set
_BUILTIN_STR_TYPE = str
_BUILTIN_TUPLE_EXACT_TYPE: object = tuple
_BUILTIN_TUPLE_TYPE: Callable[..., tuple[str, ...]] = tuple
_DICT_CONTAINS = dict.__contains__
_DICT_ITEMS = dict.items
_DICT_POP = dict.pop
_DICT_SETITEM = dict.__setitem__
_EXACT_TYPE = type
_FROZENSET_CONTAINS = frozenset.__contains__
_FROZENSET_ITER = frozenset.__iter__
_INSTANCE_CHECK = isinstance
_LENGTH = len
_LIST_APPEND = list.append
_LIST_CONTAINS = list.__contains__
_LIST_ITER = list.__iter__
_OBJECT_GETATTRIBUTE = object.__getattribute__
_OBJECT_SETATTR = object.__setattr__
_SET_ADD = set.add
_SET_CONTAINS = set.__contains__
_SET_ITER = set.__iter__
_STR_STRIP = str.strip
_VALIDATION_ERROR_TYPE = ConversationValidationError
_FAULT_RENDEZVOUS_MAX_YIELDS = 65_536

_CastTarget = TypeVar("_CastTarget")
_PostInitOwner = TypeVar("_PostInitOwner")


class _ClosedCast(Protocol):
    def __call__(
        self,
        target: type[_CastTarget],
        value: object,
        /,
    ) -> _CastTarget: ...


def _build_fault_action_post_init(
    *,
    attribute_error_type: type[AttributeError] = _ATTRIBUTE_ERROR_TYPE,
    base_exception_type: type[BaseException] = _BASE_EXCEPTION_TYPE,
    bool_type: type[bool] = _BUILTIN_BOOL_TYPE,
    exact_type: Callable[[object], type[object]] = _EXACT_TYPE,
    instance_check: Callable[..., bool] = _INSTANCE_CHECK,
    cast_operation: _ClosedCast = cast,
    object_getattribute: Callable[..., object] = _OBJECT_GETATTRIBUTE,
    str_strip: Callable[[str], str] = _STR_STRIP,
    str_type: type[str] = _BUILTIN_STR_TYPE,
    validation_error_type: type[ConversationValidationError] = (
        _VALIDATION_ERROR_TYPE
    ),
) -> Callable[[object], None]:
    """Build closed validation for one immutable fault action."""

    def fault_action_post_init(action: object) -> None:
        try:
            label_value = object_getattribute(action, "label")
            pause = object_getattribute(action, "pause")
            exception = object_getattribute(action, "exception")
        except attribute_error_type as exc:
            raise validation_error_type() from exc
        if exact_type(label_value) is not str_type:
            raise validation_error_type()
        label = cast_operation(str_type, label_value)
        if (
            not label
            or label != str_strip(label)
            or exact_type(pause) is not bool_type
            or (
                exception is not None
                and not instance_check(exception, base_exception_type)
            )
        ):
            raise validation_error_type()

    return fault_action_post_init


_FAULT_ACTION_POST_INIT = _build_fault_action_post_init()


def _bind_fault_action_post_init(
    _method: Callable[[_PostInitOwner], None],
) -> Callable[[_PostInitOwner], None]:
    """Bind the closed post-init validator during class construction."""
    _method(cast(_PostInitOwner, None))
    return cast(Callable[[_PostInitOwner], None], _FAULT_ACTION_POST_INIT)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FaultAction:
    """Schedule one repository-owned pause or fault at an await label."""

    label: str
    pause: bool = False
    exception: BaseException | None = None

    @_bind_fault_action_post_init
    def __post_init__(self) -> None:
        pass


def _build_fault_action_validator(
    action_type: type[FaultAction],
    *,
    attribute_error_type: type[AttributeError] = _ATTRIBUTE_ERROR_TYPE,
    base_exception_type: type[BaseException] = _BASE_EXCEPTION_TYPE,
    bool_type: type[bool] = _BUILTIN_BOOL_TYPE,
    exact_type: Callable[[object], type[object]] = _EXACT_TYPE,
    instance_check: Callable[..., bool] = _INSTANCE_CHECK,
    cast_operation: _ClosedCast = cast,
    object_getattribute: Callable[..., object] = _OBJECT_GETATTRIBUTE,
    str_strip: Callable[[str], str] = _STR_STRIP,
    str_type: type[str] = _BUILTIN_STR_TYPE,
    validation_error_type: type[ConversationValidationError] = (
        _VALIDATION_ERROR_TYPE
    ),
) -> Callable[[object], None]:
    """Build exact closed validation for fault actions."""

    def validate_fault_action(action: object) -> None:
        if exact_type(action) is not action_type:
            raise validation_error_type()
        try:
            label_value = object_getattribute(action, "label")
            pause = object_getattribute(action, "pause")
            exception = object_getattribute(action, "exception")
        except attribute_error_type as exc:
            raise validation_error_type() from exc
        if exact_type(label_value) is not str_type:
            raise validation_error_type()
        label = cast_operation(str_type, label_value)
        if (
            not label
            or label != str_strip(label)
            or exact_type(pause) is not bool_type
            or (
                exception is not None
                and not instance_check(exception, base_exception_type)
            )
        ):
            raise validation_error_type()

    return validate_fault_action


_validate_fault_action = _build_fault_action_validator(FaultAction)


_FaultControllerCollections: TypeAlias = tuple[
    dict[str, FaultAction],
    list[str],
    frozenset[str],
    frozenset[str],
    set[str],
    set[str],
    set[str],
]


class _FaultControllerStorage(Protocol):
    _actions: dict[str, FaultAction]
    _closed: bool
    _collections: _FaultControllerCollections
    _completed_labels: set[str]
    _entered_labels: set[str]
    _paused_labels: frozenset[str]
    _released_labels: set[str]
    _scheduled_labels: frozenset[str]
    _visited: list[str]


_FaultControllerStateValidator: TypeAlias = Callable[[object], None]
_FaultControllerInitializer: TypeAlias = Callable[
    [_FaultControllerStorage, tuple[FaultAction, ...]], None
]
_FaultControllerCollectionsReader: TypeAlias = Callable[
    [_FaultControllerStorage], _FaultControllerCollections
]
_FaultControllerCollectionsValidator: TypeAlias = Callable[
    [_FaultControllerStorage, _FaultControllerCollections], None
]
_FaultControllerVisitedReader: TypeAlias = Callable[
    [_FaultControllerStorage], tuple[str, ...]
]
_FaultControllerAsyncLabelOperation: TypeAlias = Callable[
    [_FaultControllerStorage, str], Awaitable[None]
]
_FaultControllerLabelOperation: TypeAlias = Callable[
    [_FaultControllerStorage, str], None
]
_FaultControllerCloseOperation: TypeAlias = Callable[
    [_FaultControllerStorage], None
]


def _build_fault_controller_operations(
    action_type: type[FaultAction],
    validate_action: Callable[[object], None],
    *,
    action_map_cast_type: type[dict[str, FaultAction]] = dict,
    attribute_error_type: type[AttributeError] = _ATTRIBUTE_ERROR_TYPE,
    async_sleep: Callable[[float], Awaitable[None]] = _ASYNCIO_SLEEP,
    base_exception_type: type[BaseException] = _BASE_EXCEPTION_TYPE,
    bool_type: type[bool] = _BUILTIN_BOOL_TYPE,
    cast_operation: _ClosedCast = cast,
    collections_cast_type: type[_FaultControllerCollections] = tuple,
    dict_contains: Callable[..., bool] = _DICT_CONTAINS,
    dict_items: Callable[..., ItemsView[str, FaultAction]] = _DICT_ITEMS,
    dict_pop: Callable[..., FaultAction | None] = _DICT_POP,
    dict_setitem: Callable[..., None] = _DICT_SETITEM,
    dict_type: Callable[..., dict[str, FaultAction]] = _BUILTIN_DICT_TYPE,
    exact_type: Callable[[object], type[object]] = _EXACT_TYPE,
    frozenset_contains: Callable[..., bool] = _FROZENSET_CONTAINS,
    frozenset_cast_type: type[frozenset[str]] = frozenset,
    frozenset_iter: Callable[..., Iterator[str]] = _FROZENSET_ITER,
    frozenset_type: Callable[..., frozenset[str]] = _BUILTIN_FROZENSET_TYPE,
    length: Callable[..., int] = _LENGTH,
    list_append: Callable[..., None] = _LIST_APPEND,
    list_cast_type: type[list[str]] = list,
    list_contains: Callable[..., bool] = _LIST_CONTAINS,
    list_iter: Callable[..., Iterator[str]] = _LIST_ITER,
    list_type: Callable[..., list[str]] = _BUILTIN_LIST_TYPE,
    max_yields: int = _FAULT_RENDEZVOUS_MAX_YIELDS,
    object_getattribute: Callable[..., object] = _OBJECT_GETATTRIBUTE,
    object_setattr: Callable[..., None] = _OBJECT_SETATTR,
    set_add: Callable[..., None] = _SET_ADD,
    set_cast_type: type[set[str]] = set,
    set_contains: Callable[..., bool] = _SET_CONTAINS,
    set_iter: Callable[..., Iterator[str]] = _SET_ITER,
    set_type: Callable[..., set[str]] = _BUILTIN_SET_TYPE,
    str_strip: Callable[[str], str] = _STR_STRIP,
    str_type: type[str] = _BUILTIN_STR_TYPE,
    tuple_exact_type: object = _BUILTIN_TUPLE_EXACT_TYPE,
    tuple_type: Callable[..., tuple[str, ...]] = _BUILTIN_TUPLE_TYPE,
    validation_error_type: type[ConversationValidationError] = (
        _VALIDATION_ERROR_TYPE
    ),
) -> tuple[
    _FaultControllerStateValidator,
    _FaultControllerInitializer,
    _FaultControllerCollectionsReader,
    _FaultControllerCollectionsValidator,
    _FaultControllerVisitedReader,
    _FaultControllerAsyncLabelOperation,
    _FaultControllerAsyncLabelOperation,
    _FaultControllerLabelOperation,
    _FaultControllerCloseOperation,
]:
    """Build the closed deterministic pause operation graph."""

    def validate_fault_controller_state(controller: object) -> None:
        try:
            actions = object_getattribute(controller, "_actions")
            visited = object_getattribute(controller, "_visited")
            scheduled_labels = object_getattribute(
                controller, "_scheduled_labels"
            )
            paused_labels = object_getattribute(controller, "_paused_labels")
            entered_labels = object_getattribute(controller, "_entered_labels")
            released_labels = object_getattribute(
                controller, "_released_labels"
            )
            completed_labels = object_getattribute(
                controller, "_completed_labels"
            )
            collections = object_getattribute(controller, "_collections")
            closed = object_getattribute(controller, "_closed")
        except attribute_error_type as exc:
            raise validation_error_type() from exc
        if (
            exact_type(actions) is not dict_type
            or exact_type(visited) is not list_type
            or exact_type(scheduled_labels) is not frozenset_type
            or exact_type(paused_labels) is not frozenset_type
            or exact_type(entered_labels) is not set_type
            or exact_type(released_labels) is not set_type
            or exact_type(completed_labels) is not set_type
            or exact_type(collections) is not tuple_exact_type
            or length(collections) != 7
            or exact_type(closed) is not bool_type
        ):
            raise validation_error_type()
        canonical_collections = cast_operation(
            collections_cast_type, collections
        )
        if (
            canonical_collections[0] is not actions
            or canonical_collections[1] is not visited
            or canonical_collections[2] is not scheduled_labels
            or canonical_collections[3] is not paused_labels
            or canonical_collections[4] is not entered_labels
            or canonical_collections[5] is not released_labels
            or canonical_collections[6] is not completed_labels
        ):
            raise validation_error_type()
        scheduled = cast_operation(frozenset_cast_type, scheduled_labels)
        paused = cast_operation(frozenset_cast_type, paused_labels)
        entered = cast_operation(set_cast_type, entered_labels)
        released = cast_operation(set_cast_type, released_labels)
        completed = cast_operation(set_cast_type, completed_labels)
        action_map = cast_operation(action_map_cast_type, actions)
        visits = cast_operation(list_cast_type, visited)
        for label in frozenset_iter(scheduled):
            if exact_type(label) is not str_type:
                raise validation_error_type()
        for label in frozenset_iter(paused):
            if exact_type(label) is not str_type:
                raise validation_error_type()
        for label in set_iter(entered):
            if exact_type(label) is not str_type:
                raise validation_error_type()
        for label in set_iter(released):
            if exact_type(label) is not str_type:
                raise validation_error_type()
        for label in set_iter(completed):
            if exact_type(label) is not str_type:
                raise validation_error_type()
        for label in list_iter(visits):
            if exact_type(label) is not str_type:
                raise validation_error_type()
        for label in frozenset_iter(paused):
            if not frozenset_contains(scheduled, label):
                raise validation_error_type()
        for labels in (entered, released, completed):
            for label in set_iter(labels):
                if not frozenset_contains(paused, label):
                    raise validation_error_type()
        for label in set_iter(completed):
            if not set_contains(entered, label):
                raise validation_error_type()
        for label, action in dict_items(action_map):
            if exact_type(label) is not str_type:
                raise validation_error_type()
            validate_action(action)
            action_label = cast_operation(
                str_type, object_getattribute(action, "label")
            )
            action_pause = object_getattribute(action, "pause")
            if (
                label != action_label
                or not frozenset_contains(scheduled, label)
                or action_pause != frozenset_contains(paused, label)
            ):
                raise validation_error_type()
        for label in frozenset_iter(scheduled):
            if not dict_contains(action_map, label) and not list_contains(
                visits, label
            ):
                raise validation_error_type()
        for label in frozenset_iter(paused):
            if set_contains(entered, label) == dict_contains(
                action_map, label
            ):
                raise validation_error_type()

    def fault_controller_init(
        controller: _FaultControllerStorage,
        actions: tuple[FaultAction, ...],
    ) -> None:
        if exact_type(actions) is not tuple_exact_type:
            raise validation_error_type()
        labels_list = list_type()
        paused_list = list_type()
        for item in actions:
            if exact_type(item) is not action_type:
                raise validation_error_type()
            validate_action(item)
            item_label = cast_operation(
                str_type, object_getattribute(item, "label")
            )
            list_append(labels_list, item_label)
            if object_getattribute(item, "pause"):
                list_append(paused_list, item_label)
        labels = tuple_type(labels_list)
        unique_labels = set_type(labels)
        if length(labels) != length(unique_labels):
            raise validation_error_type()
        action_map = dict_type()
        for item in actions:
            dict_setitem(
                action_map,
                object_getattribute(item, "label"),
                item,
            )
        visited = list_type()
        scheduled_labels = frozenset_type(labels)
        paused_labels = frozenset_type(tuple_type(paused_list))
        entered_labels = set_type()
        released_labels = set_type()
        completed_labels = set_type()
        collections: _FaultControllerCollections = (
            action_map,
            visited,
            scheduled_labels,
            paused_labels,
            entered_labels,
            released_labels,
            completed_labels,
        )
        object_setattr(controller, "_actions", action_map)
        object_setattr(controller, "_visited", visited)
        object_setattr(controller, "_scheduled_labels", scheduled_labels)
        object_setattr(controller, "_paused_labels", paused_labels)
        object_setattr(controller, "_entered_labels", entered_labels)
        object_setattr(controller, "_released_labels", released_labels)
        object_setattr(controller, "_completed_labels", completed_labels)
        object_setattr(controller, "_collections", collections)
        object_setattr(controller, "_closed", False)
        validate_fault_controller_state(controller)

    def fault_controller_collections(
        controller: _FaultControllerStorage,
    ) -> _FaultControllerCollections:
        validate_fault_controller_state(controller)
        collections = cast_operation(
            collections_cast_type,
            object_getattribute(controller, "_collections"),
        )
        return collections

    def validate_fault_controller_collections(
        controller: _FaultControllerStorage,
        expected: _FaultControllerCollections,
    ) -> None:
        validate_fault_controller_state(controller)
        if object_getattribute(controller, "_collections") is not expected:
            raise validation_error_type()

    def fault_controller_visited(
        controller: _FaultControllerStorage,
    ) -> tuple[str, ...]:
        validate_fault_controller_state(controller)
        return tuple_type(object_getattribute(controller, "_visited"))

    async def fault_controller_reach(
        controller: _FaultControllerStorage,
        label: str,
    ) -> None:
        expected = fault_controller_collections(controller)
        actions, visited, _, paused, entered, released, completed = expected
        if (
            exact_type(label) is not str_type
            or not label
            or label != str_strip(label)
        ):
            raise validation_error_type()
        if object_getattribute(controller, "_closed"):
            raise validation_error_type()
        list_append(visited, label)
        action = dict_pop(actions, label, None)
        if action is None:
            validate_fault_controller_collections(controller, expected)
            return
        validate_action(action)
        if object_getattribute(action, "pause"):
            set_add(entered, label)
            validate_fault_controller_collections(controller, expected)
            yields = 0
            try:
                while not set_contains(released, label):
                    validate_fault_controller_collections(controller, expected)
                    if (
                        object_getattribute(controller, "_closed")
                        or yields >= max_yields
                    ):
                        raise validation_error_type()
                    yields += 1
                    await async_sleep(0)
                    validate_fault_controller_collections(controller, expected)
            finally:
                set_add(completed, label)
                validate_fault_controller_collections(controller, expected)
        validate_fault_controller_collections(controller, expected)
        exception = object_getattribute(action, "exception")
        if exception is not None:
            raise cast_operation(base_exception_type, exception)

    async def fault_controller_wait_until_entered(
        controller: _FaultControllerStorage,
        label: str,
    ) -> None:
        expected = fault_controller_collections(controller)
        _, _, _, paused, entered, _, _ = expected
        if (
            exact_type(label) is not str_type
            or not label
            or label != str_strip(label)
        ):
            raise validation_error_type()
        if object_getattribute(
            controller, "_closed"
        ) or not frozenset_contains(paused, label):
            raise validation_error_type()
        yields = 0
        while not set_contains(entered, label):
            validate_fault_controller_collections(controller, expected)
            if (
                object_getattribute(controller, "_closed")
                or yields >= max_yields
            ):
                raise validation_error_type()
            yields += 1
            await async_sleep(0)
            validate_fault_controller_collections(controller, expected)
        validate_fault_controller_collections(controller, expected)

    def fault_controller_release(
        controller: _FaultControllerStorage,
        label: str,
    ) -> None:
        expected = fault_controller_collections(controller)
        _, _, _, paused, _, released, completed = expected
        if (
            exact_type(label) is not str_type
            or not label
            or label != str_strip(label)
        ):
            raise validation_error_type()
        if (
            object_getattribute(controller, "_closed")
            or not frozenset_contains(paused, label)
            or set_contains(released, label)
            or set_contains(completed, label)
        ):
            raise validation_error_type()
        set_add(released, label)
        validate_fault_controller_collections(controller, expected)

    def fault_controller_close(controller: _FaultControllerStorage) -> None:
        expected = fault_controller_collections(controller)
        if not object_getattribute(controller, "_closed"):
            object_setattr(controller, "_closed", True)
        validate_fault_controller_collections(controller, expected)

    return (
        validate_fault_controller_state,
        fault_controller_init,
        fault_controller_collections,
        validate_fault_controller_collections,
        fault_controller_visited,
        fault_controller_reach,
        fault_controller_wait_until_entered,
        fault_controller_release,
        fault_controller_close,
    )


(
    _validate_fault_controller_state,
    _fault_controller_init,
    _fault_controller_collections,
    _validate_fault_controller_collections,
    _fault_controller_visited,
    _fault_controller_reach,
    _fault_controller_wait_until_entered,
    _fault_controller_release,
    _fault_controller_close,
) = _build_fault_controller_operations(FaultAction, _validate_fault_action)


_FaultControllerTypeBinder: TypeAlias = Callable[[type[object]], None]


def _build_fault_controller_public_operations(
    initialize: _FaultControllerInitializer,
    visited_reader: _FaultControllerVisitedReader,
    reach_operation: _FaultControllerAsyncLabelOperation,
    wait_operation: _FaultControllerAsyncLabelOperation,
    release_operation: _FaultControllerLabelOperation,
    close_operation: _FaultControllerCloseOperation,
    *,
    exact_type: Callable[[object], type[object]] = _EXACT_TYPE,
    validation_error_type: type[ConversationValidationError] = (
        _VALIDATION_ERROR_TYPE
    ),
) -> tuple[
    _FaultControllerTypeBinder,
    _FaultControllerStateValidator,
    _FaultControllerInitializer,
    _FaultControllerVisitedReader,
    _FaultControllerAsyncLabelOperation,
    _FaultControllerAsyncLabelOperation,
    _FaultControllerLabelOperation,
    _FaultControllerCloseOperation,
]:
    """Build public controller methods over immutable closed operations."""
    controller_type: type[object] | None = None

    def bind_controller_type(value: type[object]) -> None:
        nonlocal controller_type
        controller_type = value

    def validate_public_controller(controller: object) -> None:
        if (
            controller_type is None
            or exact_type(controller) is not controller_type
        ):
            raise validation_error_type()

    def controller_init(
        self: _FaultControllerStorage,
        actions: tuple[FaultAction, ...] = (),
    ) -> None:
        validate_public_controller(self)
        initialize(self, actions)

    def controller_visited(
        self: _FaultControllerStorage,
    ) -> tuple[str, ...]:
        """Return await labels in exact visit order."""
        validate_public_controller(self)
        return visited_reader(self)

    async def controller_reach(
        self: _FaultControllerStorage,
        label: str,
    ) -> None:
        """Visit one label, pause when scheduled, and raise its fault."""
        validate_public_controller(self)
        await reach_operation(self, label)

    async def controller_wait_until_entered(
        self: _FaultControllerStorage,
        label: str,
    ) -> None:
        """Wait until a repository-owned scheduled pause is entered."""
        validate_public_controller(self)
        await wait_operation(self, label)

    def controller_release(
        self: _FaultControllerStorage,
        label: str,
    ) -> None:
        """Release a repository-owned pause before or after entry."""
        validate_public_controller(self)
        release_operation(self, label)

    def controller_close(self: _FaultControllerStorage) -> None:
        """Close rendezvous and stop paused cooperative loops."""
        validate_public_controller(self)
        close_operation(self)

    return (
        bind_controller_type,
        validate_public_controller,
        controller_init,
        controller_visited,
        controller_reach,
        controller_wait_until_entered,
        controller_release,
        controller_close,
    )


(
    _bind_deterministic_fault_controller_type,
    _validate_public_fault_controller,
    _public_fault_controller_init,
    _public_fault_controller_visited,
    _public_fault_controller_reach,
    _public_fault_controller_wait_until_entered,
    _public_fault_controller_release,
    _public_fault_controller_close,
) = _build_fault_controller_public_operations(
    _fault_controller_init,
    _fault_controller_visited,
    _fault_controller_reach,
    _fault_controller_wait_until_entered,
    _fault_controller_release,
    _fault_controller_close,
)


@final
class DeterministicFaultController:
    """Visit scripted await labels and inject deterministic async behavior."""

    __slots__ = (
        "_actions",
        "_closed",
        "_collections",
        "_completed_labels",
        "_entered_labels",
        "_paused_labels",
        "_released_labels",
        "_scheduled_labels",
        "_visited",
    )

    _actions: dict[str, FaultAction]
    _closed: bool
    _collections: _FaultControllerCollections
    _completed_labels: set[str]
    _entered_labels: set[str]
    _paused_labels: frozenset[str]
    _released_labels: set[str]
    _scheduled_labels: frozenset[str]
    _visited: list[str]

    __init__ = _public_fault_controller_init
    visited = property(_public_fault_controller_visited)
    reach = _public_fault_controller_reach
    wait_until_entered = _public_fault_controller_wait_until_entered
    release = _public_fault_controller_release
    close = _public_fault_controller_close


_bind_deterministic_fault_controller_type(DeterministicFaultController)
del _bind_deterministic_fault_controller_type


_DETERMINISTIC_FAULT_CONTROLLER_TYPE = DeterministicFaultController
_DETERMINISTIC_FAULT_CONTROLLER_REACH = _fault_controller_reach


def _build_fault_controller_adapters(
    controller_type: type[DeterministicFaultController],
    validate_state: _FaultControllerStateValidator,
    reach_operation: _FaultControllerAsyncLabelOperation,
    *,
    exact_type: Callable[[object], type[object]] = _EXACT_TYPE,
    validation_error_type: type[ConversationValidationError] = (
        _VALIDATION_ERROR_TYPE
    ),
) -> tuple[
    Callable[[object], None],
    Callable[
        [DeterministicFaultController | None], DeterministicFaultController
    ],
    Callable[[DeterministicFaultController, str], Awaitable[None]],
]:
    """Build closed validation and dispatch adapters."""

    def validate_fault_controller(controller: object) -> None:
        if exact_type(controller) is not controller_type:
            raise validation_error_type()
        validate_state(controller)

    def controller_or_default(
        controller: DeterministicFaultController | None,
    ) -> DeterministicFaultController:
        if controller is None:
            created: DeterministicFaultController = controller_type()
            return created
        validate_fault_controller(controller)
        return controller

    async def reach_fault_controller(
        controller: DeterministicFaultController,
        label: str,
    ) -> None:
        validate_fault_controller(controller)
        await reach_operation(controller, label)

    return (
        validate_fault_controller,
        controller_or_default,
        reach_fault_controller,
    )


(
    _validate_fault_controller,
    _controller_or_default,
    _reach_fault_controller,
) = _build_fault_controller_adapters(
    DeterministicFaultController,
    _validate_fault_controller_state,
    _fault_controller_reach,
)


@final
class FakeStoreBoundaryHook(StoreBoundaryHook):
    """Route store boundaries through one deterministic controller."""

    def __init__(self, controller: DeterministicFaultController) -> None:
        self._controller = _controller_or_default(controller)

    async def reach(
        self,
        boundary: StoreAwaitBoundary,
        _reach: Callable[
            [DeterministicFaultController, str], Awaitable[None]
        ] = _reach_fault_controller,
    ) -> None:
        await _reach(self._controller, f"store:{boundary.value}")


@final
class FakeCoordinatorBoundaryHook(CoordinatorBoundaryHook):
    """Route coordinator boundaries through one deterministic controller."""

    def __init__(self, controller: DeterministicFaultController) -> None:
        self._controller = _controller_or_default(controller)

    async def reach(
        self,
        boundary: CoordinatorAwaitBoundary,
        _reach: Callable[
            [DeterministicFaultController, str], Awaitable[None]
        ] = _reach_fault_controller,
    ) -> None:
        await _reach(self._controller, f"coordinator:{boundary.value}")


@final
class DeterministicFakeAuthorityResolver:
    """Resolve one configured trusted authority asynchronously."""

    def __init__(
        self,
        authority: AuthorityScope,
        controller: DeterministicFaultController | None = None,
    ) -> None:
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        self._authority = authority
        self._controller = _controller_or_default(controller)

    async def resolve(
        self,
        _reach: Callable[
            [DeterministicFaultController, str], Awaitable[None]
        ] = _reach_fault_controller,
    ) -> AuthorityScope:
        await _reach(self._controller, "authority:resolve")
        return self._authority


@final
class DeterministicFakeClock:
    """Return manually advanced aware instants asynchronously."""

    def __init__(
        self,
        value: datetime,
        controller: DeterministicFaultController | None = None,
    ) -> None:
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ConversationValidationError()
        self._value = value
        self._controller = _controller_or_default(controller)

    def set(self, value: datetime) -> None:
        """Set the next deterministic aware instant without I/O."""
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ConversationValidationError()
        self._value = value

    async def now(
        self,
        _reach: Callable[
            [DeterministicFaultController, str], Awaitable[None]
        ] = _reach_fault_controller,
    ) -> datetime:
        await _reach(self._controller, "clock:now")
        return self._value


@final
class DeterministicFakeRetryWaiter:
    """Record bounded retry waits without sleeping."""

    def __init__(
        self, controller: DeterministicFaultController | None = None
    ) -> None:
        self._controller = _controller_or_default(controller)
        self._attempts: list[int] = []

    @property
    def attempts(self) -> tuple[int, ...]:
        """Return waited retry attempt numbers."""
        return tuple(self._attempts)

    async def wait(
        self,
        attempt: int,
        _reach: Callable[
            [DeterministicFaultController, str], Awaitable[None]
        ] = _reach_fault_controller,
    ) -> None:
        if type(attempt) is not int or attempt <= 0:
            raise ConversationValidationError()
        self._attempts.append(attempt)
        await _reach(self._controller, f"retry:{attempt}")


@final
class DeterministicFakeObserver:
    """Record content-safe authoritative lifecycle observations."""

    def __init__(
        self, controller: DeterministicFaultController | None = None
    ) -> None:
        self._controller = _controller_or_default(controller)
        self._observations: list[ConversationObservation] = []

    @property
    def observations(self) -> tuple[ConversationObservation, ...]:
        """Return observations in publication order."""
        return tuple(self._observations)

    async def publish(
        self,
        observation: ConversationObservation,
        _reach: Callable[
            [DeterministicFaultController, str], Awaitable[None]
        ] = _reach_fault_controller,
    ) -> None:
        if type(observation) is not ConversationObservation:
            raise ConversationValidationError()
        await _reach(self._controller, "observer:publish")
        self._observations.append(observation)


@final
class DeterministicFakePublisher:
    """Publish each idempotency intent at most once."""

    def __init__(
        self, controller: DeterministicFaultController | None = None
    ) -> None:
        self._controller = _controller_or_default(controller)
        self._published: dict[str, PublicationIntent] = {}

    @property
    def published(self) -> tuple[PublicationIntent, ...]:
        """Return unique successfully published intents in order."""
        return tuple(self._published.values())

    async def publish(
        self,
        intent: PublicationIntent,
        _reach: Callable[
            [DeterministicFaultController, str], Awaitable[None]
        ] = _reach_fault_controller,
    ) -> None:
        if type(intent) is not PublicationIntent:
            raise ConversationValidationError()
        await _reach(self._controller, "publisher:publish")
        prior = self._published.get(intent.intent_id)
        if prior is not None and prior != intent:
            raise ConversationConflictError()
        self._published[intent.intent_id] = intent


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DeterministicFakeProviderScript:
    """Describe inert deterministic provider results and fault scheduling."""

    results: tuple[ProviderResult, ...]
    controller: DeterministicFaultController | None = None

    def __post_init__(self) -> None:
        _validate_fake_provider_script(self)


_DETERMINISTIC_FAKE_PROVIDER_SCRIPT_TYPE = DeterministicFakeProviderScript


def _validate_frozen_json_value(
    value: object,
    *,
    _depth: int = 0,
    _is_finite: Callable[[float], bool] = isfinite,
    _mapping_type: type[MappingProxyType[str, object]] = MappingProxyType,
) -> None:
    pending = [(value, _depth)]
    while pending:
        current, depth = pending.pop()
        if depth > 32:
            raise ConversationValidationError()
        if current is None or type(current) in {bool, int, str}:
            continue
        if type(current) is float:
            if not _is_finite(current):
                raise ConversationValidationError()
            continue
        if type(current) is tuple:
            pending.extend((item, depth + 1) for item in current)
            continue
        if type(current) is _mapping_type:
            for key, item in current.items():
                if type(key) is not str:
                    raise ConversationValidationError()
                pending.append((item, depth + 1))
            continue
        raise ConversationValidationError()


def _canonical_provider_item(
    item: object,
    *,
    _caller_type: type[ProviderItemCaller] = ProviderItemCaller,
    _item_id_type: type[ProviderItemId] = ProviderItemId,
    _item_type: type[ProviderItem] = ProviderItem,
    _kind_type: type[ProviderItemKind] = ProviderItemKind,
    _mapping_type: type[MappingProxyType[str, object]] = MappingProxyType,
    _model_call_id_type: type[ConversationModelCallId] = (
        ConversationModelCallId
    ),
    _opaque_state_type: type[OpaqueProviderState] = OpaqueProviderState,
    _order_type: type[ProviderItemOrder] = ProviderItemOrder,
    _phase_type: type[ProviderItemPhase] = ProviderItemPhase,
    _provider_index_type: type[ProviderItemIndex] = ProviderItemIndex,
    _validate_json: Callable[[object], None] = _validate_frozen_json_value,
) -> ProviderItem:
    if type(item) is not _item_type:
        raise ConversationValidationError()
    try:
        item_id = item.item_id
        lane_id = item.lane_id
        model_call_id = item.model_call_id
        kind = item.kind
        order = item.order
        provider_index = item.provider_index
        phase = item.phase
        caller = item.caller
        canonical_input = item.canonical_input
        normalization_version = item.normalization_version
        call_id = item.call_id
        opaque_state = item.opaque_state
        complete = item.complete
    except AttributeError as exc:
        raise ConversationValidationError() from exc
    if (
        type(item_id) is not str
        or type(lane_id) is not str
        or type(model_call_id) is not str
        or type(kind) is not _kind_type
        or type(order) is not int
        or type(provider_index) is not int
        or type(phase) is not _phase_type
        or type(caller) is not _caller_type
        or type(canonical_input) is not _mapping_type
        or type(normalization_version) is not int
        or (call_id is not None and type(call_id) is not str)
        or complete is not True
    ):
        raise ConversationValidationError()
    _validate_json(canonical_input)
    canonical_opaque_state = None
    if opaque_state is not None:
        if type(opaque_state) is not _opaque_state_type:
            raise ConversationValidationError()
        try:
            opaque_value = opaque_state._value
        except AttributeError as exc:
            raise ConversationValidationError() from exc
        canonical_opaque_state = _opaque_state_type(_value=opaque_value)
    return _item_type(
        item_id=_item_id_type(item_id),
        lane_id=lane_id,
        model_call_id=_model_call_id_type(model_call_id),
        kind=kind,
        order=_order_type(order),
        provider_index=_provider_index_type(provider_index),
        phase=phase,
        caller=caller,
        canonical_input=cast(Mapping[str, JsonValue], canonical_input),
        normalization_version=normalization_version,
        call_id=call_id,
        opaque_state=canonical_opaque_state,
        complete=complete,
    )


def _canonical_provider_result(
    result: object,
    *,
    _canonical_item: Callable[[object], ProviderItem] = (
        _canonical_provider_item
    ),
    _effective_reasoning_type: type[EffectiveReasoningContext] = (
        EffectiveReasoningContext
    ),
    _reasoning_context_type: type[ReasoningContext] = ReasoningContext,
    _reasoning_metadata_type: type[EffectiveReasoningMetadata] = (
        EffectiveReasoningMetadata
    ),
    _result_type: type[ProviderResult] = ProviderResult,
    _usage_type: type[ProviderUsage] = ProviderUsage,
) -> ProviderResult:
    if type(result) is not _result_type:
        raise ConversationValidationError()
    try:
        items = result.items
        reasoning = result.reasoning
        upstream_response_id = result.upstream_response_id
        usage = result.usage
    except AttributeError as exc:
        raise ConversationValidationError() from exc
    if type(items) is not tuple:
        raise ConversationValidationError()
    canonical_items = tuple(_canonical_item(item) for item in items)
    if type(reasoning) is not _reasoning_metadata_type:
        raise ConversationValidationError()
    try:
        requested = reasoning.requested
        effective = reasoning.effective
    except AttributeError as exc:
        raise ConversationValidationError() from exc
    if type(requested) is not _reasoning_context_type or (
        effective is not None
        and type(effective) is not _effective_reasoning_type
    ):
        raise ConversationValidationError()
    canonical_reasoning = _reasoning_metadata_type(
        requested=requested,
        effective=effective,
    )
    if type(usage) is not _usage_type:
        raise ConversationValidationError()
    try:
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
    except AttributeError as exc:
        raise ConversationValidationError() from exc
    canonical_usage = _usage_type(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    if (
        upstream_response_id is not None
        and type(upstream_response_id) is not str
    ):
        raise ConversationValidationError()
    return _result_type(
        items=canonical_items,
        reasoning=canonical_reasoning,
        upstream_response_id=upstream_response_id,
        usage=canonical_usage,
    )


def _validate_fake_provider_script(
    script: object,
    *,
    _canonical_result: Callable[[object], ProviderResult] = (
        _canonical_provider_result
    ),
    _script_type: type[DeterministicFakeProviderScript] = (
        _DETERMINISTIC_FAKE_PROVIDER_SCRIPT_TYPE
    ),
    _result_type: type[ProviderResult] = ProviderResult,
    _validate_controller: Callable[[object], None] = (
        _validate_fault_controller
    ),
) -> None:
    if type(script) is not _script_type:
        raise ConversationValidationError()
    try:
        results = script.results
        controller = script.controller
    except AttributeError as exc:
        raise ConversationValidationError() from exc
    if (
        type(results) is not tuple
        or not results
        or any(type(item) is not _result_type for item in results)
    ):
        raise ConversationValidationError()
    for result in results:
        _canonical_result(result)
    if controller is not None:
        _validate_controller(controller)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DeterministicFakeProviderStreamDiagnostics:
    """Report one canonical fake stream without exposing mutable state."""

    item_count: int
    consumed_items: int
    close_attempts: int
    closed: bool


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DeterministicFakeProviderDiagnostics:
    """Report immutable canonical fake execution diagnostics."""

    plans: tuple[ProviderPlan, ...]
    streams: tuple[DeterministicFakeProviderStreamDiagnostics, ...]
    remaining_results: int


@final
@dataclass(slots=True)
class _DeterministicFakeProviderStreamState:
    result: ProviderResult
    index: int = 0
    close_attempts: int = 0
    closed: bool = False


@final
@dataclass(slots=True)
class _DeterministicFakeProviderRuntime:
    """Own mutable state created from canonicalized inert script data."""

    script: DeterministicFakeProviderScript
    results: list[ProviderResult]
    controller: DeterministicFaultController
    plans: list[ProviderPlan]
    streams: list[_DeterministicFakeProviderStreamState]


_DETERMINISTIC_FAKE_PROVIDER_RUNTIME_TYPE = _DeterministicFakeProviderRuntime
_DETERMINISTIC_FAKE_PROVIDER_STREAM_STATE_TYPE = (
    _DeterministicFakeProviderStreamState
)


def _is_provider_plan(
    plan: object,
    *,
    _plan_types: frozenset[type[object]] = frozenset(
        {
            StatelessProviderPlan,
            StandaloneCompactProviderPlan,
            FirstStoredProviderPlan,
            StoredProviderPlan,
        }
    ),
) -> bool:
    return type(plan) in _plan_types


def _validate_provider_plan(
    plan: object,
    *,
    _is_plan: Callable[[object], bool] = _is_provider_plan,
) -> None:
    if not _is_plan(plan):
        raise ConversationValidationError()


def _validate_deterministic_fake_provider_stream(
    stream: object,
    *,
    _canonical_result: Callable[[object], ProviderResult] = (
        _canonical_provider_result
    ),
    _stream_type: type[_DeterministicFakeProviderStreamState] = (
        _DETERMINISTIC_FAKE_PROVIDER_STREAM_STATE_TYPE
    ),
) -> None:
    if type(stream) is not _stream_type:
        raise ConversationValidationError()
    try:
        result = stream.result
        index = stream.index
        close_attempts = stream.close_attempts
        closed = stream.closed
    except AttributeError as exc:
        raise ConversationValidationError() from exc
    _canonical_result(result)
    if (
        type(index) is not int
        or index < 0
        or index > len(result.items)
        or type(close_attempts) is not int
        or close_attempts < 0
        or type(closed) is not bool
    ):
        raise ConversationValidationError()


def _next_deterministic_fake_provider_result(
    runtime: _DeterministicFakeProviderRuntime,
    *,
    _canonical_result: Callable[[object], ProviderResult] = (
        _canonical_provider_result
    ),
) -> ProviderResult:
    if not runtime.results:
        raise ConversationConflictError()
    return _canonical_result(runtime.results.pop(0))


def _owned_deterministic_fake_provider_stream(
    runtime: _DeterministicFakeProviderRuntime,
    stream: _DeterministicFakeProviderStreamState,
    *,
    _validate_stream: Callable[[object], None] = (
        _validate_deterministic_fake_provider_stream
    ),
) -> _DeterministicFakeProviderStreamState:
    _validate_stream(stream)
    if not any(item is stream for item in runtime.streams):
        raise ConversationValidationError()
    return stream


def _validate_deterministic_fake_provider_runtime(
    runtime: object,
    script: DeterministicFakeProviderScript,
    *,
    _canonical_result: Callable[[object], ProviderResult] = (
        _canonical_provider_result
    ),
    _is_plan: Callable[[object], bool] = _is_provider_plan,
    _runtime_type: type[_DeterministicFakeProviderRuntime] = (
        _DETERMINISTIC_FAKE_PROVIDER_RUNTIME_TYPE
    ),
    _result_type: type[ProviderResult] = ProviderResult,
    _validate_controller: Callable[[object], None] = (
        _validate_fault_controller
    ),
    _validate_script: Callable[[object], None] = (
        _validate_fake_provider_script
    ),
    _validate_stream: Callable[[object], None] = (
        _validate_deterministic_fake_provider_stream
    ),
) -> _DeterministicFakeProviderRuntime:
    _validate_script(script)
    if type(runtime) is not _runtime_type:
        raise ConversationValidationError()
    try:
        source_script = runtime.script
        results = runtime.results
        controller = runtime.controller
        plans = runtime.plans
        streams = runtime.streams
    except AttributeError as exc:
        raise ConversationValidationError() from exc
    if source_script is not script:
        raise ConversationValidationError()
    if (
        type(results) is not list
        or any(type(item) is not _result_type for item in results)
        or type(plans) is not list
        or any(not _is_plan(item) for item in plans)
        or type(streams) is not list
    ):
        raise ConversationValidationError()
    for result in results:
        _canonical_result(result)
    _validate_controller(controller)
    for stream in streams:
        _validate_stream(stream)
    return runtime


def _build_deterministic_fake_provider_runtime(
    script: DeterministicFakeProviderScript,
    *,
    _canonical_result: Callable[[object], ProviderResult] = (
        _canonical_provider_result
    ),
    _controller: Callable[
        [DeterministicFaultController | None], DeterministicFaultController
    ] = _controller_or_default,
    _runtime_type: type[_DeterministicFakeProviderRuntime] = (
        _DETERMINISTIC_FAKE_PROVIDER_RUNTIME_TYPE
    ),
    _validate_script: Callable[[object], None] = (
        _validate_fake_provider_script
    ),
) -> _DeterministicFakeProviderRuntime:
    _validate_script(script)
    return _runtime_type(
        script=script,
        results=[_canonical_result(result) for result in script.results],
        controller=_controller(script.controller),
        plans=[],
        streams=[],
    )


async def _dispatch_deterministic_fake_provider(
    runtime: _DeterministicFakeProviderRuntime,
    script: DeterministicFakeProviderScript,
    plan: ProviderPlan,
    *,
    _next_result: Callable[
        [_DeterministicFakeProviderRuntime], ProviderResult
    ] = _next_deterministic_fake_provider_result,
    _reach: Callable[
        [DeterministicFaultController, str], Awaitable[None]
    ] = _reach_fault_controller,
    _validate_plan: Callable[[object], None] = _validate_provider_plan,
    _validate_runtime: Callable[
        [object, DeterministicFakeProviderScript],
        _DeterministicFakeProviderRuntime,
    ] = (_validate_deterministic_fake_provider_runtime),
) -> ProviderResult:
    state = _validate_runtime(runtime, script)
    _validate_plan(plan)
    state.plans.append(plan)
    await _reach(state.controller, "provider:dispatch")
    state = _validate_runtime(runtime, script)
    return _next_result(state)


async def _open_deterministic_fake_provider_stream(
    runtime: _DeterministicFakeProviderRuntime,
    script: DeterministicFakeProviderScript,
    plan: ProviderPlan,
    *,
    _next_result: Callable[
        [_DeterministicFakeProviderRuntime], ProviderResult
    ] = _next_deterministic_fake_provider_result,
    _reach: Callable[
        [DeterministicFaultController, str], Awaitable[None]
    ] = _reach_fault_controller,
    _stream_type: type[_DeterministicFakeProviderStreamState] = (
        _DETERMINISTIC_FAKE_PROVIDER_STREAM_STATE_TYPE
    ),
    _validate_plan: Callable[[object], None] = _validate_provider_plan,
    _validate_runtime: Callable[
        [object, DeterministicFakeProviderScript],
        _DeterministicFakeProviderRuntime,
    ] = (_validate_deterministic_fake_provider_runtime),
) -> _DeterministicFakeProviderStreamState:
    state = _validate_runtime(runtime, script)
    _validate_plan(plan)
    state.plans.append(plan)
    await _reach(state.controller, "provider:stream")
    state = _validate_runtime(runtime, script)
    stream = _stream_type(result=_next_result(state))
    state.streams.append(stream)
    return stream


async def _next_deterministic_fake_provider_item(
    runtime: _DeterministicFakeProviderRuntime,
    script: DeterministicFakeProviderScript,
    stream: _DeterministicFakeProviderStreamState,
    *,
    _owned_stream: Callable[
        [
            _DeterministicFakeProviderRuntime,
            _DeterministicFakeProviderStreamState,
        ],
        _DeterministicFakeProviderStreamState,
    ] = _owned_deterministic_fake_provider_stream,
    _reach: Callable[
        [DeterministicFaultController, str], Awaitable[None]
    ] = _reach_fault_controller,
    _validate_runtime: Callable[
        [object, DeterministicFakeProviderScript],
        _DeterministicFakeProviderRuntime,
    ] = (_validate_deterministic_fake_provider_runtime),
) -> ProviderItem:
    state = _validate_runtime(runtime, script)
    stream_state = _owned_stream(state, stream)
    if stream_state.closed or stream_state.index >= len(
        stream_state.result.items
    ):
        raise StopAsyncIteration
    await _reach(state.controller, f"provider:item:{stream_state.index}")
    state = _validate_runtime(runtime, script)
    stream_state = _owned_stream(state, stream)
    item = stream_state.result.items[stream_state.index]
    stream_state.index += 1
    return item


async def _terminal_deterministic_fake_provider_stream(
    runtime: _DeterministicFakeProviderRuntime,
    script: DeterministicFakeProviderScript,
    stream: _DeterministicFakeProviderStreamState,
    *,
    _owned_stream: Callable[
        [
            _DeterministicFakeProviderRuntime,
            _DeterministicFakeProviderStreamState,
        ],
        _DeterministicFakeProviderStreamState,
    ] = _owned_deterministic_fake_provider_stream,
    _reach: Callable[
        [DeterministicFaultController, str], Awaitable[None]
    ] = _reach_fault_controller,
    _validate_runtime: Callable[
        [object, DeterministicFakeProviderScript],
        _DeterministicFakeProviderRuntime,
    ] = (_validate_deterministic_fake_provider_runtime),
) -> ProviderResult:
    state = _validate_runtime(runtime, script)
    stream_state = _owned_stream(state, stream)
    await _reach(state.controller, "provider:terminal")
    state = _validate_runtime(runtime, script)
    stream_state = _owned_stream(state, stream)
    if stream_state.index != len(stream_state.result.items):
        raise ConversationValidationError()
    return stream_state.result


async def _close_deterministic_fake_provider_stream(
    runtime: _DeterministicFakeProviderRuntime,
    script: DeterministicFakeProviderScript,
    stream: _DeterministicFakeProviderStreamState,
    *,
    _owned_stream: Callable[
        [
            _DeterministicFakeProviderRuntime,
            _DeterministicFakeProviderStreamState,
        ],
        _DeterministicFakeProviderStreamState,
    ] = _owned_deterministic_fake_provider_stream,
    _reach: Callable[
        [DeterministicFaultController, str], Awaitable[None]
    ] = _reach_fault_controller,
    _validate_runtime: Callable[
        [object, DeterministicFakeProviderScript],
        _DeterministicFakeProviderRuntime,
    ] = (_validate_deterministic_fake_provider_runtime),
) -> None:
    state = _validate_runtime(runtime, script)
    stream_state = _owned_stream(state, stream)
    stream_state.close_attempts += 1
    label = (
        "provider:close"
        if stream_state.close_attempts == 1
        else f"provider:close:retry:{stream_state.close_attempts - 1}"
    )
    try:
        await _reach(state.controller, label)
    finally:
        state = _validate_runtime(runtime, script)
        stream_state = _owned_stream(state, stream)
        stream_state.closed = True


def _deterministic_fake_provider_diagnostics(
    runtime: _DeterministicFakeProviderRuntime,
    script: DeterministicFakeProviderScript,
    *,
    _diagnostics_type: type[DeterministicFakeProviderDiagnostics] = (
        DeterministicFakeProviderDiagnostics
    ),
    _stream_diagnostics_type: type[
        DeterministicFakeProviderStreamDiagnostics
    ] = DeterministicFakeProviderStreamDiagnostics,
    _validate_runtime: Callable[
        [object, DeterministicFakeProviderScript],
        _DeterministicFakeProviderRuntime,
    ] = (_validate_deterministic_fake_provider_runtime),
) -> DeterministicFakeProviderDiagnostics:
    state = _validate_runtime(runtime, script)
    return _diagnostics_type(
        plans=tuple(state.plans),
        streams=tuple(
            _stream_diagnostics_type(
                item_count=len(stream.result.items),
                consumed_items=stream.index,
                close_attempts=stream.close_attempts,
                closed=stream.closed,
            )
            for stream in state.streams
        ),
        remaining_results=len(state.results),
    )


def fake_capability_profile(
    binding: ProviderLaneBinding,
) -> ConversationCapabilityProfile:
    """Return an all-capability test-only profile for one synthetic binding."""
    if type(binding) is not ProviderLaneBinding:
        raise ConversationValidationError()
    return ConversationCapabilityProfile(
        profile_id=CapabilityProfileId(f"fake-{binding.lane_id}"),
        schema_version=1,
        revision=binding.capability_profile_revision,
        binding_alias=binding.safe_alias,
        capabilities=tuple(
            CapabilityEvidence(
                capability=capability,
                state=CapabilityEvidenceState.TEST_ONLY,
                evidence_ids=(f"fake-{capability.value}",),
            )
            for capability in ConversationCapability
        ),
        test_only=True,
    )


def fake_provider_result(
    plan: ProviderPlan,
    *,
    turn: int,
    text: str = "synthetic-output",
) -> ProviderResult:
    """Return one deterministic complete assistant item for an exact plan."""
    if not isinstance(
        plan,
        StatelessProviderPlan
        | StandaloneCompactProviderPlan
        | FirstStoredProviderPlan
        | StoredProviderPlan,
    ):
        raise ConversationValidationError()
    if type(turn) is not int or turn <= 0 or not text:
        raise ConversationValidationError()
    order = (
        len(plan.ledger.items)
        if isinstance(plan, StatelessProviderPlan)
        else 0
    )
    item_id = ProviderItemId(f"fake-item-{plan.binding.lane_id}-{turn}")
    item = ProviderItem(
        item_id=item_id,
        lane_id=plan.binding.lane_id,
        model_call_id=ConversationModelCallId(
            f"fake-model-call-{plan.binding.lane_id}-{turn}"
        ),
        kind=ProviderItemKind.MESSAGE,
        order=ProviderItemOrder(order),
        provider_index=ProviderItemIndex(0),
        phase=ProviderItemPhase.FINAL,
        caller=ProviderItemCaller.PROVIDER,
        canonical_input={
            "content": (
                {
                    "annotations": (),
                    "text": text,
                    "type": "output_text",
                },
            ),
            "id": item_id,
            "role": "assistant",
            "status": "completed",
            "type": "message",
        },
        normalization_version=PROVIDER_ITEM_NORMALIZATION_VERSION,
    )
    return ProviderResult(
        items=(item,),
        reasoning=EffectiveReasoningMetadata(
            requested=plan.reasoning.requested,
            effective=EffectiveReasoningContext.CURRENT_TURN,
        ),
        upstream_response_id=(
            UpstreamResponseId(f"fake-upstream-{plan.binding.lane_id}-{turn}")
            if isinstance(plan, FirstStoredProviderPlan | StoredProviderPlan)
            else None
        ),
        usage=ProviderUsage(input_tokens=turn * 10, output_tokens=turn * 5),
    )


def fake_compaction_result(
    plan: StatelessProviderPlan | StandaloneCompactProviderPlan,
    *,
    turn: int,
    opaque_state: bytes = b"synthetic-compaction",
) -> ProviderResult:
    """Return one deterministic fake standalone compaction result."""
    if not isinstance(
        plan, StatelessProviderPlan | StandaloneCompactProviderPlan
    ):
        raise ConversationValidationError()
    if type(turn) is not int or turn <= 0:
        raise ConversationValidationError()
    item_id = ProviderItemId(f"fake-compaction-{plan.binding.lane_id}-{turn}")
    item = ProviderItem(
        item_id=item_id,
        lane_id=plan.binding.lane_id,
        model_call_id=ConversationModelCallId(
            f"fake-compact-call-{plan.binding.lane_id}-{turn}"
        ),
        kind=ProviderItemKind.COMPACTION,
        order=ProviderItemOrder(0),
        provider_index=ProviderItemIndex(0),
        phase=ProviderItemPhase.COMPACTION,
        caller=ProviderItemCaller.PROVIDER,
        canonical_input={"id": item_id, "type": "compaction"},
        normalization_version=PROVIDER_ITEM_NORMALIZATION_VERSION,
        opaque_state=OpaqueProviderState(_value=opaque_state),
    )
    return ProviderResult(
        items=(item,),
        reasoning=EffectiveReasoningMetadata(
            requested=plan.reasoning.requested,
            effective=EffectiveReasoningContext.CURRENT_TURN,
        ),
        usage=ProviderUsage(input_tokens=turn * 10, output_tokens=0),
    )
