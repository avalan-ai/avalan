"""Keep rejected store results statically bound to their operation."""

from typing import cast

from avalan.interaction import (
    CancelInteractionCommand,
    CancelInteractionRejected,
    ControllerActivityRejected,
    CreateInteractionCommand,
    CreateInteractionRejected,
    DetachInteractionCommand,
    DueInteractionsRejected,
    InputTransitionError,
    InteractionPresentationRejected,
    RecordControllerActivityCommand,
    ScopeCancellationRejected,
    ScopeSupersessionRejected,
    SupersedeInteractionScopeCommand,
    TerminalizeDueInteractionsCommand,
    TerminalizeInteractionCommand,
    TerminalizeInteractionRejected,
    TerminalizeInteractionScopeCommand,
)

error = cast(InputTransitionError, object())
create = cast(CreateInteractionCommand, object())
cancel = cast(CancelInteractionCommand, object())
terminalize = cast(TerminalizeInteractionCommand, object())
cancel_scope = cast(TerminalizeInteractionScopeCommand, object())
supersede_scope = cast(SupersedeInteractionScopeCommand, object())
due = cast(TerminalizeDueInteractionsCommand, object())
detach = cast(DetachInteractionCommand, object())
activity = cast(RecordControllerActivityCommand, object())

CreateInteractionRejected(command=cancel, error=error)
CancelInteractionRejected(command=create, error=error)
TerminalizeInteractionRejected(command=cancel, error=error)
ScopeCancellationRejected(command=supersede_scope, error=error)
ScopeSupersessionRejected(command=cancel_scope, error=error)
DueInteractionsRejected(command=create, error=error)
InteractionPresentationRejected(command=activity, error=error)
ControllerActivityRejected(command=detach, error=error)
