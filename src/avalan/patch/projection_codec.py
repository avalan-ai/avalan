"""Declare detached immutable audience-projection wire values.

This lower-consumer module deliberately contains no parser, verifier, root
anchor, trusted-boundary import, or acceptance callback. Trusted verification
and host delivery live only in :mod:`avalan.patch.projection`.
"""

from typing import NewType

PROJECTION_ENVELOPE_SCHEMA_VERSION = 2
PROJECTION_RECEIPT_SCHEMA_VERSION = 1
PROJECTION_DELIVERY_SCHEMA_VERSION = 1
MAX_PROJECTION_ENVELOPE_BYTES = 1_048_576
MAX_PROJECTION_RECEIPT_BYTES = 4_096

ModelProjectionEnvelope = NewType("ModelProjectionEnvelope", bytes)
ApproverProjectionEnvelope = NewType("ApproverProjectionEnvelope", bytes)
AuditProjectionEnvelope = NewType("AuditProjectionEnvelope", bytes)
ModelProjectionVerificationReceipt = NewType(
    "ModelProjectionVerificationReceipt", bytes
)
ApproverProjectionVerificationReceipt = NewType(
    "ApproverProjectionVerificationReceipt", bytes
)
AuditProjectionVerificationReceipt = NewType(
    "AuditProjectionVerificationReceipt", bytes
)
ModelProjectionDelivery = NewType("ModelProjectionDelivery", bytes)
ApproverProjectionDelivery = NewType("ApproverProjectionDelivery", bytes)
AuditProjectionDelivery = NewType("AuditProjectionDelivery", bytes)
GenericToolProgressDelivery = NewType("GenericToolProgressDelivery", bytes)
AuditRecordDelivery = NewType("AuditRecordDelivery", bytes)
MetricsRecordDelivery = NewType("MetricsRecordDelivery", bytes)
TelemetryRecordDelivery = NewType("TelemetryRecordDelivery", bytes)
ServerRecordDelivery = NewType("ServerRecordDelivery", bytes)
