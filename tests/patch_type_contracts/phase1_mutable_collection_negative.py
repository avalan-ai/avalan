"""Reject mutable authority collections at the immutable domain boundary."""

from avalan.patch.domain import Capability, Lineage, MutationScope


def construct_mutable_scope(scope: MutationScope) -> MutationScope:
    """Attempt to reconstruct scope with caller-owned mutable capabilities."""
    return MutationScope(
        context_kind=scope.context_kind,
        context_id=scope.context_id,
        workspace_id=scope.workspace_id,
        domain_id=scope.domain_id,
        target_id=scope.target_id,
        protocol_id=scope.protocol_id,
        capabilities={Capability.UPDATE},
        disclosures=scope.disclosures,
        limits=scope.limits,
    )


def construct_mutable_lineage(lineage: Lineage) -> Lineage:
    """Attempt to reconstruct lineage with mutable required capabilities."""
    return Lineage(
        lineage_id=lineage.lineage_id,
        source_path=lineage.source_path,
        destination_path=lineage.destination_path,
        required_capabilities={Capability.UPDATE},
        match_strategy=lineage.match_strategy,
        commit_graph=lineage.commit_graph,
    )
