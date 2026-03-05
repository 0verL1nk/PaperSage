from agent.domain.orchestration import (
    OrchestratedTurn as DomainOrchestratedTurn,
    PolicyDecision as DomainPolicyDecision,
    TeamExecution as DomainTeamExecution,
    TeamRole as DomainTeamRole,
)
from agent.orchestration.contracts import (
    OrchestratedTurn,
    PolicyDecision,
    TeamExecution,
    TeamRole,
)


def test_orchestration_contracts_reexport_domain_types():
    assert PolicyDecision is DomainPolicyDecision
    assert TeamRole is DomainTeamRole
    assert TeamExecution is DomainTeamExecution
    assert OrchestratedTurn is DomainOrchestratedTurn
