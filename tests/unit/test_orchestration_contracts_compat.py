from agent.domain.orchestration import (
    OrchestratedTurn as DomainOrchestratedTurn,
)
from agent.domain.orchestration import (
    PolicyDecision as DomainPolicyDecision,
)
from agent.domain.orchestration import (
    TeamExecution as DomainTeamExecution,
)
from agent.domain.orchestration import (
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
