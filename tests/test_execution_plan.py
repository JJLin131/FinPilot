from __future__ import annotations

import threading

import pytest

from finpilot.agent.orchestration import (
    AgentRegistration,
    ExecutionPlan,
    ExecutionPlanNode,
    ExecutionPlanningService,
    ExecutionScheduler,
)
from finpilot.models import SessionContext, SubAgentResult


REGISTERED_AGENTS = {"QueryAgent", "TreasuryDataAgent", "TreasuryOperationAgent"}


def test_execution_plan_accepts_registered_dag():
    plan = ExecutionPlan(
        nodes=[
            ExecutionPlanNode(node_id="knowledge", agent_name="QueryAgent", task="查询适用规则"),
            ExecutionPlanNode(
                node_id="account",
                agent_name="TreasuryDataAgent",
                task="查询付款账户余额",
            ),
            ExecutionPlanNode(
                node_id="transfer",
                agent_name="TreasuryOperationAgent",
                task="根据已确认的账户信息创建转账订单",
                depends_on=["account"],
            ),
        ]
    )

    plan.validate_for_registry(REGISTERED_AGENTS)


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        ([ExecutionPlanNode(node_id="bad", agent_name="UnknownAgent", task="x")], "unregistered agent"),
        (
            [
                ExecutionPlanNode(node_id="a", agent_name="QueryAgent", task="x", depends_on=["missing"]),
            ],
            "unknown dependency",
        ),
        (
            [
                ExecutionPlanNode(node_id="a", agent_name="QueryAgent", task="x", depends_on=["b"]),
                ExecutionPlanNode(node_id="b", agent_name="TreasuryDataAgent", task="x", depends_on=["a"]),
            ],
            "cycle",
        ),
        (
            [
                ExecutionPlanNode(node_id="same", agent_name="QueryAgent", task="x"),
                ExecutionPlanNode(node_id="same", agent_name="TreasuryDataAgent", task="x"),
            ],
            "unique",
        ),
    ],
)
def test_execution_plan_rejects_invalid_registry_or_dependencies(nodes, message):
    plan = ExecutionPlan(nodes=nodes)

    with pytest.raises(ValueError, match=message):
        plan.validate_for_registry(REGISTERED_AGENTS)


def test_execution_plan_rejects_more_than_three_nodes():
    plan = ExecutionPlan(
        nodes=[
            ExecutionPlanNode(node_id=f"node-{index}", agent_name="QueryAgent", task="x")
            for index in range(4)
        ]
    )

    with pytest.raises(ValueError, match="maximum node count"):
        plan.validate_for_registry(REGISTERED_AGENTS)


def test_session_context_keeps_structured_subagent_results_without_raw_evidence():
    result = SubAgentResult(
        node_id="account",
        agent_name="TreasuryDataAgent",
        task="查询余额",
        status="SUCCEEDED",
        summary="账户可用余额为 1000 元。",
        output={"account_id": "acct-1", "available_balance": 1000},
        evidence_summary=[{"source": "treasury-api", "title": "账户余额"}],
    )
    session = SessionContext(
        user_id="user-1",
        chat_id="chat-1",
        memory_id="memory-1",
        user_message="查询余额后转账",
        normalized_intent="TREASURY_OPERATION",
        target_agent="TreasuryOperationAgent",
        subagent_results=[result],
    )

    payload = session.model_dump(mode="json")

    assert payload["subagent_results"][0]["output"]["account_id"] == "acct-1"
    assert "raw_evidence" not in payload["subagent_results"][0]


def _session() -> SessionContext:
    return SessionContext(
        user_id="user-1",
        chat_id="chat-1",
        memory_id="memory-1",
        user_message="完成任务",
        normalized_intent="FINANCE_KNOWLEDGE_QA",
        target_agent="QueryAgent",
    )


def test_scheduler_runs_independent_read_only_nodes_in_parallel():
    barrier = threading.Barrier(2, timeout=1)
    started: list[str] = []

    def execute(node: ExecutionPlanNode, session: SessionContext) -> SubAgentResult:
        del session
        started.append(node.node_id)
        barrier.wait()
        return SubAgentResult(node_id=node.node_id, agent_name=node.agent_name, task=node.task, status="SUCCEEDED")

    scheduler = ExecutionScheduler(
        {
            "QueryAgent": AgentRegistration("QueryAgent", "read_only", execute),
            "TreasuryDataAgent": AgentRegistration("TreasuryDataAgent", "read_only", execute),
        }
    )
    plan = ExecutionPlan(
        nodes=[
            ExecutionPlanNode(node_id="knowledge", agent_name="QueryAgent", task="查规则"),
            ExecutionPlanNode(node_id="account", agent_name="TreasuryDataAgent", task="查余额"),
        ]
    )

    results = scheduler.execute(plan, _session())

    assert set(started) == {"knowledge", "account"}
    assert [result.status for result in results] == ["SUCCEEDED", "SUCCEEDED"]


def test_scheduler_passes_completed_dependency_results_to_next_layer():
    def account(node: ExecutionPlanNode, session: SessionContext) -> SubAgentResult:
        assert session.subagent_results == []
        return SubAgentResult(
            node_id=node.node_id,
            agent_name=node.agent_name,
            task=node.task,
            status="SUCCEEDED",
            output={"account_id": "acct-1"},
        )

    def transfer(node: ExecutionPlanNode, session: SessionContext) -> SubAgentResult:
        assert session.subagent_results[0].output["account_id"] == "acct-1"
        return SubAgentResult(node_id=node.node_id, agent_name=node.agent_name, task=node.task, status="SUCCEEDED")

    scheduler = ExecutionScheduler(
        {
            "TreasuryDataAgent": AgentRegistration("TreasuryDataAgent", "read_only", account),
            "TreasuryOperationAgent": AgentRegistration("TreasuryOperationAgent", "operation", transfer),
        }
    )
    plan = ExecutionPlan(
        nodes=[
            ExecutionPlanNode(node_id="account", agent_name="TreasuryDataAgent", task="查余额"),
            ExecutionPlanNode(
                node_id="transfer",
                agent_name="TreasuryOperationAgent",
                task="创建转账",
                depends_on=["account"],
            ),
        ]
    )

    results = scheduler.execute(plan, _session())

    assert [result.node_id for result in results] == ["account", "transfer"]
    assert all(result.status == "SUCCEEDED" for result in results)


def test_scheduler_skips_failed_dependencies_and_keeps_independent_branch():
    def fail(node: ExecutionPlanNode, session: SessionContext) -> SubAgentResult:
        del session
        return SubAgentResult(
            node_id=node.node_id,
            agent_name=node.agent_name,
            task=node.task,
            status="FAILED",
            failure_reason="账户不可用",
        )

    def succeed(node: ExecutionPlanNode, session: SessionContext) -> SubAgentResult:
        del session
        return SubAgentResult(node_id=node.node_id, agent_name=node.agent_name, task=node.task, status="SUCCEEDED")

    scheduler = ExecutionScheduler(
        {
            "TreasuryDataAgent": AgentRegistration("TreasuryDataAgent", "read_only", fail),
            "TreasuryOperationAgent": AgentRegistration("TreasuryOperationAgent", "operation", succeed),
            "QueryAgent": AgentRegistration("QueryAgent", "read_only", succeed),
        }
    )
    plan = ExecutionPlan(
        nodes=[
            ExecutionPlanNode(node_id="account", agent_name="TreasuryDataAgent", task="查余额"),
            ExecutionPlanNode(
                node_id="transfer",
                agent_name="TreasuryOperationAgent",
                task="创建转账",
                depends_on=["account"],
            ),
            ExecutionPlanNode(node_id="knowledge", agent_name="QueryAgent", task="查规则"),
        ]
    )

    results = scheduler.execute(plan, _session())

    statuses = {result.node_id: result.status for result in results}
    assert statuses == {"account": "FAILED", "knowledge": "SUCCEEDED", "transfer": "SKIPPED"}


class FakePlanningClient:
    def __init__(self, response: str | list[str]) -> None:
        self.responses = [response] if isinstance(response, str) else list(response)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, model_name: str) -> str:
        del model_name
        self.prompts.append(prompt)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def test_planning_service_validates_llm_plan_against_registered_agents():
    client = FakePlanningClient(
        '{"nodes":[{"node_id":"account","agent_name":"TreasuryDataAgent","task":"查询账户余额"},'
        '{"node_id":"transfer","agent_name":"TreasuryOperationAgent","task":"创建转账",'
        '"depends_on":["account"]}]}'
    )
    planner = ExecutionPlanningService(client=client, model_name="test-model")

    plan = planner.plan("规划提示词", REGISTERED_AGENTS)

    assert [node.node_id for node in plan.nodes] == ["account", "transfer"]
    assert client.prompts == ["规划提示词"]


def test_execution_plan_accepts_explicit_unsupported_result():
    plan = ExecutionPlan(status="UNSUPPORTED", reason="No registered agent can handle this request.")

    plan.validate_for_registry(REGISTERED_AGENTS)


def test_execution_plan_rejects_ready_result_without_nodes():
    plan = ExecutionPlan(status="READY", reason="Ready")

    with pytest.raises(ValueError, match="at least one node"):
        plan.validate_for_registry(REGISTERED_AGENTS)


def test_planning_service_repairs_invalid_plan_once():
    client = FakePlanningClient(
        [
            '{"status":"READY","reason":"bad","nodes":[{"node_id":"bad","agent_name":"UnknownAgent","task":"x"}]}',
            '{"status":"READY","reason":"fixed","nodes":[{"node_id":"knowledge","agent_name":"QueryAgent","task":"answer"}]}',
        ]
    )
    planner = ExecutionPlanningService(client=client, model_name="test-model")

    plan = planner.plan("planning prompt", REGISTERED_AGENTS)

    assert plan.status == "READY"
    assert plan.reason == "fixed"
    assert len(client.prompts) == 2
    assert "unregistered agent" in client.prompts[1]


def test_planning_service_raises_after_failed_repair():
    client = FakePlanningClient(["not-json", "still-not-json"])
    planner = ExecutionPlanningService(client=client, model_name="test-model")

    with pytest.raises(ValueError, match="failed after 2 attempts"):
        planner.plan("planning prompt", REGISTERED_AGENTS)


def test_planning_service_rejects_unregistered_llm_node():
    client = FakePlanningClient('{"nodes":[{"node_id":"bad","agent_name":"UnknownAgent","task":"x"}]}')
    planner = ExecutionPlanningService(client=client, model_name="test-model")

    with pytest.raises(ValueError, match="unregistered agent"):
        planner.plan("规划提示词", REGISTERED_AGENTS)
