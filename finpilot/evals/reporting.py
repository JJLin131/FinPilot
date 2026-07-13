from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from finpilot.evals.models import EvalRunResult, EvalStatus, EvalSuiteResult


@dataclass(frozen=True)
class MetricDefinition:
    label: str
    description: str
    kind: str = "rate"
    source: str | None = None
    transform: Callable[[float], float] = float
    inverse_threshold: bool = False


def _error_rate(value: float) -> float:
    return 1.0 - value


SUITE_TITLES = {
    "rag_retrieval": "RAG 检索",
    "rag_generation": "RAG 生成",
    "query_rewrite_reranker": "查询改写与重排",
    "planning_orchestration": "任务规划与编排",
    "tool_calling": "工具调用",
    "end_to_end_task": "端到端任务",
    "multi_turn_memory": "多轮记忆",
    "safety_redteam": "安全与红队",
    "resilience_degradation": "韧性与降级",
    "performance_cost": "性能与成本",
    "observability_audit": "可观测性与审计",
}


METRIC_CATALOG: dict[str, dict[str, MetricDefinition]] = {
    "rag_retrieval": {
        **{
            f"recall_at_{k}": MetricDefinition(f"Recall@{k}", f"前 {k} 个结果覆盖相关文档的比例")
            for k in (1, 3, 5)
        },
        **{
            f"precision_at_{k}": MetricDefinition(f"Precision@{k}", f"前 {k} 个结果中相关文档的比例")
            for k in (1, 3, 5)
        },
        **{
            f"hit_at_{k}": MetricDefinition(f"Hit@{k}", f"前 {k} 个结果至少命中一个相关文档的样本比例")
            for k in (1, 3, 5)
        },
        **{
            f"ndcg_at_{k}": MetricDefinition(f"NDCG@{k}", f"前 {k} 个结果的排序质量")
            for k in (1, 3, 5)
        },
        "mrr": MetricDefinition("MRR", "首个相关文档倒数排名的样本均值"),
        "forbidden_hit_count": MetricDefinition("禁用文档命中数", "每个样本命中的禁用文档平均数量", "count"),
    },
    "rag_generation": {
        "faithfulness": MetricDefinition("答案忠实度", "答案陈述受检索上下文支持的程度"),
        "answer_correctness": MetricDefinition("答案正确率", "生成答案与参考答案的一致程度"),
        "answer_relevance": MetricDefinition("答案相关性", "生成答案对用户问题的相关程度"),
        "context_precision": MetricDefinition("上下文准确率", "检索上下文中相关内容所占比例"),
        "context_recall": MetricDefinition("上下文召回率", "参考答案所需信息被上下文覆盖的程度"),
        "citation_recall": MetricDefinition("引用召回率", "期望引用被实际证据覆盖的比例"),
        "required_claim_recall": MetricDefinition("必要结论覆盖率", "必要结论在答案中出现的比例"),
        "forbidden_claim_count": MetricDefinition("禁止结论命中数", "每个样本出现的禁止结论平均数量", "count"),
    },
    "query_rewrite_reranker": {
        "rewrite_success": MetricDefinition("改写成功率", "成功生成非空改写查询的样本比例"),
        "rewrite_recall_lift": MetricDefinition("改写召回提升", "改写重排后相对原始检索的召回率变化", "signed_rate"),
        "rerank_recall": MetricDefinition("重排召回率", "重排结果覆盖相关文档的比例"),
        "false_recall_count": MetricDefinition("错误召回数", "每个样本召回禁用文档的平均数量", "count"),
        "subtopic_recall": MetricDefinition("子主题召回率", "预期子主题被重排结果覆盖的比例"),
    },
    "planning_orchestration": {
        "plan_executable": MetricDefinition("计划可行率", "通过执行计划结构与依赖校验的样本比例"),
        "agent_set_accuracy": MetricDefinition("Agent 选择准确率", "所选 Agent 集合完全正确的样本比例"),
        "node_f1": MetricDefinition("计划节点 F1", "必要计划节点的精确率与召回率调和均值"),
        "dependency_f1": MetricDefinition("依赖关系 F1", "计划节点依赖边的精确率与召回率调和均值"),
        "forbidden_node_count": MetricDefinition("禁用节点命中数", "每个样本包含的禁用节点平均数量", "count"),
        "step_efficiency": MetricDefinition("步骤效率", "必要步骤数与实际计划步骤数的比值"),
    },
    "tool_calling": {
        "tool_sequence_accuracy": MetricDefinition("工具选择准确率", "工具名称及调用顺序完全正确的样本比例"),
        "tool_misjudgment_rate": MetricDefinition(
            "工具误判率",
            "工具名称或调用顺序错误的样本比例",
            source="tool_sequence_accuracy",
            transform=_error_rate,
            inverse_threshold=True,
        ),
        "argument_exact_match": MetricDefinition("参数准确率", "全部工具参数精确匹配的样本比例"),
        "parameter_error_rate": MetricDefinition(
            "参数错误率",
            "至少一个工具参数不精确匹配的样本比例",
            source="argument_exact_match",
            transform=_error_rate,
            inverse_threshold=True,
        ),
        "forbidden_tool_count": MetricDefinition("禁用工具调用数", "每个样本调用禁用工具的平均数量", "count"),
        "final_state_match": MetricDefinition("最终状态准确率", "工具执行后状态符合预期的样本比例"),
    },
    "end_to_end_task": {
        "task_success": MetricDefinition("任务成功率", "端到端完成目标且状态正确的样本比例"),
        "final_state_accuracy": MetricDefinition("最终状态准确率", "最终业务状态符合预期的样本比例"),
        "constraint_satisfaction": MetricDefinition("约束满足率", "答案满足全部必要约束的样本比例"),
        "forbidden_state_avoidance": MetricDefinition("禁止状态规避率", "未进入禁止状态的样本比例"),
    },
    "multi_turn_memory": {
        "memory_match": MetricDefinition("记忆准确率", "期望记忆完全匹配的样本比例"),
        "forget_success": MetricDefinition("遗忘成功率", "应删除记忆已被删除的样本比例"),
        "forbidden_memory_count": MetricDefinition("禁止记忆残留数", "每个样本残留的禁止记忆平均数量", "count"),
        "cross_user_leak_count": MetricDefinition("跨用户泄漏数", "每个样本检测到的跨用户记忆泄漏平均数量", "count"),
    },
    "safety_redteam": {
        "safety_action_match": MetricDefinition("安全动作准确率", "允许、阻断、脱敏或审批动作正确的样本比例"),
        "safety_code_match": MetricDefinition("安全原因码准确率", "安全原因码符合预期的样本比例"),
        "side_effect_match": MetricDefinition("副作用控制率", "实际副作用次数符合预期的样本比例"),
        "false_block": MetricDefinition("合法请求误拦截率", "合法请求被错误阻断的样本比例"),
    },
    "resilience_degradation": {
        "status_match": MetricDefinition("降级状态准确率", "故障场景返回预期状态的样本比例"),
        "attempt_budget_respected": MetricDefinition("重试预算遵守率", "调用次数未超过预算的样本比例"),
        "side_effect_match": MetricDefinition("副作用控制率", "故障场景副作用次数符合预期的样本比例"),
    },
    "performance_cost": {
        "p95_latency_ms": MetricDefinition("P95 延迟", "单个性能样本内多次执行的 P95 延迟均值", "milliseconds"),
        "latency_within_limit": MetricDefinition("延迟达标率", "P95 延迟不超过 case 上限的样本比例"),
        "total_tokens": MetricDefinition("Token 用量", "每个性能样本的平均总 Token 数", "tokens"),
        "tokens_within_limit": MetricDefinition("Token 达标率", "Token 用量不超过 case 上限的样本比例"),
        "total_cost": MetricDefinition("调用成本", "每个性能样本的平均模型调用成本", "cost"),
        "cost_within_limit": MetricDefinition("成本达标率", "调用成本不超过 case 上限的样本比例"),
        "usage_available": MetricDefinition("Usage 可用率", "模型返回有效 Token usage 的样本比例"),
    },
    "observability_audit": {
        "trace_coverage": MetricDefinition("Trace 覆盖率", "必需 span 被采集的比例"),
        "score_coverage": MetricDefinition("评分覆盖率", "必需评测 score 被记录的比例"),
        "audit_coverage": MetricDefinition("审计事件覆盖率", "必需审计事件被记录的比例"),
        "sensitive_field_leak_count": MetricDefinition("敏感字段泄漏数", "每个样本暴露的敏感字段平均数量", "count"),
    },
}


def write_capability_report(
    result: EvalRunResult | EvalSuiteResult,
    *,
    output_dir: Path | str = Path("evals/reports"),
    gate_config: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    suites = result.suites if isinstance(result, EvalRunResult) else [result]
    scope = "all" if isinstance(result, EvalRunResult) else result.suite
    filename = f"{now:%Y%m%d-%H%M%S-%f}-{result.mode}-{scope}.md"
    report_path = output_path / filename
    report_path.write_text(_render_report(result, suites, now, gate_config or {}), encoding="utf-8")
    return report_path


def _render_report(
    result: EvalRunResult | EvalSuiteResult,
    suites: list[EvalSuiteResult],
    generated_at: datetime,
    gate_config: dict[str, Any],
) -> str:
    lines = [
        "# FinPilot 能力测评报告",
        "",
        f"- 生成时间：{generated_at.astimezone():%Y-%m-%d %H:%M:%S %Z}",
        f"- 测评模式：`{result.mode}`",
        f"- Schema：`{result.schema_version}`",
        f"- 整体状态：**{_status_text(result.status)}**",
    ]
    if isinstance(result, EvalRunResult):
        gate_status = str(result.release_gate.get("status") or "UNKNOWN")
        lines.extend(["", f"发布门禁：**{'通过' if gate_status == 'PASSED' else '未通过'}**"])
        reasons = result.release_gate.get("reasons") or []
        if reasons:
            lines.extend(["", "门禁未通过原因："])
            lines.extend(f"- {_escape(str(reason))}" for reason in reasons)
    thresholds = gate_config.get("metric_thresholds") or {}
    for suite in suites:
        lines.extend(_render_suite(suite, thresholds.get(suite.suite) or {}))
    lines.extend(
        [
            "",
            "## 报告口径",
            "",
            "百分比指标为所有有效样本对应 case 指标的算术平均；错误率由对应准确率逐样本取补数后汇总。",
            "环境或评测器不可用时指标标记为“不可用”，不按 0 分参与统计。Langfuse 用于 trace 下钻和趋势分析，本文件用于版本验收与归档。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_suite(suite: EvalSuiteResult, thresholds: dict[str, Any]) -> list[str]:
    title = SUITE_TITLES.get(suite.suite, suite.suite)
    pass_rate = suite.passed_cases / suite.total_cases if suite.total_cases else None
    lines = [
        "",
        f"## {title}",
        "",
        f"执行状态：**{_status_text(suite.status)}**  ",
        f"样本通过率：**{_format_value(pass_rate, 'rate') if pass_rate is not None else '不可用'}**（{suite.passed_cases}/{suite.total_cases}）",
    ]
    if suite.environment.get("available") is False:
        lines.extend(["", "环境状态：**不可用**"])
        messages = {
            str(item.get("message") or item.get("code") or "依赖不可用")
            for item in suite.environment.get("blocking_failures") or []
        }
        lines.extend(f"- {_escape(message)}" for message in sorted(messages))
    lines.extend(
        [
            "",
            "| 指标 | 当前值 | 有效样本数 | 目标 | 结论 | 计算口径 |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    definitions = dict(METRIC_CATALOG.get(suite.suite) or {})
    for metric_name in suite.metrics:
        definitions.setdefault(metric_name, MetricDefinition(metric_name, "有效样本指标的算术平均", "number"))
    for metric_name, definition in definitions.items():
        source = definition.source or metric_name
        values = _metric_values(suite, source, definition.transform)
        value = definition.transform(float(suite.metrics[source])) if source in suite.metrics else None
        limit = _metric_limit(thresholds.get(source), definition.inverse_threshold)
        lines.append(
            "| {label} | {value} | {count} | {target} | {verdict} | {description} |".format(
                label=_escape(definition.label),
                value=_format_value(value, definition.kind) if value is not None else "不可用",
                count=len(values),
                target=_format_limit(limit, definition.kind),
                verdict=_metric_verdict(value, limit),
                description=_escape(definition.description),
            )
        )
    return lines


def _metric_values(suite: EvalSuiteResult, source: str, transform: Callable[[float], float]) -> list[float]:
    values = []
    for item in suite.results:
        value = item.metrics.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(transform(float(value)))
    return values


def _metric_limit(limit: Any, inverse: bool) -> dict[str, float]:
    if not isinstance(limit, dict):
        return {}
    if not inverse:
        return {key: float(value) for key, value in limit.items() if key in {"min", "max"}}
    converted: dict[str, float] = {}
    if "min" in limit:
        converted["max"] = 1.0 - float(limit["min"])
    if "max" in limit:
        converted["min"] = 1.0 - float(limit["max"])
    return converted


def _format_limit(limit: dict[str, float], kind: str) -> str:
    if "min" in limit:
        return f"≥ {_format_value(limit['min'], kind)}"
    if "max" in limit:
        return f"≤ {_format_value(limit['max'], kind)}"
    return "观察项"


def _metric_verdict(value: float | None, limit: dict[str, float]) -> str:
    if value is None:
        return "不可用"
    if not limit:
        return "观察项"
    passed = ("min" not in limit or value >= limit["min"]) and ("max" not in limit or value <= limit["max"])
    return "达标" if passed else "未达标"


def _format_value(value: float | None, kind: str) -> str:
    if value is None:
        return "不可用"
    if kind in {"rate", "signed_rate"}:
        prefix = "+" if kind == "signed_rate" and value > 0 else ""
        return f"{prefix}{value:.2%}"
    if kind == "milliseconds":
        return f"{value:.2f} ms"
    if kind == "tokens":
        return f"{value:,.0f}"
    if kind == "cost":
        return f"{value:.6f}"
    return f"{value:.4f}"


def _status_text(status: EvalStatus) -> str:
    return {
        EvalStatus.PASSED: "通过",
        EvalStatus.FAILED: "未通过",
        EvalStatus.BLOCKED: "阻断",
        EvalStatus.ENV_UNAVAILABLE: "环境不可用",
        EvalStatus.FIXTURE_UNAVAILABLE: "测试替身不可用",
        EvalStatus.EVALUATOR_ERROR: "评测器错误",
    }[status]


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
