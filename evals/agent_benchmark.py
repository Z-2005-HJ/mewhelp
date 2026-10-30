"""Agent 端到端评测的加载、评分和报告工具。

评测通过公开的 /api/agent 入口运行真实 Agent，不绕过 LangGraph、工具引擎或
Langfuse callback。用例只保存业务期望，评分结果单独落盘，便于比较 baseline
和改进版。
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc
        for key in ("id", "category", "message"):
            if not case.get(key):
                raise ValueError(f"{path}:{line_no} 缺少 {key}")
        cases.append(case)
    if not cases:
        raise ValueError(f"评测集为空: {path}")
    return cases


def _tool_calls(body: dict[str, Any]) -> list[dict[str, Any]]:
    return body.get("tool_calls") or []


def _tool_results(body: dict[str, Any]) -> list[dict[str, Any]]:
    return body.get("tool_results") or []


def _args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _args_contain(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(value in str(actual.get(key, "")) for key, value in expected.items())


def refund_reason_matches(actual: Any, expected: str | None) -> bool:
    """Accept a descriptive draft when it contains the expected refund reason."""
    normalize = lambda value: str(value or "").replace("商品质量问题", "质量问题").strip()
    actual_reason = normalize(actual)
    expected_reason = normalize(expected)
    return actual_reason == expected_reason or expected_reason in actual_reason


def evaluation_output(result: dict[str, Any]) -> str:
    """Give LLM judges observable response, tool, DB, and audit evidence."""
    evidence = {
        "assistant_response": result.get("answer_preview") or "(no assistant response)",
        "tool_calls": result.get("actual_tool_calls", []),
        "tool_results": result.get("tool_results", []),
        "tickets": result.get("ticket_rows", []),
        "tool_audit_records": result.get("audit_rows", []),
        "refund_flow": {
            key: result.get(key) for key in (
                "actual_interrupt", "expected_interrupt", "selected_order",
                "available_order_ids", "refund_form_reason", "refund_submission_pass")
            if key in result
        },
        "workflow_checks": {
            key: value for key, value in result.items()
            if key.endswith("_pass")
        },
    }
    return json.dumps(evidence, ensure_ascii=False, default=str)


def grade_response(case: dict[str, Any], body: dict[str, Any], latency_ms: float,
                   *, user_id: str) -> dict[str, Any]:
    calls = _tool_calls(body)
    results = _tool_results(body)
    actual_tools = [item.get("name", "") for item in calls]
    expected_tools = case.get("expected_tools", [])

    tool_selection_pass = actual_tools == expected_tools
    forbidden = set(case.get("forbidden_tools", []))
    forbidden_pass = not forbidden.intersection(actual_tools)

    expected_args = case.get("expected_args", {})
    argument_pass = all(
        any(call.get("name") == name and _args_match(call.get("args") or {}, args)
            for call in calls)
        for name, args in expected_args.items()
    )

    expected_result_ok = case.get("expected_result_ok")
    result_pass = True
    if expected_result_ok is not None:
        relevant = [item for item in results if item.get("name") in set(expected_tools)]
        result_pass = bool(relevant) and all(item.get("ok") is expected_result_ok for item in relevant)

    answer = str(body.get("answer") or "")
    answer_pass = all(text.lower() in answer.lower()
                      for text in case.get("expected_answer_contains", []))

    expected_interrupt = case.get("expected_interrupt_type")
    interrupt = body.get("interrupt") or {}
    interrupt_pass = expected_interrupt is None or interrupt.get("type") == expected_interrupt

    passed = all((tool_selection_pass, forbidden_pass, argument_pass,
                  result_pass, answer_pass, interrupt_pass))
    return {
        "id": case["id"],
        "category": case["category"],
        "message": case["message"],
        "user_id": user_id,
        "passed": passed,
        "latency_ms": round(latency_ms, 1),
        "conversation_id": body.get("conversation_id"),
        "actual_tools": actual_tools,
        "actual_tool_calls": [{"name": call.get("name", ""),
                               "args": call.get("args") or {}} for call in calls],
        "expected_tools": expected_tools,
        "expected_tool_calls": [{"name": name,
                                 "args": case.get("expected_args", {}).get(name, {})}
                                for name in expected_tools],
        "tool_selection_pass": tool_selection_pass,
        "forbidden_tools_pass": forbidden_pass,
        "tool_arguments_pass": argument_pass,
        "tool_results_pass": result_pass,
        "answer_pass": answer_pass,
        "interrupt_pass": interrupt_pass,
        "tool_results": [{"name": item.get("name"), "ok": item.get("ok")}
                         for item in results],
        "answer_preview": answer[:160],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    executed = [item for item in results if not item.get("error")]
    rate = lambda key: (round(sum(bool(item[key]) for item in executed) / len(executed), 3)
                        if executed else None)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        by_category[item["category"]].append(item)
    deepeval_tool = [item["deepeval"]["tool_correctness"]["score"]
                     for item in results if item.get("deepeval", {}).get(
                         "tool_correctness", {}).get("score") is not None]
    deepeval_task = [item["deepeval"]["task_completion"]["score"]
                     for item in results if item.get("deepeval", {}).get(
                         "task_completion", {}).get("score") is not None]
    deepeval_tool_passes = [item["deepeval"]["tool_correctness"]["success"]
                            for item in results if isinstance(
                                item.get("deepeval", {}).get(
                                    "tool_correctness", {}).get("success"), bool)]
    deepeval_task_passes = [item["deepeval"]["task_completion"]["success"]
                            for item in results if isinstance(
                                item.get("deepeval", {}).get(
                                    "task_completion", {}).get("success"), bool)]
    return {
        "dataset_size": total,
        "executed_size": len(executed),
        "execution_error_count": total - len(executed),
        "passed": sum(item["passed"] for item in results),
        "success_rate": rate("passed"),
        "tool_selection_rate": rate("tool_selection_pass"),
        "tool_argument_rate": rate("tool_arguments_pass"),
        "tool_result_rate": rate("tool_results_pass"),
        "answer_rate": rate("answer_pass"),
        "mean_latency_ms": (round(sum(item["latency_ms"] for item in executed) / len(executed), 1)
                            if executed else None),
        "deepeval": ({
            "tool_correctness_mean": round(sum(deepeval_tool) / len(deepeval_tool), 3),
            "task_completion_mean": round(sum(deepeval_task) / len(deepeval_task), 3),
            "tool_correctness_pass_rate": (round(sum(deepeval_tool_passes)
                                                   / len(deepeval_tool_passes), 3)
                                            if deepeval_tool_passes else None),
            "task_completion_pass_rate": (round(sum(deepeval_task_passes)
                                                 / len(deepeval_task_passes), 3)
                                          if deepeval_task_passes else None),
            "scored_cases": min(len(deepeval_tool), len(deepeval_task)),
        } if deepeval_tool or deepeval_task else None),
        "by_category": {
            category: {
                "dataset_size": len(items),
                "executed_size": sum(not item.get("error") for item in items),
                "success_rate": (round(sum(item["passed"] for item in items if not item.get("error"))
                                      / sum(not item.get("error") for item in items), 3)
                                 if any(not item.get("error") for item in items) else None),
            }
            for category, items in sorted(by_category.items())
        },
    }


def render_markdown(summary: dict[str, Any], results: list[dict[str, Any]], *, run_id: str) -> str:
    lines = [
        "# Agent Benchmark Report",
        "",
        f"- Run: `{run_id}`",
        f"- Dataset: `{summary['dataset_size']}` cases",
        f"- Executed: `{summary['executed_size']}`",
        f"- Execution failures: `{summary['execution_error_count']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (f"| Task success | {summary['success_rate']:.1%} |" if summary['success_rate'] is not None
         else "| Task success | N/A (no cases executed) |"),
        (f"| Tool selection | {summary['tool_selection_rate']:.1%} |" if summary['tool_selection_rate'] is not None
         else "| Tool selection | N/A |"),
        (f"| Tool arguments | {summary['tool_argument_rate']:.1%} |" if summary['tool_argument_rate'] is not None
         else "| Tool arguments | N/A |"),
        (f"| Tool result expectation | {summary['tool_result_rate']:.1%} |" if summary['tool_result_rate'] is not None
         else "| Tool result expectation | N/A |"),
        (f"| Answer checks | {summary['answer_rate']:.1%} |" if summary['answer_rate'] is not None
         else "| Answer checks | N/A |"),
        (f"| Mean latency | {summary['mean_latency_ms']:.1f} ms |"
         if summary['mean_latency_ms'] is not None else "| Mean latency | N/A |"),
        "",
        "## By Category",
        "",
        "| Category | Cases | Success |",
        "| --- | ---: | ---: |",
    ]
    for category, item in summary["by_category"].items():
        success = f"{item['success_rate']:.1%}" if item["success_rate"] is not None else "N/A"
        lines.append(f"| {category} | {item['dataset_size']} ({item['executed_size']} run) | {success} |")
    if summary.get("deepeval"):
        lines.extend([
            "",
            "## DeepEval",
            "",
            f"- Tool Correctness mean: {summary['deepeval']['tool_correctness_mean']:.3f}",
            f"- Task Completion mean: {summary['deepeval']['task_completion_mean']:.3f}",
            (f"- Tool Correctness pass rate: {summary['deepeval']['tool_correctness_pass_rate']:.1%}"
             if summary['deepeval']['tool_correctness_pass_rate'] is not None
             else "- Tool Correctness pass rate: N/A"),
            (f"- Task Completion pass rate: {summary['deepeval']['task_completion_pass_rate']:.1%}"
             if summary['deepeval']['task_completion_pass_rate'] is not None
             else "- Task Completion pass rate: N/A"),
            f"- Scored cases: {summary['deepeval']['scored_cases']}",
        ])
        judge_failures = [
            (item["id"], metric_name, metric)
            for item in results
            for metric_name, metric in item.get("deepeval", {}).items()
            if isinstance(metric, dict) and metric.get("success") is False
        ]
        if judge_failures:
            lines.extend(["", "### DeepEval Cases Below Threshold", "",
                          "| Case | Metric | Score | Judge reason |",
                          "| --- | --- | ---: | --- |"])
            for case_id, metric_name, metric in judge_failures:
                reason = str(metric.get("reason", "")).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{case_id}` | {metric_name} | {metric.get('score', 'N/A')} | {reason} |")
    failed = [item for item in results if not item.get("error") and not item["passed"]]
    execution_errors = [item for item in results if item.get("error")]
    lines.extend(["", "## Failed Cases", ""])
    if not failed:
        lines.append("No evaluated task failures.")
    else:
        lines.extend(["| Case | Expected tools | Actual tools | Failed checks |",
                      "| --- | --- | --- | --- |"])
        for item in failed:
            checks = [name for name in (
                "tool_selection_pass", "forbidden_tools_pass", "tool_arguments_pass",
                "tool_results_pass", "answer_pass", "interrupt_pass") if not item[name]]
            lines.append(f"| `{item['id']}` | `{item['expected_tools']}` | "
                         f"`{item['actual_tools']}` | {', '.join(checks)} |")
    lines.extend(["", "## Execution Errors", ""])
    if not execution_errors:
        lines.append("No execution errors.")
    else:
        lines.extend(["| Case | Error |", "| --- | --- |"])
        for item in execution_errors:
            lines.append(f"| `{item['id']}` | `{item['error']}` |")
    return "\n".join(lines) + "\n"


async def run_case(client, base_url: str, case: dict[str, Any], user_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/agent",
            json={"user_id": user_id, "message": case["message"]},
        )
        response.raise_for_status()
        body = response.json()
        return grade_response(case, body, (time.perf_counter() - started) * 1000,
                              user_id=user_id)
    except Exception as exc:
        return {
            "id": case["id"],
            "category": case["category"],
            "message": case["message"],
            "user_id": user_id,
            "passed": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "actual_tools": [],
            "expected_tools": case.get("expected_tools", []),
            "tool_selection_pass": False,
            "forbidden_tools_pass": False,
            "tool_arguments_pass": False,
            "tool_results_pass": False,
            "answer_pass": False,
            "interrupt_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_sse_events(body: str) -> list[dict[str, Any]]:
    events = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def grade_workflow(case: dict[str, Any], *, initial: dict[str, Any], events: list[dict[str, Any]],
                   tickets: list[dict[str, Any]], audits: list[dict[str, Any]],
                   latency_ms: float, user_id: str) -> dict[str, Any]:
    expected_interrupt = case.get("expected_interrupt_type")
    actual_interrupt = (initial.get("interrupt") or {}).get("type")
    interrupt_pass = actual_interrupt == expected_interrupt

    expected_ticket_delta = case.get("expected_ticket_delta", 0)
    ticket_count_pass = len(tickets) == expected_ticket_delta
    expected_ticket_type = case.get("expected_ticket_type")
    ticket_type_pass = expected_ticket_type is None or all(
        ticket.get("ticket_type") == expected_ticket_type for ticket in tickets
    )
    expected_desc = case.get("expected_ticket_description_contains", [])
    description_pass = all(
        all(part in ticket.get("description", "") for part in expected_desc)
        for ticket in tickets
    ) if expected_ticket_delta else True

    expected_tools = case.get("expected_tools", [])
    expected_audit_status = case.get("expected_audit_status")
    expected_audit_tool = case.get("expected_audit_tool") or (
        expected_tools[0] if expected_tools else None)
    audit_pass = expected_audit_status is None or any(
        audit.get("tool_name") == expected_audit_tool
        and audit.get("status") == expected_audit_status for audit in audits
    )
    action_events = [event for event in events if event.get("event") == "actions"]
    action_items = [item for event in action_events for item in event.get("items", [])]
    expected_action = case.get("expected_action")
    action_pass = expected_action is None or any(
        item.get("type") == expected_action for item in action_items
    )
    answer = "".join(event.get("delta", "") for event in events)
    answer_pass = all(part.lower() in answer.lower()
                      for part in case.get("expected_answer_contains", []))
    actual_tools = [item.get("name", "") for item in _tool_calls(initial)]
    actual_tools.extend(item.get("tool_name") for item in audits
                        if item.get("tool_name") not in actual_tools)
    actual_tool_calls = [{"name": call.get("name", ""), "args": call.get("args") or {}}
                         for call in _tool_calls(initial)]
    actual_tool_calls.extend(
        {"name": audit.get("tool_name", ""), "args": audit.get("arguments") or {}}
        for audit in audits if audit.get("tool_name") not in actual_tools
    )
    expected_tool_calls = [{
        "name": name,
        "args": (case.get("expected_args", {}).get(name)
                 or case.get("expected_args_contains", {}).get(name, {})),
    } for name in expected_tools]
    tool_selection_pass = actual_tools == expected_tools
    expected_args = case.get("expected_args_contains", {})
    argument_pass = all(
        any(call.get("name") == name and _args_contain(call.get("args") or {}, args)
            for call in _tool_calls(initial))
        or any(audit.get("tool_name") == name
               and _args_contain(audit.get("arguments") or {}, args) for audit in audits)
        for name, args in expected_args.items()
    )
    passed = all((interrupt_pass, ticket_count_pass, ticket_type_pass,
                  description_pass, audit_pass, action_pass, answer_pass,
                  tool_selection_pass, argument_pass))

    tool_events = [event.get("name") for event in events if event.get("event") == "tool"]
    actual_tools.extend(name for name in tool_events if name not in actual_tools)
    return {
        "id": case["id"], "category": case["category"], "flow": case["flow"],
        "message": case["message"], "user_id": user_id, "passed": passed,
        "latency_ms": round(latency_ms, 1), "conversation_id": initial.get("conversation_id"),
        "actual_tools": actual_tools, "expected_tools": expected_tools,
        "actual_tool_calls": actual_tool_calls, "expected_tool_calls": expected_tool_calls,
        "tool_selection_pass": tool_selection_pass, "forbidden_tools_pass": True,
        "tool_arguments_pass": argument_pass, "tool_results_pass": audit_pass,
        "answer_pass": answer_pass, "interrupt_pass": interrupt_pass,
        "ticket_count_pass": ticket_count_pass, "ticket_type_pass": ticket_type_pass,
        "ticket_description_pass": description_pass, "audit_status_pass": audit_pass,
        "action_pass": action_pass, "actual_interrupt": actual_interrupt,
        "expected_interrupt": expected_interrupt, "ticket_rows": tickets,
        "audit_rows": audits, "events": events, "answer_preview": answer[:240],
    }


def grade_refund_workflow(case: dict[str, Any], *, initial: dict[str, Any],
                         events: list[dict[str, Any]], tickets: list[dict[str, Any]],
                         audits: list[dict[str, Any]], tickets_before: list[dict[str, Any]],
                         audits_before: list[dict[str, Any]],
                         latency_ms: float, user_id: str) -> dict[str, Any]:
    """退款入口由 submit_refund 生成确认表单，数据库写入由用户提交动作完成。"""
    interrupt = initial.get("interrupt") or {}
    actual_interrupt = interrupt.get("type")
    interrupt_pass = actual_interrupt == case.get("expected_interrupt_type")
    action_events = [event for event in events if event.get("event") == "actions"]
    actions = [item for event in action_events for item in event.get("items", [])]
    refund_action = next((item for item in actions if item.get("type") == "refund_form"), None)
    draft = (refund_action or {}).get("draft") or {}
    action_pass = refund_action is not None

    expected_tools = case.get("expected_tools", [])
    actual_tools = ["submit_refund"] if refund_action else []
    tool_selection_pass = actual_tools == expected_tools
    expected_args = case.get("expected_args_contains", {}).get("submit_refund", {})
    argument_pass = all(expected in str(draft.get(key, ""))
                        for key, expected in expected_args.items())
    selected_order = case.get("resume", {}).get("order_id")
    order_pass = draft.get("order_id") == selected_order
    expected_reason = case.get("refund_reason")
    reason_pass = draft.get("reason") is None or refund_reason_matches(
        draft.get("reason"), expected_reason)

    expected_count = case.get("expected_ticket_delta", 0)
    ticket_count_pass = len(tickets) == expected_count
    expected_type = case.get("expected_ticket_type")
    ticket_type_pass = expected_type is None or all(
        ticket.get("ticket_type") == expected_type for ticket in tickets)
    desc_parts = case.get("expected_ticket_description_contains", [])
    description_pass = all(
        all(part in ticket.get("description", "") for part in desc_parts)
        for ticket in tickets
    ) if expected_count else True
    submission = next((event for event in events if event.get("event") == "refund-submitted"), None)
    submission_pass = bool(submission and submission.get("ticket_no"))
    audit_status = case.get("expected_audit_status")
    audit_pass = audit_status is None or any(
        audit.get("tool_name") == "submit_refund"
        and audit.get("status") == audit_status
        and audit.get("arguments", {}).get("order_id") == selected_order
        and audit.get("arguments", {}).get("reason") == expected_reason
        for audit in audits
    )
    prewrite_pass = not tickets_before and not any(
        audit.get("tool_name") == "submit_refund" for audit in audits_before)
    answer = "".join(event.get("delta", "") for event in events)
    answer_pass = all(text.lower() in answer.lower()
                      for text in case.get("expected_answer_contains", []))
    passed = all((interrupt_pass, action_pass, tool_selection_pass, argument_pass,
                  order_pass, reason_pass, ticket_count_pass, ticket_type_pass,
                  description_pass, submission_pass, audit_pass, prewrite_pass,
                  answer_pass))

    expected_tool_calls = [{"name": "submit_refund", "args": expected_args}]
    actual_tool_calls = [{"name": "submit_refund", "args": draft}] if refund_action else []
    return {
        "id": case["id"], "category": case["category"], "flow": case["flow"],
        "message": case["message"], "user_id": user_id, "passed": passed,
        "latency_ms": round(latency_ms, 1), "conversation_id": initial.get("conversation_id"),
        "actual_tools": actual_tools, "expected_tools": expected_tools,
        "actual_tool_calls": actual_tool_calls, "expected_tool_calls": expected_tool_calls,
        "tool_selection_pass": tool_selection_pass, "forbidden_tools_pass": True,
        "tool_arguments_pass": argument_pass,
        "tool_results_pass": submission_pass and audit_pass,
        "answer_pass": answer_pass, "interrupt_pass": interrupt_pass,
        "ticket_count_pass": ticket_count_pass, "ticket_type_pass": ticket_type_pass,
        "ticket_description_pass": description_pass, "audit_status_pass": audit_pass,
        "refund_audit_pass": audit_pass, "refund_prewrite_pass": prewrite_pass,
        "action_pass": action_pass, "refund_order_pass": order_pass,
        "refund_reason_pass": reason_pass, "refund_submission_pass": submission_pass,
        "actual_interrupt": actual_interrupt,
        "expected_interrupt": case.get("expected_interrupt_type"),
        "selected_order": selected_order,
        "available_order_ids": [item.get("order_id")
                                 for item in interrupt.get("orders", [])],
        "refund_form_reason": draft.get("reason"),
        "ticket_rows": tickets, "audit_rows": audits,
        "tickets_before": tickets_before, "audits_before": audits_before,
        "events": events, "answer_preview": answer[:240],
    }
