"""统一 Agent 基准评测。

前置: make dev 已启动,再运行 `make eval-agent-benchmark`。
结果写入 data/agent_eval/，同一批用例可重复比较 baseline 和改进版。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
from datetime import datetime, timezone

import httpx

from evals.agent_benchmark import (
    grade_workflow,
    grade_refund_workflow,
    load_cases,
    parse_sse_events,
    render_markdown,
    refund_reason_matches,
    run_case,
    summarize,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "evals/agent_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "data/agent_eval"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=pathlib.Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--user-id-prefix", default="agent-eval")
    parser.add_argument("--deepeval", action="store_true",
                        help="Use DeepEval Task Completion and Tool Correctness LLM metrics")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        results = []
        for case in cases:
            user_id = f"{args.user_id_prefix}-{run_id}-{case['id']}"
            if case.get("flow"):
                result = await run_workflow_case(client, args.base_url, case, user_id)
            else:
                result = await run_case(client, args.base_url, case, user_id)
            results.append(result)

    if args.deepeval:
        from evals.deepeval_metrics import ApplicationJudge, score_with_deepeval

        scorable = [(result, case) for result, case in zip(results, cases)
                    if not result.get("error")]
        judge = ApplicationJudge() if scorable else None
        for result, case in zip(results, cases):
            if result.get("error"):
                result["deepeval"] = {"skipped": "Agent execution failed"}
                continue
            try:
                result["deepeval"] = score_with_deepeval(result, case, judge)
            except Exception as exc:
                result["deepeval"] = {"error": f"{type(exc).__name__}: {exc}"}

    summary = summarize(results)
    payload = {"run_id": run_id, "summary": summary, "results": results}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / f"{run_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "latest.md").write_text(
        render_markdown(summary, results, run_id=run_id), encoding="utf-8")

    success = (f"{summary['success_rate']:.1%}" if summary["success_rate"] is not None
               else "N/A, no cases executed")
    print(f"Agent benchmark: {summary['passed']}/{summary['dataset_size']} passed; "
          f"{summary['executed_size']} executed; success={success}; "
          f"execution_errors={summary['execution_error_count']}")
    print(f"Report: {args.output / 'latest.md'}")


async def _sse_post(client: httpx.AsyncClient, url: str, payload: dict) -> list[dict]:
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return parse_sse_events(response.text)


async def run_workflow_case(client: httpx.AsyncClient, base_url: str,
                            case: dict, user_id: str) -> dict:
    from app.db.base import async_session
    from app.db.models import Ticket, ToolAuditLog
    from sqlalchemy import select

    started = asyncio.get_running_loop().time()
    initial = {}
    events = []
    tickets_before: list[dict] = []
    audits_before: list[dict] = []
    try:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/agent",
            json={"user_id": user_id, "message": case["message"]},
        )
        response.raise_for_status()
        initial = response.json()
        conversation_id = initial["conversation_id"]
        interrupt_type = (initial.get("interrupt") or {}).get("type")

        async with async_session() as session:
            tickets_before = [
                {"ticket_no": row.ticket_no, "ticket_type": row.ticket_type,
                 "description": row.description}
                for row in (await session.execute(
                    select(Ticket).where(Ticket.conversation_id == conversation_id))).scalars()
            ]
            audits_before = [
                {"tool_name": row.tool_name, "status": row.status,
                 "arguments": row.arguments or {}}
                for row in (await session.execute(
                    select(ToolAuditLog).where(ToolAuditLog.conversation_id == conversation_id))).scalars()
            ]

        flow = case["flow"]
        if flow == "refund_select_and_submit":
            selected = case["resume"]["order_id"]
            initial_interrupt = initial.get("interrupt") or {}
            orders = initial_interrupt.get("orders", [])
            if (initial_interrupt.get("type") == "select_order"
                    and any(item.get("order_id") == selected for item in orders)):
                events = await _sse_post(
                    client, f"{base_url.rstrip('/')}/api/actions/resume",
                    {"conversation_id": conversation_id, "order_id": selected})
                interrupt_event = next(
                    (event for event in events if event.get("event") == "interrupt"), None)
                action_event = next(
                    (event for event in events if event.get("event") == "actions"), None)
                refund_action = next((item for item in (action_event or {}).get("items", [])
                                      if item.get("type") == "refund_form"), None)
                draft = (refund_action or {}).get("draft") or {}
                expected_refund_args = case.get("expected_args_contains", {}).get(
                    "submit_refund", {})
                draft_matches = (
                    not interrupt_event
                    and refund_action is not None
                    and draft.get("order_id") == selected
                    and (draft.get("reason") is None or refund_reason_matches(
                        draft.get("reason"), case["refund_reason"]))
                    and "submit_refund" in case.get("expected_tools", [])
                    and all(expected in str(draft.get(key, ""))
                            for key, expected in expected_refund_args.items())
                )
                if draft_matches:
                    refund_response = await client.post(
                        f"{base_url.rstrip('/')}/api/actions/create-refund",
                        json={"conversation_id": conversation_id,
                              "order_id": selected, "reason": case["refund_reason"]},
                    )
                    refund_response.raise_for_status()
                    events.append({"event": "refund-submitted", **refund_response.json()})

        if flow in {"ticket_confirm", "ticket_cancel"}:
            expected_interrupt = case.get("expected_interrupt_type")
            if interrupt_type == expected_interrupt and not tickets_before:
                pre_tool_calls = [call.get("name", "")
                                  for call in initial.get("tool_calls", [])]
                expected_tool_calls = case.get("expected_tools", [])
                expected_args = case.get("expected_args_contains", {})
                args_match = all(
                    any(call.get("name") == tool_name and all(
                        expected in str((call.get("args") or {}).get(key, ""))
                        for key, expected in values.items())
                        for call in initial.get("tool_calls", []))
                    for tool_name, values in expected_args.items()
                )
                if pre_tool_calls == expected_tool_calls and args_match:
                    events = await _sse_post(
                        client, f"{base_url.rstrip('/')}/api/actions/resume",
                        {"conversation_id": conversation_id,
                         "confirmed": bool(case["resume"]["confirmed"])})
        else:
            if flow != "refund_select_and_submit":
                raise ValueError(f"不支持的 workflow: {flow}")

        async with async_session() as session:
            ticket_rows = [
                {"ticket_no": row.ticket_no, "ticket_type": row.ticket_type,
                 "description": row.description}
                for row in (await session.execute(
                    select(Ticket).where(Ticket.conversation_id == conversation_id))).scalars()
            ]
            audit_rows = [
                {"tool_name": row.tool_name, "status": row.status,
                 "arguments": row.arguments or {}}
                for row in (await session.execute(
                    select(ToolAuditLog).where(ToolAuditLog.conversation_id == conversation_id))).scalars()
            ]
        created_tickets = ticket_rows[len(tickets_before):]
        new_audits = audit_rows[len(audits_before):]
        if flow == "refund_select_and_submit":
            return grade_refund_workflow(
                case, initial=initial, events=events, tickets=created_tickets,
                audits=new_audits, tickets_before=tickets_before,
                audits_before=audits_before,
                latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
                user_id=user_id)
        return grade_workflow(
            case, initial=initial, events=events, tickets=created_tickets,
            audits=new_audits, latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
            user_id=user_id)
    except Exception as exc:
        calls = initial.get("tool_calls", [])
        return {
            "id": case["id"], "category": case["category"], "flow": case["flow"],
            "message": case["message"], "user_id": user_id, "passed": False,
            "latency_ms": round((asyncio.get_running_loop().time() - started) * 1000, 1),
            "conversation_id": initial.get("conversation_id"),
            "actual_tools": [call.get("name", "") for call in calls],
            "expected_tools": case.get("expected_tools", []),
            "actual_tool_calls": [{"name": call.get("name", ""),
                                   "args": call.get("args") or {}} for call in calls],
            "expected_interrupt": case.get("expected_interrupt_type"),
            "actual_interrupt": (initial.get("interrupt") or {}).get("type"),
            "tool_selection_pass": False, "forbidden_tools_pass": False,
            "tool_arguments_pass": False, "tool_results_pass": False,
            "answer_pass": False, "interrupt_pass": False,
            "ticket_count_pass": False, "ticket_type_pass": False,
            "ticket_description_pass": False, "audit_status_pass": False,
            "action_pass": False, "error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    asyncio.run(main())
