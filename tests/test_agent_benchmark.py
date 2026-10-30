from evals.agent_benchmark import (
    evaluation_output,
    grade_response,
    grade_refund_workflow,
    grade_workflow,
    parse_sse_events,
    refund_reason_matches,
    render_markdown,
    summarize,
)


def test_grade_response_checks_tool_arguments_results_and_answer():
    case = {
        "id": "order_lookup",
        "category": "business_tool",
        "message": "查订单 1001",
        "expected_tools": ["query_order"],
        "expected_args": {"query_order": {"order_id": "1001"}},
        "expected_result_ok": True,
        "expected_answer_contains": ["已付款"],
    }
    result = grade_response(case, {
        "conversation_id": 1,
        "answer": "订单 1001 已付款。",
        "tool_calls": [{"name": "query_order", "args": {"order_id": "1001"}}],
        "tool_results": [{"name": "query_order", "ok": True}],
    }, 120.0, user_id="eval-case")

    assert result["passed"] is True
    assert result["tool_arguments_pass"] is True
    assert result["tool_results_pass"] is True


def test_grade_workflow_checks_confirmation_and_database_state():
    case = {
        "id": "ticket_confirm",
        "category": "multi_turn_write",
        "flow": "ticket_confirm",
        "message": "登记售后",
        "expected_interrupt_type": "confirm_ticket",
        "expected_tools": ["create_ticket"],
        "expected_args_contains": {
            "create_ticket": {"description": "猫砂盆", "ticket_type": "售后"}},
        "expected_ticket_delta": 1,
        "expected_ticket_type": "售后",
        "expected_ticket_description_contains": ["猫砂盆"],
        "expected_audit_status": "成功",
        "expected_answer_contains": ["工单"],
    }
    result = grade_workflow(
        case,
        initial={
            "conversation_id": 7,
            "interrupt": {"type": "confirm_ticket"},
            "tool_calls": [{"name": "create_ticket", "args": {
                "description": "猫砂盆漏电", "ticket_type": "售后"}}],
        },
        events=[{"delta": "已创建工单"}],
        tickets=[{"ticket_no": "T1", "ticket_type": "售后", "description": "猫砂盆漏电"}],
        audits=[{"tool_name": "create_ticket", "status": "成功",
                 "arguments": {"description": "猫砂盆漏电", "ticket_type": "售后"}}],
        latency_ms=500,
        user_id="eval-case",
    )

    assert result["passed"] is True
    assert result["ticket_count_pass"] is True
    assert result["audit_status_pass"] is True
    assert result["tool_arguments_pass"] is True


def test_grade_workflow_counts_missing_confirmation_as_task_failure():
    case = {
        "id": "ticket_confirm",
        "category": "multi_turn_write",
        "flow": "ticket_confirm",
        "message": "登记售后工单",
        "expected_interrupt_type": "confirm_ticket",
        "expected_tools": ["create_ticket"],
        "expected_ticket_delta": 1,
        "expected_audit_status": "成功",
    }
    result = grade_workflow(
        case, initial={"conversation_id": 9, "answer": "请补充信息"},
        events=[], tickets=[], audits=[], latency_ms=100, user_id="eval-case",
    )

    assert result["passed"] is False
    assert result["interrupt_pass"] is False
    assert result["tool_selection_pass"] is False
    assert result["ticket_count_pass"] is False


def test_parse_sse_events_ignores_invalid_and_done_frames():
    events = parse_sse_events(
        'data: {"delta":"ok"}\n\ndata: [DONE]\n\ndata: {bad}\n')
    assert events == [{"delta": "ok"}]


def test_grade_refund_workflow_checks_selected_order_submission_and_ticket():
    case = {
        "id": "refund",
        "category": "multi_turn_refund",
        "flow": "refund_select_and_submit",
        "message": "商品有质量问题，申请退款",
        "expected_interrupt_type": "select_order",
        "resume": {"order_id": "1001"},
        "expected_action": "refund_form",
        "refund_reason": "质量问题",
        "expected_tools": ["submit_refund"],
        "expected_args_contains": {"submit_refund": {"order_id": "1001"}},
        "expected_ticket_type": "退款",
        "expected_ticket_delta": 1,
        "expected_ticket_description_contains": ["订单号=1001", "原因=质量问题"],
        "expected_audit_status": "成功",
        "expected_answer_contains": ["申请"],
    }
    result = grade_refund_workflow(
        case,
        initial={"conversation_id": 8, "interrupt": {"type": "select_order"}},
        events=[
            {"event": "actions", "items": [{"type": "refund_form", "draft": {
                "order_id": "1001", "reason": "商品质量问题"}}]},
            {"delta": "请核对退款申请"},
            {"event": "refund-submitted", "ticket_no": "T2"},
        ],
        tickets=[{"ticket_no": "T2", "ticket_type": "退款",
                  "description": "退款申请 订单号=1001 原因=质量问题"}],
        audits=[{"tool_name": "submit_refund", "status": "成功",
                 "arguments": {"order_id": "1001", "reason": "质量问题"}}],
        tickets_before=[], audits_before=[],
        latency_ms=700,
        user_id="eval-case",
    )
    assert result["passed"] is True
    assert result["tool_selection_pass"] is True
    assert result["refund_submission_pass"] is True
    assert result["refund_audit_pass"] is True


def test_summary_includes_optional_deepeval_scores():
    results = [{
        "id": "case",
        "category": "business_tool",
        "passed": True,
        "latency_ms": 50,
        "tool_selection_pass": True,
        "tool_arguments_pass": True,
        "tool_results_pass": True,
        "answer_pass": True,
        "deepeval": {
            "tool_correctness": {"score": 1.0, "success": True},
            "task_completion": {"score": 0.6, "success": False,
                                 "reason": "Confirm step was skipped."},
        },
    }]
    result = summarize(results)
    assert result["deepeval"] == {
        "tool_correctness_mean": 1.0,
        "task_completion_mean": 0.6,
        "tool_correctness_pass_rate": 1.0,
        "task_completion_pass_rate": 0.0,
        "scored_cases": 1,
    }
    report = render_markdown(result, results, run_id="test")
    assert "Task Completion pass rate: 0.0%" in report
    assert "DeepEval Cases Below Threshold" in report
    assert "Confirm step was skipped." in report


def test_evaluation_output_includes_database_and_audit_evidence():
    output = evaluation_output({
        "answer_preview": "已取消建单",
        "actual_tool_calls": [{"name": "create_ticket", "args": {}}],
        "ticket_rows": [],
        "audit_rows": [{"tool_name": "create_ticket", "status": "权限拒绝"}],
        "audit_status_pass": True,
    })

    assert "已取消建单" in output
    assert '"tickets": []' in output
    assert "权限拒绝" in output


def test_refund_reason_normalizes_descriptive_model_output():
    assert refund_reason_matches("商品质量问题", "质量问题") is True
    assert refund_reason_matches("签收后发现商品存在质量问题", "质量问题") is True
    assert refund_reason_matches("七天无理由", "质量问题") is False


def test_execution_errors_are_not_reported_as_agent_quality_failures():
    summary = summarize([{
        "id": "case",
        "category": "business_tool",
        "passed": False,
        "error": "ConnectError",
        "latency_ms": 10,
    }])
    assert summary["execution_error_count"] == 1
    assert summary["executed_size"] == 0
    assert summary["success_rate"] is None
