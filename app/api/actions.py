# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.db import repository
from app.graph import runtime
from app.schemas.actions import (
    CreateRefundRequest,
    CreateRefundResponse,
    CreateTicketRequest,
    CreateTicketResponse,
    ResumeRequest,
)
from app.tools import business

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket_action(req: CreateTicketRequest) -> CreateTicketResponse:
    """建工单按钮:用户点了才写 tickets 表(复用 ch02 工单能力)。转人工是另一件事,纯前端。"""
    try:
        ticket_no = await repository.create_ticket(
            req.conversation_id, req.description, req.ticket_type)
    except SQLAlchemyError:
        logger.exception("建工单失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="工单系统暂时不可用,请稍后重试")
    return CreateTicketResponse(ticket_no=ticket_no)


@router.post("/api/actions/create-refund", response_model=CreateRefundResponse)
async def create_refund_action(req: CreateRefundRequest) -> CreateRefundResponse:
    """退款表单提交:从会话身份校验订单归属,再写退款工单和工具审计。"""
    desc = f"退款申请 订单号={req.order_id} 原因={req.reason}"
    args = {"order_id": req.order_id, "reason": req.reason}
    conversation = await repository.get_conversation(req.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    started = time.monotonic()
    if not business.owns_order(conversation.user_id, req.order_id):
        await _audit_refund(
            req.conversation_id, args, "权限拒绝", "退款订单不属于当前会话用户",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        raise HTTPException(status_code=403, detail="该订单不属于当前用户,无法提交退款")

    try:
        ticket_no = await repository.create_ticket(req.conversation_id, desc, "退款")
    except SQLAlchemyError:
        await _audit_refund(
            req.conversation_id, args, "失败", "退款工单写入失败",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        logger.exception("退款单创建失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="退款系统暂时不可用,请稍后重试")
    await _audit_refund(
        req.conversation_id, args, "成功", f"退款工单已创建:{ticket_no}",
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    return CreateRefundResponse(ticket_no=ticket_no)


async def _audit_refund(conversation_id: int, args: dict, status: str,
                        result_summary: str, *, duration_ms: int) -> None:
    try:
        await repository.insert_tool_audit(
            conversation_id=conversation_id, tool_call_id=None,
            tool_name="submit_refund", tool_source="builtin", mcp_server=None,
            arguments=args, result_summary=result_summary, status=status,
            error_message=result_summary if status != "成功" else None,
            retry_count=0, duration_ms=duration_ms,
        )
    except Exception:
        logger.exception("退款工具审计写入失败 conv=%s status=%s", conversation_id, status)


@router.post("/api/actions/resume")
async def resume_action(req: ResumeRequest):
    """中断续跑(SSE,事件与 /api/chat 同构):订单选择器点选回 order_id;
    工单预览卡回 confirmed(ch08,resume 值为 {"confirmed": bool},agent_tools 按此放行/拒绝)。"""
    if req.order_id is None and req.confirmed is None:
        raise HTTPException(status_code=400, detail="order_id 与 confirmed 至少传一个")
    resume_value = req.order_id if req.order_id is not None else {"confirmed": bool(req.confirmed)}

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in runtime.stream_resume(req.conversation_id, resume_value):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "citations":
                    yield _sse({"event": "citations", "items": ev["items"]})
                elif ev["type"] == "actions":
                    yield _sse({"event": "actions", "items": ev["items"]})
                elif ev["type"] == "interrupt":
                    yield _sse({"event": "interrupt",
                                **{k: v for k, v in ev.items() if k != "type"}})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except runtime.ConversationNotFound:
            yield "event: error\n"
            yield _sse({"message": "会话不存在"})
            return
        except Exception:
            logger.exception("续跑失败 conv=%s", req.conversation_id)
            yield "event: error\n"
            yield _sse({"message": "上游暂时不可用,请稍后重试"})
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
