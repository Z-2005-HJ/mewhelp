# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import re
from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.core import llm
from app.core.llm import get_chat_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他")
_ORDER_NUMBER = re.compile(r"(?<!\d)\d{4,}(?!\d)")


def infer_explicit_order_intent(query: str) -> str | None:
    """Keep an explicit order lookup on the business route if LLM classification fails."""
    if not _ORDER_NUMBER.search(query or ""):
        return None
    if any(term in query for term in ("物流", "快递", "轨迹", "到哪", "进度")):
        return "物流"
    if any(term in query for term in ("退款", "退货")):
        return "退款退货"
    if any(term in query for term in ("售后", "维修", "保修")):
        return "售后"
    if any(term in query for term in ("订单", "订单号", "单号", "状态", "查", "查询")):
        return "订单"
    return None


class _Intent(BaseModel):
    intent: Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他"] = Field(
        description="九类意图之一")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="判断把握 0-1")


async def classify(query: str, history: str = "") -> dict:
    """九类意图 + confidence(ch08 增「人工」:明确要求建工单/转人工)。扁平字段避开 glm 嵌套 502;
    解析失败/越界 → 归「其他」(最保守兜底,替换 ch05 的回退闲聊)。
    模型默认走 settings.intent_model(空则回落 chat_model)——先求准。"""
    model = llm.structured(_Intent, model=settings.intent_model or None, temperature=0)
    try:
        r: _Intent = await (INTENT_CLASSIFY_PROMPT | model).ainvoke(
            {"query": query, "history": history or "(无)"})
    except Exception:
        inferred = infer_explicit_order_intent(query)
        return {"intent": inferred or "其他", "confidence": 0.8 if inferred else 0.0}
    intent = r.intent if r.intent in INTENTS else "其他"
    if intent == "其他":
        intent = infer_explicit_order_intent(query) or intent
    return {"intent": intent, "confidence": float(r.confidence)}
