"""DeepEval metrics using the application's configured OpenAI-compatible judge model."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall

from evals.agent_benchmark import evaluation_output
from app.config import settings
from app.core.llm import get_chat_model


class ApplicationJudge(DeepEvalBaseLLM):
    """Reuse the project model endpoint instead of requiring a separate OPENAI_API_KEY."""

    def __init__(self):
        self._chat = get_chat_model(streaming=False, temperature=0)
        super().__init__(model=settings.chat_model)

    def load_model(self):
        return self._chat

    @staticmethod
    def _content(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)

    def generate(self, prompt: str, schema=None, **kwargs) -> str:
        if schema is not None:
            response = self._chat.with_structured_output(
                schema, method="function_calling").invoke(prompt)
            if hasattr(response, "model_dump_json"):
                return response.model_dump_json()
            if hasattr(response, "model_dump"):
                return json.dumps(response.model_dump(), ensure_ascii=False)
            return json.dumps(response, ensure_ascii=False, default=str)
        return self._content(self._chat.invoke(prompt))

    async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
        if schema is not None:
            response = await self._chat.with_structured_output(
                schema, method="function_calling").ainvoke(prompt)
            if hasattr(response, "model_dump_json"):
                return response.model_dump_json()
            if hasattr(response, "model_dump"):
                return json.dumps(response.model_dump(), ensure_ascii=False)
            return json.dumps(response, ensure_ascii=False, default=str)
        return self._content(await self._chat.ainvoke(prompt))

    def get_model_name(self) -> str:
        return settings.chat_model


def score_with_deepeval(result: dict[str, Any], case: dict[str, Any], judge: ApplicationJudge) -> dict:
    expected_tool_calls = result.get("expected_tool_calls") or [
        {"name": name, "args": case.get("expected_args", {}).get(name, {})}
        for name in case.get("expected_tools", [])
    ]
    actual_tool_calls = result.get("actual_tool_calls") or [
        {"name": name, "args": {}} for name in result.get("actual_tools", [])
    ]
    expected_tools = [ToolCall(name=item["name"], input_parameters=item.get("args") or {})
                      for item in expected_tool_calls]
    actual_tools = [ToolCall(name=item["name"], input_parameters=item.get("args") or {})
                    for item in actual_tool_calls]
    test_case = LLMTestCase(
        name=case["id"],
        input=case["message"],
        actual_output=evaluation_output(result),
        expected_output=case.get("evaluation_goal", case["message"]),
        tools_called=actual_tools,
        expected_tools=expected_tools,
        completion_time=(result.get("latency_ms") or 0) / 1000,
    )

    tool_metric = ToolCorrectnessMetric(
        threshold=1.0,
        model=judge,
        async_mode=False,
        should_exact_match=True,
        should_consider_ordering=True,
    )
    tool_score = tool_metric.measure(test_case, _show_indicator=False)

    task_metric = TaskCompletionMetric(
        task=case.get("evaluation_goal", case["message"]),
        model=judge,
        eval_mode="llm",
        async_mode=False,
        threshold=0.7,
    )
    task_score = task_metric.measure(test_case, _show_indicator=False)
    return {
        "tool_correctness": {"score": tool_score, "success": tool_metric.success,
                             "reason": tool_metric.reason},
        "task_completion": {"score": task_score, "success": task_metric.success,
                             "reason": task_metric.reason},
    }
