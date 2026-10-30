# MewHelp — 电商智能客服 Agent

《AI Agent 智能客服实战》的配套源码。一个能查订单物流、答政策 FAQ、走退款子流程、
挖知识补库、还能微调一个主题分类器的完整客服系统。

代码是随课程一章章长出来的，不分支：ch01 一个纯对话接口起步，到 ch10 收尾时是下面这套东西。

## 技术栈

FastAPI + LangGraph / LangChain + SQLAlchemy / MySQL + Milvus。

聊天、嵌入、重排三组上游各自直连，没有网关那一层。模型名和地址都在 `.env` 里配
(`CHAT_*` / `EMBED_*` / `RERANK_*` 三组)，换供应商、换模型不用改代码。

## 跑起来

```bash
cp .env.example .env      # 填 CHAT_* / EMBED_* / RERANK_* 三组
docker compose up -d      # MySQL
make seed                 # 灌业务测试数据
make dev                  # 依赖容器 + MCP :8101/:8102 + 应用 :8000
```

浏览器打开 <http://localhost:8000> 就是聊天页。

**详细的安装、配置、建知识库、常见问题，看飞书那篇「MewHelp 项目源码下载」**，
这里只留一条能把服务拉起来的最短路径。两边写岔了以那篇为准。

## 代码怎么组织

| 位置 | 装的是什么 |
| - | - |
| `app/api/` | HTTP 入口。聊天、Agent、知识库录入、复核、验收页、成本看板 |
| `app/graph/` | LangGraph 那张图。`state` 状态、`nodes` 节点、`routing` 分流规则、`build` 组装 |
| `app/core/` | 单点能力。上游客户端、检索、重排、意图、指代、摘要、置信度、飞轮、可观测 |
| `app/kb/` | 知识怎么进库。切块、嵌入、双写 MySQL 与 Milvus、去重、从对话里挖问答对 |
| `app/tools/` | 工具系统。内置 `@tool`、MCP 客户端、注册表、统一执行引擎 |
| `app/db/` | 表模型与仓储 |
| `app/static/` | 前端页面。聊天、知识库录入、飞轮待审、观测与成本、主题分布、分类器验收 |
| `mcp_servers/` | 两台业务 MCP Server，物流和售后各一台，独立进程 |
| `sql/` | 各章的建表与迁移，容器首启按文件名顺序自动执行 |
| `scripts/` | 建库、评估、微调这些离线活 |
| `docs/superpowers/` | 各章的 spec 和 plan。课程实战篇教的就是这套流程，留着当范本 |

## 各章长出了什么，怎么验

`make test` 跑全部单测，不打真实模型。下面这些要真服务在跑。

| 章 | 这一章长出来的东西 | 验收 |
| - | - | - |
| ch01 | 流式对话、结构化提取 | `make eval` |
| ch02 | 五个 `@tool` 业务工具，单轮 Function Calling | `make eval-agent` |
| ch03 | 切块、嵌入、MySQL 与 Milvus 双写、对话挖知识 | `make kb-build` `make kb-vectorize` `make eval-retrieval` |
| ch04 | 混合检索、RRF、重排、Query 改写、四策略评估 | `make smoke-rag` `make eval-rag` |
| ch05 | LangGraph workflow 骨架 + 主力 Agent 的 ReAct 环 | `make eval-ch05` |
| ch06 | 分流器、指代消解、退款子流程的 interrupt/resume | `make smoke-interrupt` `make eval-ch06` |
| ch07 | 上下文管理。滑窗、摘要、前缀缓存 | `make eval-ch07` |
| ch08 | 工具系统。MCP 动态发现、统一执行引擎、审计日志 | `make eval-ch08` |
| ch09 | Langfuse 自部署、数据飞轮、成本账 | `make langfuse-up` `make flywheel` `make eval-flywheel` `make cost-report` |
| ch10 | 主题分类器。语料、微调、阈值扫描、ONNX 推理服务 | `make ch10-corpus` `make ch10-train` `make ch10-eval` |

## Agent 评测基准

项目已经内置一套统一的端到端 Agent benchmark。用例在 `evals/agent_cases.jsonl`，
通过真实的 `/api/agent` 入口运行，不绕过 LangGraph、工具执行引擎或 Langfuse 回调。
每条用例可以检查工具选择顺序、参数、工具结果状态、答案要点、拒绝工具和耗时。

```bash
make dev
make eval-agent-benchmark
```

需要额外的大模型裁判分数时运行 `make eval-agent-benchmark-deepeval`，该命令会安装
可选 `eval` 依赖组，并追加 DeepEval Task Completion 和 Tool Correctness 指标。

结果写入 `data/agent_eval/latest.json` 和 `data/agent_eval/latest.md`。修改 Agent
提示词、路由或工具策略后，重复运行同一批用例即可比较 baseline 与新版本；评测结果中的
`user_id` 是本轮 trace 的 `eval_user_id` metadata，可用它回看 Langfuse 中对应的执行链路。
评测集也覆盖退款选单与提交、工单确认与取消，并检查确认前后数据库记录和工具审计结果。

这套 benchmark 是电商客服项目的评测入口，`scripts/eval_agent.py`、`eval_ch04.py`
和 `eval_ch08.py` 继续保留，分别用于工具选择、RAG 和工单确认流的章节级验收。

每条命令的前置条件（哪些服务得先起、哪张表得先建）写在飞书那篇文档的「各章验收」段里。
`make help` 也能看到带说明的完整目标清单。

## 端口

| 端口 | 是什么 |
| - | - |
| 8000 | 应用 |
| 8101 / 8102 | 业务 MCP Server，物流 / 售后 |
| 8110 | ch10 主题分类器推理服务（`make classifier-up` 之后） |
| 3000 | Langfuse（`make langfuse-up` 之后） |
| 19530 | Milvus |

应用那几个页面：`/` 聊天、`/kb` 知识库录入、`/review` 飞轮待审、`/observability` 观测与成本、
`/topics` 主题分布、`/acceptance` 分类器验收。
