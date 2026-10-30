# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""mock 数据源(纯函数,随机种子固定 → 同键稳定)。
ch08 起工具实现迁至 app/tools/builtin/(注册即定义),本模块只留被 graph 节点
(fetch_order)与 builtin 工具共用的快照函数,不再定义任何 @tool。"""
import random


def order_snapshot(order_id: str) -> dict:
    """订单快照(纯函数,随机种子固定 → 同 order_id 稳定)。query_order 工具与 fetch_order 节点同源。"""
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": rng.choice(["待付款", "已付款", "已发货", "已签收"]),
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]),
        "tracking_no": f"SF{rng.randint(10**11, 10**12 - 1)}",
    }


def owns_order(user_id: str, order_id: str) -> bool:
    """这一单是不是这个人的。归属判断只有这一处,工具和节点都调它,不各判各的。

    空 user_id 一律不放行:身份是注入进来的,注入没接上就是空串,那种情况下放行等于没做校验。"""
    if not user_id or not order_id:
        return False
    return any(o["order_id"] == order_id for o in list_user_orders(user_id))


# 课程演示单。1001 用于物流查询,2002 用于待付款场景,1003 用于已签收售后场景。
# 每个账号都能直接跑这些例子;想看归属校验拦人,报别的账号的私有单号即可。
DEMO_ORDER_IDS = ("1001", "2002", "1003")


def list_user_orders(user_id: str) -> list[dict]:
    """按 user_id 稳定列出该用户的订单(mock,不落库):三笔演示单 + 2-4 笔随机单。
    每笔用 order_snapshot 同源,前端选中后回填 order_id 即可 query_order。"""
    rng = random.Random(f"user_orders:{user_id}")
    ids = list(DEMO_ORDER_IDS)
    for _ in range(rng.randint(2, 4)):
        oid = str(rng.randint(1000, 9999))
        if oid not in ids:
            ids.append(oid)
    out = []
    for oid in ids:
        s = order_snapshot(oid)
        out.append({"order_id": oid, "product": s["product"],
                    "status": s["status"], "amount": s["amount"]})
    return out
