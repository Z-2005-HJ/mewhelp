# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
def test_create_refund_writes_ticket(client, monkeypatch):
    audits = []

    async def fake_conversation(cid):
        assert cid == 7
        return type("Conversation", (), {"user_id": "eval-user"})()

    async def fake_create(cid, desc, ttype):
        assert cid == 7 and ttype == "退款" and "1001" in desc and "质量问题" in desc
        return "T20260716001"

    async def fake_audit(**kwargs):
        audits.append(kwargs)

    from app.api import actions
    monkeypatch.setattr(actions.repository, "get_conversation", fake_conversation)
    monkeypatch.setattr(actions.business, "owns_order", lambda uid, oid: uid == "eval-user" and oid == "1001")
    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    monkeypatch.setattr(actions.repository, "insert_tool_audit", fake_audit)
    r = client.post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "1001", "reason": "质量问题"})
    assert r.status_code == 200
    assert r.json()["ticket_no"] == "T20260716001"
    assert r.json()["status"] == "退款申请已提交"
    assert len(audits) == 1
    assert audits[0]["tool_name"] == "submit_refund"
    assert audits[0]["arguments"] == {"order_id": "1001", "reason": "质量问题"}
    assert audits[0]["status"] == "成功"


def test_create_refund_rejects_order_not_owned_by_conversation(client, monkeypatch):
    audits = []

    async def fake_conversation(cid):
        return type("Conversation", (), {"user_id": "eval-user"})()

    async def fake_audit(**kwargs):
        audits.append(kwargs)

    async def should_not_create(*args):
        raise AssertionError("unauthorized refund must not write a ticket")

    from app.api import actions
    monkeypatch.setattr(actions.repository, "get_conversation", fake_conversation)
    monkeypatch.setattr(actions.business, "owns_order", lambda uid, oid: False)
    monkeypatch.setattr(actions.repository, "insert_tool_audit", fake_audit)
    monkeypatch.setattr(actions.repository, "create_ticket", should_not_create)
    r = client.post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "9999", "reason": "质量问题"})

    assert r.status_code == 403
    assert audits[0]["tool_name"] == "submit_refund"
    assert audits[0]["status"] == "权限拒绝"
    assert audits[0]["arguments"]["order_id"] == "9999"


def test_create_refund_rejects_bad_reason(client):
    r = client.post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "1001", "reason": "乱填"})
    assert r.status_code == 422


def test_resume_endpoint_streams(client, monkeypatch):
    async def fake_stream_resume(cid, resume_value):
        assert cid == 3 and resume_value == "1001"
        yield {"type": "delta", "text": "这一单可以退款"}
        yield {"type": "actions", "items": [{"type": "refund_form", "draft": {"order_id": "1001"}}]}
        yield {"type": "done", "conversation_id": 3}
    from app.api import actions
    monkeypatch.setattr(actions.runtime, "stream_resume", fake_stream_resume)
    r = client.post("/api/actions/resume", json={"conversation_id": 3, "order_id": "1001"})
    assert r.status_code == 200
    body = r.text
    assert "这一单可以退款" in body
    assert "refund_form" in body and "[DONE]" in body


def test_resume_endpoint_confirmed_branch(client, monkeypatch):
    """ch08:工单预览卡确认/取消走 confirmed 字段,resume 值为 {"confirmed": bool}。"""
    async def fake_stream_resume(cid, resume_value):
        assert cid == 3 and resume_value == {"confirmed": False}
        yield {"type": "delta", "text": "已取消建单"}
        yield {"type": "done", "conversation_id": 3}
    from app.api import actions
    monkeypatch.setattr(actions.runtime, "stream_resume", fake_stream_resume)
    r = client.post("/api/actions/resume", json={"conversation_id": 3, "confirmed": False})
    assert r.status_code == 200 and "已取消建单" in r.text


def test_resume_endpoint_requires_one_of(client):
    """order_id 与 confirmed 都不传 → 400,不进图。"""
    r = client.post("/api/actions/resume", json={"conversation_id": 3})
    assert r.status_code == 400
