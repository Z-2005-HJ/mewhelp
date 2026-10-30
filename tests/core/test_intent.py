from app.core.intent import infer_explicit_order_intent


def test_explicit_order_lookup_falls_back_to_business_intent():
    assert infer_explicit_order_intent("帮我查一下订单1001的状态") == "订单"
    assert infer_explicit_order_intent("订单1001的物流到哪了") == "物流"
    assert infer_explicit_order_intent("订单1001申请退款") == "退款退货"


def test_order_intent_fallback_ignores_questions_without_order_numbers():
    assert infer_explicit_order_intent("退货政策是什么") is None
    assert infer_explicit_order_intent("MH-LP100 有库存吗") is None
