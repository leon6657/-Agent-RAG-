from app import query


def test_answer_style_instruction_prefers_structured_chinese_markdown():
    instruction = query.ANSWER_STYLE_INSTRUCTIONS

    assert "Markdown" in instruction
    assert "\u6838\u5fc3\u4f5c\u7528" in instruction
    assert "\u793a\u4f8b" in instruction
    assert "\u6ce8\u610f\u4e8b\u9879" in instruction
    assert "\u8868\u683c" in instruction
    assert "\u4e0d\u8981\u4f7f\u7528 ###" in instruction
    assert "\u8fde\u7eed\u7f16\u53f7" in instruction
