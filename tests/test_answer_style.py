from app import query


def test_answer_style_instruction_prefers_structured_chinese_markdown():
    instruction = query.ANSWER_STYLE_INSTRUCTIONS

    assert "Markdown" in instruction
    assert "核心作用" in instruction
    assert "示例" in instruction
    assert "注意事项" in instruction
    assert "表格" in instruction
