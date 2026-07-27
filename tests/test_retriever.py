from app.retriever import _tokenize


def test_tokenize_splits_chinese_terms_with_jieba():
    tokens = _tokenize("列表推导式的语法是什么？")

    assert "列表" in tokens
    assert "推导" in tokens or "推导式" in tokens
    assert "语法" in tokens
