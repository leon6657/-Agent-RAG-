from pathlib import Path


def test_workbench_page_contains_core_regions():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert 'class="app-shell"' in html
    assert 'id="controlPanel"' in html
    assert 'id="chatPanel"' in html
    assert 'id="evidencePanel"' in html
    assert 'data-mode="query"' in html
    assert 'data-mode="chat"' in html
    assert 'data-question=' in html
    assert 'id="trustState"' in html
    assert 'id="sourceList"' in html
    assert 'id="monitorPanel"' in html
    assert 'id="reportView"' in html
    assert 'id="kbFileList"' in html
    assert 'id="logView"' in html
    assert 'id="historyList"' in html
    assert 'id="metricRecall"' in html
    assert 'id="metricTopScore"' in html


def test_workbench_page_offers_enough_example_questions():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert html.count('class="example-btn"') >= 10
    assert "Python 中列表和元组有什么区别？" in html
    assert "Embedding 模型在 RAG 系统中负责什么？" in html
    assert "如何把这个项目部署到公网？" in html
    assert "为什么 Render 免费实例第一次回答会比较慢？" in html
    assert 'class="section examples-section"' in html
    assert ".examples-section{min-height:0;flex:1}" in html
    assert ".example-list{display:flex;flex-direction:column;gap:8px;min-height:0;flex:1;overflow:auto;" in html


def test_workbench_page_keeps_existing_api_contract():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert 'fetch("/query/stream"' in html
    assert 'fetch("/chat/stream"' in html
    assert 'fetch("/" + mode' in html
    assert "JSON.stringify({question: q})" in html
    assert "data.sources" in html
    assert "data.top_score" in html
    assert 'fetch("/reports")' in html
    assert 'fetch("/metrics")' in html
    assert 'fetch("/kb/files")' in html
    assert 'fetch("/logs/recent")' in html
    assert 'fetch("/config")' in html
    assert 'fetch("/ingest"' in html


def test_workbench_page_streams_and_formats_answers():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert "response.body.getReader()" in html
    assert "TextDecoder" in html
    assert "appendAssistantToken" in html
    assert "function renderAnswerHtml" in html
    assert "<strong>$1</strong>" in html
    assert "function renderMarkdownTable" in html
    assert "answer-content" in html
    assert "function renderMarkdownHeading" in html
    assert "<h3>" in html
    assert 'start="' in html
    assert "orderedListNext" in html


def test_workbench_page_uses_chinese_labels():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert "&#20005;&#26684; RAG" in html
    assert "&#21457;&#36865;" in html
    assert "&#35777;&#25454;" in html
    assert "&#21487;&#20449;&#29366;&#24577;" in html
    assert "&#30417;&#25511;" in html
    assert "&#37325;&#26032;&#24314;&#24211;" in html


def test_workbench_page_uses_local_history():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert "localStorage" in html
    assert "ragWorkbenchHistory" in html
    assert "function saveHistory" in html
    assert "function renderHistory" in html


def test_workbench_page_refreshes_dynamic_metrics_and_logs():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert "function loadMetrics" in html
    assert "formatMetric" in html
    assert "buildReportSummary" in html
    assert "await loadMetrics()" in html
    assert "await loadLogs()" in html
    assert "最近一次问题" in html


def test_workbench_layout_uses_fixed_chat_and_scrollable_monitor_cards():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert ".app-shell{min-height:100vh;" in html
    assert "grid-template-rows:clamp(620px,calc(100vh - 140px),calc(100vh - 32px)) auto" in html
    assert ".control-panel,.chat-panel,.evidence-panel{height:clamp(620px,calc(100vh - 140px),calc(100vh - 32px));" in html
    assert ".messages{min-height:0;overflow-y:auto;" in html
    assert 'class="panel monitor-panel collapsed"' in html
    assert 'id="toggleMonitorBtn"' in html
    assert ".monitor-panel.collapsed .monitor-grid{display:none}" in html
    assert "function toggleMonitor" in html
    assert ".monitor-card{height:260px;display:grid;grid-template-rows:auto minmax(0,1fr);" in html
    assert ".monitor-scroll{height:100%;max-height:none;overflow:auto;" in html
    assert ".file-list,.history-list,.config-list{min-height:0;overflow:auto;" in html
