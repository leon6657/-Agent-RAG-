from pathlib import Path


def test_dockerfile_builds_fixed_knowledge_demo():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile
    assert "python main.py --ingest" in dockerfile
    assert "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "build-essential" not in dockerfile


def test_dockerignore_excludes_local_only_and_secret_files():
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert ".env" in dockerignore
    assert ".venv-conda/" in dockerignore
    assert ".git/" in dockerignore
    assert "logs/" in dockerignore
    assert "__pycache__/" in dockerignore


def test_demo_deployment_doc_mentions_required_cloud_settings():
    doc = Path("docs/deployment-demo.md").read_text(encoding="utf-8")

    assert "固定知识库 Demo" in doc
    assert "DEEPSEEK_API_KEY" in doc
    assert "DEEPSEEK_MODEL" in doc
    assert "RAG_MIN_SCORE" in doc
    assert "/health" in doc
    assert "Render" in doc
    assert "Railway" in doc


def test_requirements_are_pinned_for_reproducible_docker_builds():
    requirements = [
        line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    package_requirements = [line for line in requirements if not line.startswith("--")]

    assert requirements
    assert all("==" in line for line in package_requirements)
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in requirements
    assert "langchain-core==1.5.1" in requirements
    assert "langchain-community==0.4.2" in requirements
    assert "langchain-openai==1.4.1" in requirements
    assert "torch==2.13.0+cpu" in requirements
    assert "numpy==2.4.6" in requirements
