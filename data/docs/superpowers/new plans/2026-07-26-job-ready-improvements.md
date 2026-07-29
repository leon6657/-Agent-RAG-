# Job Ready Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RAG knowledge-base project safer, easier to reproduce, and more convincing as a university job-search portfolio project.

**Architecture:** Keep the existing RAG architecture intact: Markdown ingestion, JSON+numpy vector store, DeepSeek generation, FastAPI service, and evaluation runner. Add portfolio-focused polish around secrets, dependencies, README accuracy, and retrieval evaluation diagnostics.

**Tech Stack:** Python 3.9+, LangChain, DeepSeek OpenAI-compatible API, sentence-transformers BGE embeddings, numpy JSON vector store, FastAPI, pytest.

## Global Constraints

- Do not rewrite the whole application.
- Do not commit or expose real API keys.
- Keep current CLI commands working: `python main.py --ingest`, `python main.py --query`, `python main.py --chat`, `python main.py --serve`, and `python evaluation/runner.py`.
- Prefer small, testable improvements that are easy to explain in interviews.
- Keep documentation aligned with current measured project state: 30 evaluation questions, 49 indexed chunks, and current Recall@4 of 0.433.

---

### Task 1: Secret Hygiene and Reproducible Setup

**Files:**
- Create: `.env.example`
- Create: `requirements.txt`
- Modify: `.env`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Existing `app.config.Config` environment loading.
- Produces: A safe example env file and dependency list for new users.

- [ ] Replace local `.env` API key value with a placeholder.
- [ ] Add `.env.example` with `DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL`.
- [ ] Update `pyproject.toml` dependencies to match actual imports.
- [ ] Add `requirements.txt` for simple `pip install -r requirements.txt` onboarding.

### Task 2: Evaluation Diagnostics

**Files:**
- Create: `tests/test_evaluation_runner.py`
- Modify: `evaluation/runner.py`

**Interfaces:**
- Consumes: Existing `run_evaluation(search_fn, questions, k=4)`.
- Produces: `results["cases"]`, `results["misses"]`, and an optional Markdown report writer.

- [ ] Write tests proving evaluation records per-question hit/miss details.
- [ ] Implement case-level diagnostics without changing aggregate metric behavior.
- [ ] Add `write_markdown_report(path, baseline, optimized)` to save readable reports.
- [ ] Keep `python evaluation/runner.py` printing the existing console summary.

### Task 3: Portfolio Documentation

**Files:**
- Modify: `README.md`
- Modify: `USAGE.md`

**Interfaces:**
- Consumes: Current code and measured evaluation output.
- Produces: A README first screen suitable for recruiters and interviewers.

- [ ] Add a concise portfolio overview near the top of README.
- [ ] Correct current project numbers: 11 source documents, 49 chunks, 30 QA pairs.
- [ ] Clearly separate historical optimization claims from current measured results.
- [ ] Update usage instructions to mention `.env.example` and current model config.

### Task 4: Verification

**Files:**
- Test: `tests/`
- Test: `evaluation/runner.py`

**Interfaces:**
- Consumes: All changed files.
- Produces: Fresh verification evidence.

- [ ] Run targeted tests for config and evaluation runner.
- [ ] Run full pytest suite.
- [ ] Run `python evaluation/runner.py` to verify the main evaluation entrypoint still works.
