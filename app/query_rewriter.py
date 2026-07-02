"""Query rewriting: Multi-Query generation for better retrieval."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

_MULTI_QUERY_PROMPT = """You are a helpful assistant that helps search.
Given the original question, generate {n} different versions of the question.
Each version should rephrase the question from a different angle.
Keep the meaning the same but use different wording.

Original question: {question}

Generate {n} alternative questions, one per line, numbered 1-{n}:"""


def generate_queries(question: str, n: int = 3) -> list:
    """Generate n query variations using the LLM."""
    from app.chain import build_llm
    prompt = ChatPromptTemplate.from_template(_MULTI_QUERY_PROMPT)
    chain = prompt | build_llm() | StrOutputParser()
    result = chain.invoke({"question": question, "n": n})
    variations = []
    for line in result.strip().split("\n"):
        line = line.strip()
        # Remove numbering like "1. " or "1: "
        if line and (line[0].isdigit() or line.startswith("-")):
            line = line.split(".", 1)[-1] if "." in line else line
            line = line.split(":", 1)[-1] if ":" in line else line
            line = line.strip()
        if line and len(line) > 5:
            variations.append(line)
    return variations[:n]


if __name__ == "__main__":
    import os
    os.environ["HF_HOME"] = str(os.path.join(os.path.dirname(__file__), "..", ".hf_cache"))
    qs = generate_queries("Python装饰器是什么？", n=3)
    print("Original: Python装饰器是什么？")
    for i, q in enumerate(qs, 1):
        print(f"  {i}. {q}")
