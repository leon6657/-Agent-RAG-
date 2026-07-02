"""LCEL chain definition for RAG question answering."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import config


def build_prompt() -> ChatPromptTemplate:
    template = (
        "You are a helpful assistant answering questions based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "---\n\n"
        "Question: {question}\n\n"
        "Answer the question based on the context above. "
        "If the context doesn't contain enough information, say so clearly. "
        "Answer in the same language as the question."
    )
    return ChatPromptTemplate.from_template(template)


def build_llm():
    if not config.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set. Add it to .env file.")
    return ChatOpenAI(
        model=config.deepseek_model,
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_api_base,
        temperature=0.3,
        timeout=30,
        max_retries=1,
    )


def build_chain(retriever):
    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    prompt = build_prompt()
    llm = build_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
