"""CLI entry point for the RAG knowledge base.

Usage:
    python main.py --ingest       Build/update vector store
    python main.py --query        Interactive Q and A session
    python main.py --chat         Interactive Agent session
"""

import argparse
import os
import sys

from app.ingest import run_ingest
from app.query import ask, ask_stream
from app.config import config


def main():
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Base"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Index all .md files in data/ into vector store",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Start interactive Q and A session",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive Agent session with memory",
    )

    args = parser.parse_args()

    if not args.ingest and not args.query and not args.chat:
        parser.print_help()
        return

    os.environ.setdefault(
        "HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache")
    )

    if args.ingest:
        num_chunks = run_ingest()
        if num_chunks > 0:
            print(f"\nDone! Indexed {num_chunks} chunks.")

    if args.query:
        key = config.deepseek_api_key
        if not key or key.startswith("sk-your"):
            print("ERROR: DEEPSEEK_API_KEY not configured.")
            sys.exit(1)

        print("RAG Knowledge Base - Interactive Q and A")
        print("Type 'exit' to quit, 'clear' to clear screen")
        print("-" * 50)

        while True:
            try:
                question = input("\nYou: ").strip()
                if not question:
                    continue
                if question.lower() == "exit":
                    print("Goodbye!")
                    break
                if question.lower() == "clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue

                print("\nAssistant: ", end="", flush=True)
                for chunk in ask_stream(question):
                    print(chunk, end="", flush=True)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

    if args.chat:
        key = config.deepseek_api_key
        if not key or key.startswith("sk-your"):
            print("ERROR: DEEPSEEK_API_KEY not configured.")
            sys.exit(1)

        from app.agent import chat as agent_chat

        print("Agent Mode - search notes or chat freely")
        print("Type 'exit' to quit")
        print("-" * 50)

        while True:
            try:
                msg = input("\nYou: ").strip()
                if not msg:
                    continue
                if msg.lower() == "exit":
                    print("Goodbye!")
                    break

                print("\nAgent: ", end="", flush=True)
                response = agent_chat(msg)
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


if __name__ == "__main__":
    main()
