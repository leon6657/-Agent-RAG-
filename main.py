"""CLI entry point for the RAG knowledge base.

Usage:
    python main.py --ingest       Build/update vector store
    python main.py --query        Interactive Q&A session
"""

import argparse
import sys

from app.ingest import run_ingest
from app.query import ask, ask_stream
from app.config import config


def main():
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Base - Q&A with your Markdown notes"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Index all .md files in data/ into ChromaDB",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Start interactive Q&A session",
    )

    args = parser.parse_args()

    if not args.ingest and not args.query:
        parser.print_help()
        return

    if args.ingest:
        if not config.deepseek_api_key or config.deepseek_api_key.startswith("sk-your"):
            print("ERROR: DEEPSEEK_API_KEY not configured.")
            print("Set it in .env file or as environment variable.")
            sys.exit(1)
        import os
        os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".hf_cache")

        num_chunks = run_ingest()
        if num_chunks > 0:
            print(f"\nDone! Indexed {num_chunks} chunks.")

    if args.query:
        if not config.deepseek_api_key or config.deepseek_api_key.startswith("sk-your"):
            print("ERROR: DEEPSEEK_API_KEY not configured.")
            print("Set it in .env file or as environment variable.")
            sys.exit(1)
        import os
        os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".hf_cache")

        print("RAG Knowledge Base - Interactive Q&A")
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
                    import os as _os
                    _os.system("cls" if _os.name == "nt" else "clear")
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


if __name__ == "__main__":
    main()
