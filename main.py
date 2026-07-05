# CLI entry point for the RAG knowledge base.

import argparse
import os
import sys

from app.ingest import run_ingest
from app.query import ask, ask_stream
from app.config import config


def main():
    parser = argparse.ArgumentParser(description="RAG Knowledge Base")
    parser.add_argument("--ingest", action="store_true", help="Index .md files into vector store")
    parser.add_argument("--query", action="store_true", help="Interactive Q and A session")
    parser.add_argument("--tool", action="store_true", help="Tool calling agent with function calling")
parser.add_argument("--chat", action="store_true", help="Interactive Agent session with memory")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI web service")

    args = parser.parse_args()

    if not any([args.ingest, args.query, args.chat, args.tool, args.serve]):
        parser.print_help()
        return

    os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))

    if args.tool:
        from app.tool_agent import chat
        print("Tool mode\n" + "-" * 40)
        while True:
            msg = input("\nYou: ").strip()
            if not msg or msg.lower() == "exit":
                if msg: print("Goodbye!")
                break
            print("\nAssistant: ", end="", flush=True)
            print(chat(msg))
            print()
        return

    if args.serve:
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
        return

    if args.ingest:
        num = run_ingest()
        if num > 0:
            print(f"\nDone! Indexed {num} chunks.")

    if args.query or args.chat:
        key = config.deepseek_api_key
        if not key or key.startswith("sk-your"):
            print("ERROR: DEEPSEEK_API_KEY not configured.")
            sys.exit(1)

    if args.query:
        print("RAG Knowledge Base - Interactive Q and A")
        print("Type 'exit' to quit\n" + "-" * 50)
        while True:
            try:
                q = input("\nYou: ").strip()
                if not q or q.lower() == "exit":
                    if q: print("Goodbye!")
                    break
                if q.lower() == "clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue
                print("\nAssistant: ", end="", flush=True)
                for c in ask_stream(q):
                    print(c, end="", flush=True)
                print()
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

    if args.chat:
        from app.agent import chat as agent_chat
        print("Agent Mode - search notes or chat freely")
        print("Type 'exit' to quit\n" + "-" * 50)
        while True:
            try:
                msg = input("\nYou: ").strip()
                if not msg or msg.lower() == "exit":
                    if msg: print("Goodbye!")
                    break
                print("\nAgent: ", end="", flush=True)
                resp = agent_chat(msg)
                print(resp)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


if __name__ == "__main__":
    main()
