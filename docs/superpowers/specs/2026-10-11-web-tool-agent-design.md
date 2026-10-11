# V2 Web Tool Agent Design

## Goal

Upgrade the web Agent mode from deterministic retrieval routing to a controlled
Function Calling loop. The model can decide whether to invoke a read-only tool,
observe its result, and then produce the final answer.

## Scope

- Reuse the existing OpenAI-compatible function calling protocol.
- Expose two tools only: `search_knowledge_base` and `get_current_time`.
- Integrate the loop with `/chat` and `/chat/stream`.
- Preserve Chinese streaming answers, source cards, timing metadata, and
  short-term conversation history.
- Emit tool lifecycle events for the existing web UI: selected, running, and
  completed.
- Bound execution to three model/tool iterations and return a clear error event
  for invalid arguments or provider failures.

## Out Of Scope

- Web search, external write tools, and autonomous background tasks.
- Multi-Agent supervisor, handoffs, persistent memory, and user accounts.
- Changes to strict RAG endpoints (`/query` and `/query/stream`).

## Architecture

`/chat` and `/chat/stream` call a new reusable tool-agent runner. It builds
chat messages, sends available tool schemas to the model, executes requested
read-only tools, adds `tool` messages back to the conversation, and repeats
until the model returns final text or the iteration limit is reached.

The knowledge-base tool returns structured context and source metadata rather
than only a rendered string. The runner accumulates the source metadata so the
API can keep returning the existing `sources` and `top_score` fields.

For streaming, tool events are sent before final token events. Tool execution
itself is synchronous and local; only the final model response is streamed.

## Error Handling

- Unknown tools and invalid JSON arguments produce a bounded tool-result error
  for the model to recover from.
- Provider failures produce a user-visible error event and do not retry
  indefinitely.
- A maximum of three iterations prevents loops and uncontrolled token costs.

## Tests

- The model can answer directly without a tool call.
- The model calls the knowledge-base tool and returned source metadata reaches
  the stream metadata.
- The model calls the time tool and receives its result.
- A malformed tool request does not crash the API.
- The iteration bound ends an endless tool-call sequence safely.
