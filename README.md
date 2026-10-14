# Agent From Scratch

A minimal LLM agent — tool calling, the agent loop, and an execution
sandbox — built directly on the Anthropic API with **no agent framework**.

This is Part 1 of a series building up a full agent stack: tool-calling
loop → orchestration → observability → evals → API. Framework code
(LangGraph, CrewAI, etc.) hides this loop behind abstractions; building it
by hand first means every framework choice later is an informed one.

## Why no framework?

Every agent framework is a wrapper around the same core loop: send messages
+ tool schemas to the model, execute whatever tool it asks for, feed the
result back, repeat. Writing that loop by hand — including the failure
modes (a model calling a tool that doesn't exist, an infinite loop, a
tool throwing an exception) — is the fastest way to actually understand
what frameworks are doing for you.

## What's here

- **`agent/agent.py`** — the core loop: call the model, detect `tool_use`,
  execute the tool, feed the result back, repeat until the model returns
  plain text or a turn limit is hit.
- **`agent/tools.py`** — three tools with a shared registry pattern
  (JSON schema + Python function):
  - `calculator` — safe arithmetic (AST-based, not `eval()`)
  - `write_file` / `read_file` — sandboxed to a `workspace/` directory,
    with path traversal blocked
- **`main.py`** — CLI entry point (interactive chat or single-question mode)
- **`tests/test_tools.py`** — unit tests for the tools, including a test
  that a path-traversal attempt (`../../evil.txt`) gets sandboxed

## Design decisions worth calling out

- **Tools are sandboxed on purpose.** File tools can't escape `workspace/`
  — `os.path.basename()` strips any `../` a model (or a prompt injection)
  might try to sneak in. Agents that can freely read/write the filesystem
  are a common real-world security mistake.
- **The calculator doesn't use `eval()`.** It parses the expression into
  an AST and only allows arithmetic operators — running arbitrary
  LLM-generated strings through `eval()` is a classic vulnerability.
- **`MAX_TURNS` guards against infinite loops.** If the model keeps
  calling tools without ever producing a final answer, the loop bails
  out instead of running forever.
- **`on_event` hook.** The loop reports `user_message`, `llm_response`,
  and `tool_call` events through a callback instead of hardcoding
  `print()` calls. Right now `main.py` just logs them to the console —
  this hook is where tracing (Langfuse/Phoenix) gets wired in next,
  without touching the loop itself.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
export $(cat .env | xargs)   # or use direnv / python-dotenv
```

## Run it

```bash
# Interactive chat
python main.py

# Single question
python main.py "Calculate 47 * 213 and save the result to a file called answer.txt"
```

Example trace from that second command:

```
[llm]  turn=0 stop_reason=tool_use latency=1.2s usage={...}
[tool] calculator({'expression': '47 * 213'}) -> 10011
[llm]  turn=1 stop_reason=tool_use latency=0.9s usage={...}
[tool] write_file({'filename': 'answer.txt', 'content': '10011'}) -> Wrote 5 characters to answer.txt
[llm]  turn=2 stop_reason=end_turn latency=0.6s usage={...}

Agent: I calculated 47 × 213 = 10011 and saved it to answer.txt.
```

## Run tests

```bash
python tests/test_tools.py
```

## What's next in this series

- [ ] Swap the framework-free loop for a LangGraph version, compare
- [ ] Real observability (Langfuse) instead of console logging
- [ ] A small eval set + LLM-as-judge scoring
- [ ] Wrap it as a FastAPI service with streaming

## License

MIT
