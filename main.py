"""
Run the agent from the command line.

Usage:
    python main.py                       # interactive chat
    python main.py "What is 42 * 17?"    # single question
"""

import sys

from agent.agent import Agent

SYSTEM_PROMPT = """You are a helpful assistant with access to a calculator and
a sandboxed file read/write tool. Use tools whenever they'd give a more
reliable answer than reasoning it out yourself (e.g. always use the
calculator for arithmetic). Be concise."""


def log_event(event: str, data: dict) -> None:
    """
    Minimal observability hook: just prints to console for now.
    In Step 3 this gets swapped for something that sends traces to
    Langfuse/Phoenix instead of (or in addition to) printing.
    """
    if event == "tool_call":
        print(f"  [tool] {data['name']}({data['input']}) -> {data['result']}")
    elif event == "llm_response":
        print(f"  [llm]  turn={data['turn']} stop_reason={data['stop_reason']} "
              f"latency={data['latency_s']}s usage={data['usage']}")


def main():
    agent = Agent(system_prompt=SYSTEM_PROMPT, on_event=log_event)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = agent.run(question)
        print(f"\nAgent: {answer}")
        return

    print("Agent ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        answer = agent.run(user_input)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
