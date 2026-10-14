"""
Run the Gemini version of the agent (free tier).

Setup:
    1. Free key: https://aistudio.google.com/apikey
    2. set GOOGLE_API_KEY=your-key      (Windows)
    3. pip install google-genai
    4. python main_gemini.py "your question"
"""

import os
import sys

from agent.agent_gemini import GeminiAgent

SYSTEM_PROMPT = """You are a helpful assistant with access to a calculator and
a sandboxed file read/write tool. Use tools whenever they'd give a more
reliable answer than reasoning it out yourself. Be concise."""


def log_event(event: str, data: dict) -> None:
    if event == "tool_call":
        print(f"  [tool] {data['name']}({data['input']}) -> {data['result']}")
    elif event == "llm_response":
        print(f"  [llm]  turn={data['turn']} latency={data['latency_s']}s "
              f"tool_call={data['has_tool_call']}")


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: set GOOGLE_API_KEY in your environment (see .env).")
        sys.exit(1)

    agent = GeminiAgent(system_prompt=SYSTEM_PROMPT, api_key=api_key, on_event=log_event)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\nAgent: {agent.run(question)}")
        return

    print("Agent ready (Gemini free tier). Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        print(f"Agent: {agent.run(user_input)}\n")


if __name__ == "__main__":
    main()
