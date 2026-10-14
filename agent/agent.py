"""
The agent loop — the actual "agent" part of an agent.

The pattern, stripped to its essence:

    1. Send the conversation + tool schemas to the model.
    2. If the model's response contains a tool_use block, run that tool
       locally and feed the result back as a tool_result message.
    3. Repeat until the model responds with plain text (stop_reason == "end_turn").

Everything else (LangGraph, CrewAI, AutoGPT, etc.) is a more elaborate
version of this same loop, usually adding: multiple agents talking to
each other, persistent state/memory, parallel tool calls, retries, and
observability hooks.
"""

import json
import os
import time

import anthropic

from agent.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

MODEL = "claude-sonnet-4-5"
MAX_TURNS = 8  # safety cap so a confused agent can't loop forever


class Agent:
    def __init__(self, system_prompt: str, on_event=None):
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.system_prompt = system_prompt
        self.messages = []
        # on_event lets us plug in observability later without touching the loop
        self.on_event = on_event or (lambda event, data: None)

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self.on_event("user_message", {"content": user_input})

        for turn in range(MAX_TURNS):
            start = time.time()
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=self.system_prompt,
                tools=TOOL_SCHEMAS,
                messages=self.messages,
            )
            latency = time.time() - start

            self.on_event("llm_response", {
                "stop_reason": response.stop_reason,
                "usage": response.usage.model_dump() if response.usage else None,
                "latency_s": round(latency, 2),
                "turn": turn,
            })

            # Save the assistant's turn (may include text + tool_use blocks)
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # Model is done — extract and return the final text
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                return final_text

            # Model wants to use one or more tools — execute each and collect results
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = self._execute_tool(block.name, block.input)
                self.on_event("tool_call", {
                    "name": block.name,
                    "input": block.input,
                    "result": result,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            # Feed the tool results back to the model as the next user turn
            self.messages.append({"role": "user", "content": tool_results})

        return "Stopped: hit MAX_TURNS without a final answer. Possible loop."

    def _execute_tool(self, name: str, tool_input: dict) -> str:
        fn = TOOL_FUNCTIONS.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            return fn(**tool_input)
        except Exception as e:
            return f"Error running tool '{name}': {e}"
