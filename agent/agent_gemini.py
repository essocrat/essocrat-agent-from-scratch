"""
Same agent loop as before, now on Google's NEW SDK (google-genai).

Why the switch: Gemini 3 models attach a hidden "thought signature" to every
tool request, and it must be sent back with the tool result. The old
google.generativeai package drops it (causing a 400 error); this SDK keeps it
as long as we store the model's reply exactly as it came back.
"""

import time

from google import genai
from google.genai import types

from agent.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

MODEL = "gemini-3.5-flash-lite"
MAX_TURNS = 8


def _to_gemini_tools():
    """Convert our tool schemas into Gemini's format."""
    declarations = [
        {
            "name": s["name"],
            "description": s["description"],
            "parameters": s["input_schema"],
        }
        for s in TOOL_SCHEMAS
    ]
    return [types.Tool(function_declarations=declarations)]


class GeminiAgent:
    def __init__(self, system_prompt: str, api_key: str | None = None, on_event=None):
        self.client = genai.Client(api_key=api_key)
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=_to_gemini_tools(),
        )
        self.contents = []  # the whole conversation, kept by us
        self.on_event = on_event or (lambda event, data: None)

    def run(self, user_input: str) -> str:
        self.on_event("user_message", {"content": user_input})
        self.contents.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        for turn in range(MAX_TURNS):
            start = time.time()
            response = self.client.models.generate_content(
                model=MODEL,
                contents=self.contents,
                config=self.config,
            )
            latency = time.time() - start

            calls = response.function_calls or []
            self.on_event("llm_response", {
                "turn": turn,
                "latency_s": round(latency, 2),
                "has_tool_call": bool(calls),
            })

            # Save the model's reply EXACTLY as received (keeps thought signatures)
            self.contents.append(response.candidates[0].content)

            if not calls:
                return response.text

            # Run each requested tool and collect the results
            result_parts = []
            for call in calls:
                args = dict(call.args or {})
                result = self._execute_tool(call.name, args)
                self.on_event("tool_call", {
                    "name": call.name,
                    "input": args,
                    "result": result,
                })
                result_parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result},
                    )
                )

            # Send the results back as the next turn
            self.contents.append(types.Content(role="user", parts=result_parts))

        return "Stopped: hit MAX_TURNS without a final answer. Possible loop."

    def _execute_tool(self, name: str, tool_input: dict) -> str:
        fn = TOOL_FUNCTIONS.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            return fn(**tool_input)
        except Exception as e:
            return f"Error running tool '{name}': {e}"
