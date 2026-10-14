"""
Same agent loop as agent.py, but using Google's Gemini API instead of
Anthropic's — because Gemini has a genuinely free tier, good for learning
without spending money. The core loop logic is identical: this is the
point. Only the API calls and response format differ.
"""

import time

import google.generativeai as genai

from agent.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

MODEL = "gemini-3.8-flash"
MAX_TURNS = 8


def _to_gemini_tools():
    """Convert our tool schemas (Anthropic-style) into Gemini's format."""
    declarations = []
    for schema in TOOL_SCHEMAS:
        declarations.append({
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["input_schema"],
        })
    return [{"function_declarations": declarations}]


class GeminiAgent:
    def __init__(self, system_prompt: str, on_event=None):
        self.model = genai.GenerativeModel(
            model_name=MODEL,
            system_instruction=system_prompt,
            tools=_to_gemini_tools(),
        )
        self.chat = self.model.start_chat()
        self.on_event = on_event or (lambda event, data: None)

    def run(self, user_input: str) -> str:
        self.on_event("user_message", {"content": user_input})
        message = user_input

        for turn in range(MAX_TURNS):
            start = time.time()
            response = self.chat.send_message(message)
            latency = time.time() - start

            part = response.candidates[0].content.parts[0]
            has_function_call = hasattr(part, "function_call") and part.function_call.name

            self.on_event("llm_response", {
                "turn": turn,
                "latency_s": round(latency, 2),
                "has_tool_call": bool(has_function_call),
            })

            if not has_function_call:
                return response.text

            fn_call = part.function_call
            result = self._execute_tool(fn_call.name, dict(fn_call.args))
            self.on_event("tool_call", {
                "name": fn_call.name,
                "input": dict(fn_call.args),
                "result": result,
            })

            # Feed the tool result back as the next message
            message = genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fn_call.name,
                        response={"result": result},
                    )
                )]
            )

        return "Stopped: hit MAX_TURNS without a final answer. Possible loop."

    def _execute_tool(self, name: str, tool_input: dict) -> str:
        fn = TOOL_FUNCTIONS.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            return fn(**tool_input)
        except Exception as e:
            return f"Error running tool '{name}': {e}"
