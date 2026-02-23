import os
import json
import argparse

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Content,
    Part,
    FunctionResponse,
    GenerateContentConfig,
)
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter

from tools_schema import ToolsSchemaForTest  # the one you defined
from system_instruction import system_instruction
from turns import turns


def build_contents(turn_index: int) -> list[Content]:
    """Concatenate all previous user/assistant turns plus the current user input
    to build the conversation context for the model."""
    conversation_parts: list[Content] = []
    # Conversation history
    for i in range(turn_index):
        if turns[i]["input"]:
            conversation_parts.append(Content(role="user", parts=[Part(text=turns[i]["input"])]))
        if turns[i]["required_function_call"]:
            name = turns[i]["required_function_call"]["name"]
            conversation_parts.append(
                Content(
                    role="model",
                    parts=[{"function_call": turns[i]["required_function_call"]}],
                )
            )
            conversation_parts.append(
                Content(
                    role="user",
                    parts=[
                        Part(
                            function_response=FunctionResponse(
                                name=name,
                                response=turns[i]["function_call_response"],
                            )
                        )
                    ],
                )
            )
        if turns[i]["golden_text"]:
            conversation_parts.append(
                Content(role="model", parts=[Part(text=turns[i]["golden_text"])])
            )

    # Current user input
    if turns[turn_index]["input"]:
        conversation_parts.append(
            Content(role="user", parts=[Part(text=turns[turn_index]["input"])])
        )
    return conversation_parts


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Gemini 2.5 Pro content generation test")
parser.add_argument(
    "--turn-index",
    type=int,
    default=0,
    help="Index of the turn to use from the turns list.",
)
args = parser.parse_args()

adapter = GeminiLLMAdapter()
google_tools_json = adapter.to_provider_tools_format(ToolsSchemaForTest)

print(google_tools_json)

# --- Test inference call with Gemini 2.5 Pro ---

# Load environment variables from .env file (if present)
load_dotenv()

# Configure Google API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

# Instantiate client using the new SDK pattern
client = genai.Client(api_key=api_key)

contents = build_contents(args.turn_index)
print("===\nContents:\n", contents, "\n===")

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=contents,
    config=GenerateContentConfig(
        system_instruction=system_instruction,
        tools=google_tools_json,
        # thinking_config=ThinkingConfig(thinking_budget=0)
    ),
)

print("Raw response:", response, "\n")

# --- Extract text content and function calls ---
texts: list[str] = []
function_calls = []

for candidate in response.candidates:
    for part in candidate.content.parts:
        # Collect any text returned by the model
        if getattr(part, "text", None):
            texts.append(part.text)
        # Collect any function calls returned by the model
        if getattr(part, "function_call", None):
            function_calls.append(part.function_call)

# Print text content
print("\n=== Model Text Response ===")
if texts:
    print("\n".join(texts))
else:
    print("No text content found.")

# Print function calls
print("\n=== Function Call(s) ===")
if function_calls:
    for idx, fc in enumerate(function_calls, start=1):
        info = {"name": fc.name, "args": fc.args}
        print(json.dumps(info, indent=2))
else:
    print("No function calls found.")
