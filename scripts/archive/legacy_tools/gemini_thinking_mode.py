"""
Utility module for Google Gemini models with thinking mode support.

This module provides a custom GoogleLLMService that properly handles:
- Thinking mode with thinking_budget and include_thoughts
- Function calls with thought_signature (required for gemini-3-pro-preview)
- Proper message formatting for Gemini's Content/Part structure
"""

from typing import Any, AsyncIterator, Dict, List, Optional
import json
from datetime import datetime
from pathlib import Path

from loguru import logger
from google.genai import types as genai_types
from google.genai.types import Content, GenerateContentResponse

from pipecat.frames.frames import (
    Frame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.services.google.llm import GoogleLLMService as PipecatGoogleLLMService


class _GeminiThinkingModeContentFrame(Frame):
    """Frame containing Gemini thinking mode Content objects."""

    def __init__(self, contents: List[Content]):
        super().__init__()
        self.contents = contents


class GeminiThinkingModeLLMService(PipecatGoogleLLMService):
    """
    Google LLM Service with thinking mode support.

    This service:
    1. Captures raw Content objects from Gemini API responses
    2. Emits _GeminiThinkingModeContentFrame for processing thoughts vs text
    3. Properly formats function response messages
    4. Cleans up duplicate function call messages
    """

    def __init__(self, *args, chunk_log_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._captured_candidate_contents: List[Content] = []
        self._chunk_log_path = chunk_log_path
        self._chunk_index = 0
        self._turn_number = 0

        # Initialize chunk log file
        if self._chunk_log_path:
            log_path = Path(self._chunk_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Clear existing file
            with open(log_path, 'w') as f:
                f.write('')  # Start fresh
            logger.info(f"Chunk logging enabled: {self._chunk_log_path}")

    async def cancel(self, frame):
        await super().cancel(frame)
        try:
            await self._client.aio.aclose()
        except Exception:
            pass

    def _log_chunk(self, chunk: GenerateContentResponse, chunk_index: int):
        """Log a raw chunk to the chunk log file."""
        if not self._chunk_log_path:
            return

        try:
            # Extract key information from the chunk
            chunk_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "turn_number": self._turn_number,
                "chunk_index": chunk_index,
                "candidates": [],
                "usage_metadata": None,
                "model_version": getattr(chunk, 'model_version', None),
            }

            # Extract candidates
            if hasattr(chunk, 'candidates') and chunk.candidates:
                for candidate in chunk.candidates:
                    candidate_data = {
                        "finish_reason": getattr(candidate, 'finish_reason', None),
                        "safety_ratings": [],
                        "content": None,
                    }

                    # Extract content
                    if hasattr(candidate, 'content') and candidate.content:
                        content = candidate.content
                        content_data = {
                            "role": getattr(content, 'role', None),
                            "parts": [],
                        }

                        # Extract parts
                        if hasattr(content, 'parts') and content.parts:
                            for part in content.parts:
                                part_data = {}

                                # Text
                                if hasattr(part, 'text') and part.text:
                                    part_data['text'] = part.text
                                    part_data['thought'] = getattr(part, 'thought', False)

                                # Function call
                                if hasattr(part, 'function_call') and part.function_call:
                                    fc = part.function_call
                                    part_data['function_call'] = {
                                        'name': getattr(fc, 'name', None),
                                        'args': dict(getattr(fc, 'args', {})),
                                        'id': getattr(fc, 'id', None),
                                    }

                                # Function response
                                if hasattr(part, 'function_response') and part.function_response:
                                    fr = part.function_response
                                    part_data['function_response'] = {
                                        'name': getattr(fr, 'name', None),
                                        'response': getattr(fr, 'response', None),
                                    }

                                # thought_signature at Part level (CRITICAL!)
                                if hasattr(part, 'thought_signature') and part.thought_signature:
                                    # Convert bytes to hex string for JSON serialization
                                    part_data['thought_signature'] = part.thought_signature.hex()

                                content_data['parts'].append(part_data)

                        candidate_data['content'] = content_data

                    # Safety ratings
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        for rating in candidate.safety_ratings:
                            candidate_data['safety_ratings'].append({
                                'category': getattr(rating, 'category', None),
                                'probability': getattr(rating, 'probability', None),
                            })

                    chunk_data['candidates'].append(candidate_data)

            # Extract usage metadata
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage = chunk.usage_metadata
                chunk_data['usage_metadata'] = {
                    'prompt_tokens': getattr(usage, 'prompt_token_count', None),
                    'completion_tokens': getattr(usage, 'candidates_token_count', None),
                    'total_tokens': getattr(usage, 'total_token_count', None),
                    'cache_read_input_tokens': getattr(usage, 'cached_content_token_count', None),
                }

            # Write to JSONL file
            with open(self._chunk_log_path, 'a') as f:
                f.write(json.dumps(chunk_data) + '\n')

        except Exception as e:
            logger.error(f"Failed to log chunk: {e}")

    async def _stream_content(
        self, params_from_context
    ) -> AsyncIterator[GenerateContentResponse]:
        """Stream content and capture candidate contents for thought processing."""
        try:
            base_stream = await super()._stream_content(params_from_context)
        except Exception:
            raise

        self._captured_candidate_contents.clear()
        # Increment turn number and reset chunk index
        self._turn_number += 1
        self._chunk_index = 0

        async def _capturing_stream() -> AsyncIterator[GenerateContentResponse]:
            async for chunk in base_stream:
                # Log this chunk
                self._log_chunk(chunk, self._chunk_index)
                self._chunk_index += 1
                # Check if chunk has function_calls with thought_signature
                if hasattr(chunk, 'function_calls') and chunk.function_calls:
                    logger.debug(f"=== CHUNK has function_calls! ===")
                    for i, fc in enumerate(chunk.function_calls):
                        logger.debug(f"Chunk function_call {i}: {fc}")
                        logger.debug(f"Chunk function_call {i} attributes: {dir(fc)}")
                        if hasattr(fc, 'thought_signature'):
                            logger.debug(f"Chunk function_call {i} HAS thought_signature: {fc.thought_signature}")

                candidates = getattr(chunk, "candidates", None)
                candidate = candidates[0] if candidates else None

                content = getattr(candidate, "content", None) if candidate else None

                if content is not None:
                    # Debug: Log raw content structure
                    logger.debug(f"=== RAW CHUNK from Gemini ===")
                    logger.debug(f"Content type: {type(content)}")
                    logger.debug(f"Content role: {getattr(content, 'role', 'N/A')}")
                    if hasattr(content, 'parts') and content.parts:
                        for i, part in enumerate(content.parts):
                            logger.debug(f"  Part {i} type: {type(part)}")
                            if hasattr(part, 'text'):
                                logger.debug(f"  Part {i} has text: {bool(part.text)}")
                                logger.debug(f"  Part {i} is thought: {getattr(part, 'thought', False)}")
                            if hasattr(part, 'function_call'):
                                fc = part.function_call
                                if fc:
                                    logger.debug(f"  Part {i} has function_call: {fc.name if hasattr(fc, 'name') else 'N/A'}")
                                    # Check for thought_signature at PART level (not function_call level!)
                                    if hasattr(part, 'thought_signature'):
                                        logger.debug(f"  Part {i} HAS thought_signature at Part level: {part.thought_signature[:50]}...")
                                    else:
                                        logger.debug(f"  Part {i} NO thought_signature at Part level")
                                    # Log Part-level attributes
                                    part_dict = vars(part) if hasattr(part, '__dict__') else {}
                                    logger.debug(f"  Part {i} attributes: {list(part_dict.keys())}")

                    self._captured_candidate_contents.append(content)

                yield chunk

        return _capturing_stream()

    async def push_frame(
        self,
        frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        """Emit thinking mode content frame when response ends."""
        if isinstance(frame, LLMFullResponseEndFrame):
            if self._captured_candidate_contents:
                # Don't use model_copy - just pass the original Content objects
                # They should already be complete with thought_signature
                contents_copy: List[Content] = list(self._captured_candidate_contents)

                # Debug: Check if thought_signature is present
                for i, content in enumerate(contents_copy):
                    for part in (content.parts or []):
                        if hasattr(part, 'function_call') and part.function_call:
                            has_ts = hasattr(part.function_call, 'thought_signature')
                            logger.debug(f"Content {i} function_call has thought_signature: {has_ts}")
                            if has_ts:
                                logger.debug(f"  thought_signature value: {part.function_call.thought_signature}")

                await super().push_frame(
                    _GeminiThinkingModeContentFrame(contents=contents_copy)
                )
                self._captured_candidate_contents.clear()
            else:
                logger.debug("LLMFullResponseEndFrame but no candidate contents")

        await super().push_frame(frame, direction)

    def _is_sanitized_function_message(self, message: Any) -> bool:
        """Check if message is a function response message."""
        if isinstance(message, LLMSpecificMessage):
            return False
        if isinstance(message, dict):
            parts = message.get("parts") or []
            if parts and any(
                isinstance(part, dict)
                and part.get("function_response") is not None
                for part in parts
            ):
                return True
            if message.get("role") == "tool":
                return True
        return False

    @staticmethod
    def _is_sanitized_function_call_only(message: Any) -> bool:
        """Check if message contains only function calls."""
        # Handle Content objects directly
        if hasattr(message, 'parts') and hasattr(message, 'role'):
            # This is a Content object
            parts = message.parts or []
            if not parts:
                return False
            # Check if all parts are function_call parts (no text, no function_response)
            return all(
                hasattr(part, 'function_call')
                and part.function_call is not None
                and (not hasattr(part, 'text') or not part.text)
                and (not hasattr(part, 'function_response') or not part.function_response)
                for part in parts
            )

        # Handle LLMSpecificMessage wrapping a Content object
        if isinstance(message, LLMSpecificMessage):
            inner_message = message.message
            if hasattr(inner_message, 'parts'):
                return GeminiThinkingModeLLMService._is_sanitized_function_call_only(inner_message)

        # Handle dict-based messages
        if not isinstance(message, dict):
            return False
        parts = message.get("parts") or []
        if not parts:
            tool_calls = message.get("tool_calls") or []
            if tool_calls and all(
                isinstance(call, dict)
                and call.get("function")
                and not message.get("content")
                for call in tool_calls
            ):
                return True
            return False
        return all(
            isinstance(part, dict)
            and part.get("function_call") is not None
            and not part.get("function_response")
            and not part.get("text")
            for part in parts
        )

    def _find_previous_function_call_name(
        self, messages: List[Any], start_index: int
    ) -> Optional[str]:
        """Find the function name from a previous function call message."""
        for cursor in range(start_index, -1, -1):
            candidate = messages[cursor]
            if isinstance(candidate, LLMSpecificMessage):
                candidate_parts = getattr(candidate.message, "parts", None) or []
                for part in reversed(candidate_parts):
                    function_call = getattr(part, "function_call", None)
                    if function_call and getattr(function_call, "name", None):
                        return getattr(function_call, "name")
            elif isinstance(candidate, dict):
                candidate_parts = candidate.get("parts") or []
                for part in reversed(candidate_parts):
                    if not isinstance(part, dict):
                        continue
                    function_call = part.get("function_call")
                    if function_call:
                        name = function_call.get("name")
                        if name:
                            return name
                tool_calls = candidate.get("tool_calls") or []
                for tool_call in reversed(tool_calls):
                    if not isinstance(tool_call, dict):
                        continue
                    function_payload = tool_call.get("function", {})
                    if not isinstance(function_payload, dict):
                        continue
                    name = function_payload.get("name")
                    if name:
                        return name
        return None

    def _create_function_response_message(
        self,
        name: Optional[str],
        response_payload: Any,
        extra_fields: Dict[str, Any],
    ) -> LLMSpecificMessage:
        """Create a properly formatted function response message for Gemini."""
        resolved_name = name or "tool_call_result"
        part = genai_types.Part.from_function_response(
            name=resolved_name,
            response=response_payload if response_payload is not None else {},
        )

        # Add extra fields like will_continue, scheduling, parts
        for field in ("will_continue", "scheduling", "parts"):
            value = extra_fields.get(field)
            if value is not None:
                try:
                    setattr(part.function_response, field, value)
                except AttributeError:
                    pass

        content = Content(role="user", parts=[part])
        adapter = self.get_llm_adapter()
        llm_id = adapter.id_for_llm_specific_messages if adapter else "google"
        return LLMSpecificMessage(llm=llm_id, message=content)

    def _convert_function_response_message(
        self,
        raw_message: Dict[str, Any],
        preceding_messages: List[Any],
    ) -> LLMSpecificMessage:
        """Convert a function response message to proper Gemini format."""
        response_dict: Dict[str, Any] = {}
        response_payload: Any = {}
        name: Optional[str] = None

        parts = raw_message.get("parts") or []
        if parts:
            first_part = parts[0]
            if isinstance(first_part, dict):
                response_dict = first_part.get("function_response", {}) or {}

        if response_dict:
            response_payload = response_dict.get("response", {})
            name = response_dict.get("name")
        else:
            raw_content = raw_message.get("content")
            if isinstance(raw_content, str):
                try:
                    response_payload = json.loads(raw_content)
                except json.JSONDecodeError:
                    response_payload = {"text": raw_content}
            elif raw_content is not None:
                response_payload = raw_content

        if not name:
            name = self._find_previous_function_call_name(
                preceding_messages, len(preceding_messages) - 1
            )

        return self._create_function_response_message(
            name,
            response_payload,
            response_dict,
        )

    @staticmethod
    def _is_function_response_only(message: Any) -> bool:
        """Check if message contains only function responses (no text or function calls)."""
        if hasattr(message, 'parts') and hasattr(message, 'role'):
            # Content object
            parts = message.parts or []
            if not parts or message.role != 'user':
                return False
            return all(
                hasattr(part, 'function_response')
                and part.function_response is not None
                and (not hasattr(part, 'text') or not part.text)
                and (not hasattr(part, 'function_call') or not part.function_call)
                for part in parts
            )
        elif isinstance(message, dict):
            # Dict format
            if message.get('role') != 'user':
                return False
            parts = message.get('parts', [])
            if not parts:
                return False
            return all(
                isinstance(p, dict)
                and 'function_response' in p
                and p.get('function_response') is not None
                and not p.get('text')
                and not p.get('function_call')
                for p in parts
            )
        return False

    def _remove_duplicate_function_call_messages(self, context: LLMContext) -> None:
        """Remove duplicate function call messages and deduplicate consecutive function responses."""
        messages = context.get_messages()
        normalized_messages: List[Any] = []
        changed = False
        idx = 0

        while idx < len(messages):
            message = messages[idx]

            # Skip function-call-only messages
            if self._is_sanitized_function_call_only(message):
                changed = True
                idx += 1
                continue

            # Skip pairs of LLMSpecificMessage + function-call-only
            next_index = idx + 1
            if (
                isinstance(message, LLMSpecificMessage)
                and next_index < len(messages)
                and self._is_sanitized_function_call_only(messages[next_index])
            ):
                normalized_messages.append(message)
                changed = True
                idx += 2
                continue

            # Deduplicate consecutive function_response entries
            # Keep only the FIRST function_response in a sequence
            if self._is_function_response_only(message):
                # Add this first function_response
                if self._is_sanitized_function_message(message):
                    normalized_messages.append(
                        self._convert_function_response_message(
                            message, normalized_messages
                        )
                    )
                else:
                    normalized_messages.append(message)

                # Skip ALL following consecutive function_response-only messages
                idx += 1
                while idx < len(messages) and self._is_function_response_only(messages[idx]):
                    logger.debug(f"Skipping duplicate function_response at index {idx}")
                    changed = True
                    idx += 1
                continue

            # Convert function response messages (that have other content too)
            if self._is_sanitized_function_message(message):
                normalized_messages.append(
                    self._convert_function_response_message(
                        message, normalized_messages
                    )
                )
                changed = True
                idx += 1
                continue

            normalized_messages.append(message)
            idx += 1

        if changed:
            context.set_messages(normalized_messages)

    def _safe_message_to_json(self, message: Any) -> Dict[str, Any]:
        """Safely convert a message to JSON, handling both Content objects and dicts."""
        if hasattr(message, 'to_json_dict'):
            return message.to_json_dict()
        elif isinstance(message, dict):
            # Already a dict, return as-is
            return message
        else:
            # Fallback for other types
            return {"type": str(type(message)), "value": str(message)[:100]}

    async def _process_context(self, context: Any):
        """Process context before generating content."""
        # Check if context has get_messages method (works for both LLMContext and GoogleLLMContext)
        if hasattr(context, 'get_messages') and callable(context.get_messages):
            logger.debug(f"[PRE-CLEANUP] Context has {len(context.get_messages())} messages")
            self._remove_duplicate_function_call_messages(context)
            logger.debug(f"[POST-CLEANUP] Context has {len(context.get_messages())} messages")

            # Log message types for debugging
            for i, msg in enumerate(context.get_messages()):
                msg_type = type(msg).__name__
                if isinstance(msg, LLMSpecificMessage):
                    has_fc = any(hasattr(p, 'function_call') and p.function_call for p in (msg.message.parts or []))
                    # Check if function_call has thought_signature
                    has_ts = False
                    if has_fc:
                        for p in (msg.message.parts or []):
                            if hasattr(p, 'function_call') and p.function_call:
                                has_ts = hasattr(p.function_call, 'thought_signature') and p.function_call.thought_signature
                                break
                    logger.debug(f"  [{i}] LLMSpecificMessage (has_fc: {has_fc}, has_thought_sig: {has_ts})")
                elif isinstance(msg, dict):
                    has_fc = bool(msg.get('parts') and any(p.get('function_call') for p in msg.get('parts', []) if isinstance(p, dict)))
                    logger.debug(f"  [{i}] dict (has_fc: {has_fc})")
                else:
                    logger.debug(f"  [{i}] {msg_type}")

            # Monkey-patch get_messages_for_logging to handle both dicts and Content objects
            original_get_messages = getattr(context, 'get_messages_for_logging', None)
            if original_get_messages:
                def safe_get_messages_for_logging():
                    try:
                        messages = context.get_messages()
                        return [self._safe_message_to_json(m) for m in messages]
                    except Exception:
                        return []

                # Temporarily replace the method
                context.get_messages_for_logging = safe_get_messages_for_logging

        result = await super()._process_context(context)
        return result


class GeminiThinkingModeTracker(FrameProcessor):
    """
    Frame processor that handles Gemini thinking mode output.

    This processor:
    1. Processes _GeminiThinkingModeContentFrame to extract and log thoughts
    2. Passes all frames downstream including LLMTextFrame for proper aggregation
    3. Logs thoughts separately for debugging (does not add them to context)
    """

    def __init__(self, llm_service: 'GeminiThinkingModeLLMService', *, log_thoughts: bool = True):
        super().__init__()
        self._llm_service = llm_service
        self._log_thoughts = log_thoughts

    async def process_frame(self, frame: Any, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, _GeminiThinkingModeContentFrame):
            # Log thoughts for debugging - we don't add them back to the context
            # The context only needs user messages, function calls, and function responses
            for content in frame.contents:
                for part in content.parts or []:
                    if part.text and part.thought and self._log_thoughts:
                        logger.debug(f"[THOUGHT]: {part.text}")

        # Pass all frames downstream, including LLMTextFrame
        # The assistant aggregator needs these to properly complete and emit timestamp frames
        await self.push_frame(frame, direction)


def create_gemini_thinking_mode_service(
    api_key: str,
    model: str,
    thinking_budget: int = 1,
    include_thoughts: bool = True,
    chunk_log_path: Optional[str] = None,
    **kwargs
) -> GeminiThinkingModeLLMService:
    """
    Create a Gemini LLM service with thinking mode support.

    Args:
        api_key: Google API key
        model: Model name (e.g., "models/gemini-3-pro-preview")
        thinking_budget: Thinking budget (1 = minimal, higher = more thinking)
        include_thoughts: Whether to include thoughts in the response
        chunk_log_path: Optional path to log all raw chunk output from Gemini API
        **kwargs: Additional arguments passed to GoogleLLMService

    Returns:
        Configured GeminiThinkingModeLLMService instance
    """
    params = GeminiThinkingModeLLMService.InputParams(
        extra={
            "thinking_config": {
                "thinking_budget": thinking_budget,
                "include_thoughts": include_thoughts,
            }
        }
    )

    return GeminiThinkingModeLLMService(
        api_key=api_key,
        model=model,
        params=params,
        chunk_log_path=chunk_log_path,
        **kwargs
    )
