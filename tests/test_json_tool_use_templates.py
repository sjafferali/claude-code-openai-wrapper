"""Tests for the tool-use-aware JSON instruction templates.

These verify the shape and content of the new templates that resolve the
JSON-mode + tool-use conflict — they must explicitly permit tool calls
before the final JSON.
"""

from __future__ import annotations

from src.message_adapter import MessageAdapter


class TestJSONModeInstructionWithTools:
    """The instruction must allow tool calls before the JSON."""

    def test_existing_strict_instruction_unchanged(self):
        """The original strict instruction must not be touched."""
        text = MessageAdapter.JSON_MODE_INSTRUCTION
        assert "ENTIRE response must be valid JSON" in text
        assert "first keystroke must be { or [" in text

    def test_with_tools_explicitly_allows_tool_calls(self):
        text = MessageAdapter.JSON_MODE_INSTRUCTION_WITH_TOOLS
        # Must permit calling tools
        assert "MAY" in text or "may" in text
        assert "call tools" in text.lower()
        # And clarify that intermediate calls don't violate the JSON rule
        assert "final message" in text.lower()
        # Old strict "ENTIRE response" wording must be GONE in the
        # with-tools variant — otherwise the conflict persists
        assert "ENTIRE response must be valid JSON" not in text

    def test_with_tools_still_constrains_final_message(self):
        text = MessageAdapter.JSON_MODE_INSTRUCTION_WITH_TOOLS
        # Final-message brackets still required
        assert "{ or [" in text
        assert "} or ]" in text
        # Markdown still forbidden in final message
        assert "code fence" in text.lower() or "markdown" in text.lower()


class TestJSONPromptSuffixWithTools:
    def test_existing_strict_suffix_unchanged(self):
        text = MessageAdapter.JSON_PROMPT_SUFFIX
        assert "RAW JSON ONLY" in text

    def test_with_tools_mentions_tool_calls(self):
        text = MessageAdapter.JSON_PROMPT_SUFFIX_WITH_TOOLS
        assert "tool" in text.lower()
        assert "final message" in text.lower()


class TestJSONSchemaTemplateWithTools:
    def test_existing_strict_template_unchanged(self):
        text = MessageAdapter.JSON_SCHEMA_TEMPLATE
        assert "{schema_json}" in text
        assert "Do not include any text before or after the JSON" in text

    def test_with_tools_template_permits_tool_calls(self):
        # Normalize whitespace so line breaks in the template don't trip assertions
        flat = " ".join(MessageAdapter.JSON_SCHEMA_TEMPLATE_WITH_TOOLS.split())
        # Schema placeholder preserved
        assert "{schema_json}" in flat
        # Tool calls explicitly allowed
        assert "Tool calls during your turn" in flat
        # "Final message" framing is what makes this compatible with tools
        assert "final message" in flat.lower()

    def test_with_tools_template_formats_with_schema_json(self):
        """Sanity: schema substitution works on the new template."""
        out = MessageAdapter.JSON_SCHEMA_TEMPLATE_WITH_TOOLS.format(
            schema_json='{"type": "object"}'
        )
        assert '{"type": "object"}' in out
        assert "Tool calls" in out
