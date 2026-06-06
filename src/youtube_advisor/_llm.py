"""Thin Anthropic SDK wrapper used by gen_skill_md + gen_benchmark."""
from __future__ import annotations

import os

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def should_skip_llm(no_llm: bool | None) -> bool:
    """Resolve effective LLM-skip flag.

    Explicit ``no_llm`` (True/False) wins. When None, auto-detect from
    ANTHROPIC_API_KEY: skip LLM when the key is absent.
    """
    if no_llm is not None:
        return bool(no_llm)
    return not os.environ.get("ANTHROPIC_API_KEY")


def call_tool(prompt: str, *, tool_name: str, schema: dict,
              max_tokens: int = 2000, model: str = DEFAULT_MODEL) -> dict:
    """Call Claude with a forced tool_use and return the tool's input dict."""
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        tools=[{
            "name": tool_name,
            "description": f"Return {tool_name} payload",
            "input_schema": schema,
        }],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("No tool_use block in LLM response")
