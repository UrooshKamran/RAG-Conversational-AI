import os
import json
import requests
from cart_manager import CartManager

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
SUMMARIZER_MODEL = "qwen2.5:1.5b"

SUMMARIZER_SYSTEM = (
    "You are a conversation summarizer. "
    "Given a set of conversation turns, produce a concise factual summary "
    "(2-4 sentences) capturing: what the user asked for, any items discussed, "
    "decisions made, and current cart state if mentioned. "
    "Do NOT greet or explain yourself — output the summary only."
)


def _llm_summarize(existing_summary: str, new_turns: list) -> str:
    """
    Ask the LLM to merge `new_turns` into `existing_summary`.
    Returns the updated summary string.
    Falls back to a simple join on any network/parse error.
    """
    turns_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in new_turns
    )

    user_prompt = ""
    if existing_summary:
        user_prompt += f"[EXISTING SUMMARY]\n{existing_summary}\n\n"
    user_prompt += f"[NEW TURNS TO INCORPORATE]\n{turns_text}"

    payload = {
        "model": SUMMARIZER_MODEL,
        "messages": [
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {"num_predict": 200, "temperature": 0.3},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception:
        # Graceful fallback: plain concatenation — never crash the main chat
        fallback_lines = [
            f"{m['role'].capitalize()}: {m['content']}" for m in new_turns
        ]
        base = (existing_summary + "\n" + "\n".join(fallback_lines)).strip()
        return base


class MemoryManager:
    """
    Manages conversation history with a rolling-window + LLM-summary strategy.

    Strategy:
    - Keep the last MAX_TURNS messages in active memory.
    - When history exceeds MAX_TURNS, the oldest pair of messages is handed to
      an LLM which merges them into a running natural-language summary block.
    - The summary is prepended to every prompt so the assistant never loses
      context from earlier in the conversation.
    - Cart state is always injected fresh from CartManager (never truncated).
    """

    MAX_TURNS = 6  # Max messages in active window

    def __init__(self, cart: CartManager):
        self.cart = cart
        self.active_history = []
        self.summary_block = ""

    def add_message(self, role: str, content: str):
        """Append a new message to active history, trim if needed."""
        self.active_history.append({"role": role, "content": content})
        if len(self.active_history) > self.MAX_TURNS:
            self._trim()

    def _trim(self):
        """
        Pop the oldest pair of messages and let the LLM summarize them into
        summary_block.  Replaces the old hard-coded 200-character truncation.
        """
        to_summarize = self.active_history[:2]
        self.active_history = self.active_history[2:]

        # Delegate summarization to the LLM — no hard-coded char limits
        self.summary_block = _llm_summarize(self.summary_block, to_summarize)

    def build_messages(self, system_prompt: str) -> list:
        """
        Build the full messages list to send to Ollama.
        Order: system prompt + cart state + LLM summary block + active history.
        """
        system_content = system_prompt
        system_content += f"\n\n{self.cart.to_context_string()}"

        if self.summary_block:
            system_content += (
                "\n\n[EARLIER CONVERSATION SUMMARY]\n"
                "The following is an AI-generated summary of earlier parts of "
                "this conversation:\n"
                + self.summary_block
            )

        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.active_history)
        return messages

    def reset(self):
        """Clear all history and summary."""
        self.active_history = []
        self.summary_block = ""

    def get_turn_count(self) -> int:
        return len(self.active_history)
