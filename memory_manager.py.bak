from cart_manager import CartManager


class MemoryManager:
    """
    Manages conversation history with a rolling window strategy.

    Strategy:
    - Keep the last MAX_TURNS messages in active memory
    - When history exceeds MAX_TURNS, older turns are summarized into a compact
      memory block prepended to the active window
    - Cart state is always injected fresh from CartManager (never truncated)
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
        """Pop oldest pair of messages, compress into summary_block."""
        to_summarize = self.active_history[:2]
        self.active_history = self.active_history[2:]

        new_lines = []
        for msg in to_summarize:
            role = msg["role"].capitalize()
            content = msg["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            new_lines.append(f"{role}: {content}")

        new_summary = "\n".join(new_lines)
        self.summary_block = (self.summary_block + "\n" + new_summary).strip() if self.summary_block else new_summary

    def build_messages(self, system_prompt: str) -> list:
        """
        Build the full messages list to send to Ollama.
        Order: system prompt + cart state + summary block + active history.
        """
        system_content = system_prompt
        system_content += f"\n\n{self.cart.to_context_string()}"

        if self.summary_block:
            system_content += (
                "\n\n[EARLIER CONVERSATION SUMMARY]\n"
                "The following is a summary of earlier parts of this conversation:\n"
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
