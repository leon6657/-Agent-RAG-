"""Simple sliding-window conversation memory."""

from typing import List, Tuple


class SimpleMemory:
    """Stores conversation history with a sliding window."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.messages: List[Tuple[str, str]] = []

    def add_user(self, message: str):
        self.messages.append(("user", message))
        self._trim()

    def add_assistant(self, message: str):
        self.messages.append(("assistant", message))
        self._trim()

    def _trim(self):
        if len(self.messages) > self.window_size * 2:
            self.messages = self.messages[-(self.window_size * 2):]

    def get_history(self) -> str:
        lines = []
        for role, content in self.messages:
            prefix = "Human: " if role == "user" else "Assistant: "
            lines.append(f"{prefix}{content}")
        return "\n".join(lines)

    def clear(self):
        self.messages = []
