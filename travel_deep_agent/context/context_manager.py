"""
context/context_manager.py
----------------------------
This is SHORT-TERM / WORKING memory for a single trip-planning session.

Every subagent's output gets appended here so the *next* subagent can see
what already happened (e.g. the Itinerary agent can see what flights and
hotels were found). Left unchecked this history grows forever and blows the
model's context window, so once it passes CHAR_BUDGET we ask Gemini to
compress the oldest turns into one short summary paragraph and keep only the
most recent turns verbatim. This is "context management".
"""
from dataclasses import dataclass, field


@dataclass
class ContextManager:
    char_budget: int = 6000        # roughly ~1500 tokens, kept small for the demo
    keep_recent: int = 3           # always keep this many turns verbatim
    messages: list = field(default_factory=list)   # [{"role": .., "content": ..}]
    running_summary: str = ""

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def _total_chars(self) -> int:
        return sum(len(m["content"]) for m in self.messages) + len(self.running_summary)

    def get_context(self, model=None) -> str:
        """Return the context as one string, compressing older turns if needed."""
        if self._total_chars() > self.char_budget and len(self.messages) > self.keep_recent:
            self._compress(model)

        parts = []
        if self.running_summary:
            parts.append(f"[Summary of earlier steps]\n{self.running_summary}")
        for m in self.messages:
            parts.append(f"[{m['role']}] {m['content']}")
        return "\n\n".join(parts)

    def _compress(self, model) -> None:
        """Fold everything except the last `keep_recent` messages into
        `running_summary`. Falls back to plain truncation if no model is
        available (e.g. offline unit tests)."""
        old = self.messages[:-self.keep_recent]
        self.messages = self.messages[-self.keep_recent:]
        old_text = "\n".join(f"[{m['role']}] {m['content']}" for m in old)

        if model is None:
            # Cheap fallback: just keep the tail of the raw text.
            self.running_summary = (self.running_summary + "\n" + old_text)[-1500:]
            return

        prompt = (
            "Summarize the following trip-planning notes in 4-6 bullet points, "
            "keeping only concrete facts (prices, dates, names, decisions). "
            "Be terse.\n\nExisting summary:\n" + self.running_summary +
            "\n\nNew notes to fold in:\n" + old_text
        )
        response = model.generate_content(prompt)
        self.running_summary = response.text.strip()
