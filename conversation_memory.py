#conversation_memory.py
#让Agent记住当前终端最后几轮对话

class ConversationMemory:
    def __init__(self, max_turns: int=5):
        self.max_turns = max_turns
        self.history = []
        self.summary = ""

    def add_turn(self, user_input: str,assitant_response: str):
        turn = {
            "user": user_input,
            "assistant": assitant_response
        }
        self.history.append(turn)

        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_recent_history(self):
        return self.history

    def format_history(self) -> str:
        return self.format_recent_history(self.max_turns)

    def format_recent_history(self, max_turns: int = 2) -> str:
        if not self.history:
            return "No previous conversation in this session"

        recent_history = self.history[-max_turns:]

        formatted_turns = []

        for index, turn in enumerate(recent_history):
            formatted_turns.append(
                f"Turn {index}\n"
                f"User: {turn['user']}\n"
                f"Assistant: {turn['assistant']}"

            )

        return "\n\n".join(formatted_turns)

    def set_summary(self, summary: str):
        # v1.2.6 provides the storage interface. Automatic summarization will
        # be introduced later with an explicit update policy.
        self.summary = summary.strip()

    def get_summary(self) -> str:
        return self.summary

    def clear_history(self):
        self.history = []
        self.summary = ""
