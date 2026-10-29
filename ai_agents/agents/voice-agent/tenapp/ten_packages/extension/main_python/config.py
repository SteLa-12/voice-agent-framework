from pydantic import BaseModel


class MainControlConfig(BaseModel):
    greeting: str = "Hello, I am your AI assistant."
    system_prompt: str = ""
    interrupt_prompt: str = "The user's last message is incomplete; they are hesitating or searching for a word. Politely interrupt with a short completion or suggestion (one sentence), then let them continue."
