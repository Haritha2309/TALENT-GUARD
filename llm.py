"""Optional Claude-powered copilot. Needs ANTHROPIC_API_KEY (Streamlit secrets or env var)."""
import os

DEFAULT_MODEL = "claude-sonnet-5-5"


def get_key(secrets=None):
    try:
        if secrets is not None and "ANTHROPIC_API_KEY" in secrets:
            return secrets["ANTHROPIC_API_KEY"]
    except Exception:  # no secrets.toml present
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def ask(api_key, system, history, model=DEFAULT_MODEL):
    """history: list of (role, text) ending with the latest user message."""
    import anthropic

    msgs = [{"role": r, "content": t} for r, t in history[-8:]]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(model=model, max_tokens=900, system=system, messages=msgs)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
