"""通用等待工具。"""

import time

from langchain_core.tools import tool
from pydantic import BaseModel, Field

MAX_SLEEP_SECONDS = 300.0


class SleepInput(BaseModel):
    seconds: float = Field(
        description="How many seconds to wait before continuing. Must be > 0 and <= 300.",
    )
    reason: str = Field(
        default="",
        description="Optional short reason for the wait.",
    )


@tool("sleep", args_schema=SleepInput)
def sleep(seconds: float, reason: str = "") -> str:
    """Block the current agent turn briefly while async work continues elsewhere."""
    if seconds <= 0 or seconds > MAX_SLEEP_SECONDS:
        return f"Error: seconds must be > 0 and <= {int(MAX_SLEEP_SECONDS)}"
    time.sleep(seconds)
    normalized_reason = str(reason or "").strip()
    if normalized_reason:
        return f"Slept for {seconds} seconds. reason={normalized_reason}"
    return f"Slept for {seconds} seconds."
