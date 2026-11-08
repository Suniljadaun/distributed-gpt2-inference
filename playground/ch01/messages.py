"""
messages.py - Chapter 1, demo 1
It's for checking how we describe the messages our system passes around,
with the help of dataclasses and type hints.
Run:  python playground/ch01/messages.py
"""

from dataclasses import dataclass, field, replace

import time


@dataclass
class Request:
    """
    A user's generation request, as the coordinator stores it. Mutable: it
    changes as tokens arrive."""

    request_id: int
    prompt: str
    max_new_tokens: int = 5
    generated: list[str] = field(default_factory=list)  # each request gets its own empty list
    created_at: float = field(default_factory=time.perf_counter)

    def is_done(self) -> bool:
        return len(self.generated) >= self.max_new_tokens


@dataclass(frozen=True)
class StepMessage:
    """
    One unit of work sent to a pipeline stage. frozen=True means:
    read-only once created."""

    request_id: int
    step_id: int  # 0, 1, 2, ... increases by one per generated token
    payload: str


def main() -> None:
    r1 = Request(request_id=1, prompt="Distributed systems are")
    r2 = Request(request_id=2, prompt="The KV-cache", max_new_tokens=2)
    print("r1 =", r1)

    r1.generated.append("hard")
    print("r1.generated =", r1.generated, "| r2.generated =", r2.generated, "(separate lists)")
    print("r2 done?", r2.is_done())

    msg = StepMessage(request_id=1, step_id=0, payload="hidden-state-bytes")
    print("msg =", msg)
    try:
        msg.step_id = 99  # type: ignore[misc]
    except Exception as e:
        print("cannot modify a frozen message:", type(e).__name__)

    next_msg = replace(msg, step_id=msg.step_id + 1)  # make a NEW message instead
    print("next =", next_msg)
    print("equal by value?", msg == StepMessage(1, 0, "hidden-state-bytes"))


if __name__ == "__main__":
    main()
