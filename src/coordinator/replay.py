"""
replay.py - Chapter 13
Generation that survives worker restarts. Every token is committed to the TokenLog before it is
used. If a worker dies or forgets the request (new generation, empty cache), the request is
REPLAYED: one prefill of prompt + all committed tokens but the last rebuilds every worker's
KV-cache, then decoding continues from where the log says it stopped.
"""
import math
import time
from dataclasses import dataclass

import grpc

from src.coordinator.pipeline import N_CTX, DeadlineExceeded, Pipeline, StaleReply
from src.coordinator.token_log import TokenLog

RECOVERABLE = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.FAILED_PRECONDITION}


@dataclass
class LoggedResult:
    new_ids: list[int]
    replays: int = 0              # times the request was rebuilt from the log
    replayed_tokens: int = 0      # positions re-processed by those replays
    mismatches: int = 0           # replays whose prediction disagreed with the log (the log wins)


def generate_logged(pipe: Pipeline, log: TokenLog, request_id: str, prompt_ids: list[int], max_new_tokens: int,
                    eos_id: int | None = None, deadline: float = math.inf, health=None,
                    max_replays: int = 3) -> LoggedResult:
    """Like pipeline.generate, but logged and recoverable. If the log already holds part of this request
    (e.g. the coordinator restarted), it is resumed, not started again."""
    if not log.has(request_id):
        log.start(request_id, prompt_ids, max_new_tokens)
    result = LoggedResult([])
    attempt = 0
    while not _finished(log, request_id, eos_id):
        wid = f"{request_id}~{attempt}"            # a fresh name on the workers for every (re)play
        try:
            _run(pipe, log, wid, request_id, eos_id, deadline, result)
        except (grpc.RpcError, StaleReply) as e:
            recoverable = isinstance(e, StaleReply) or e.code() in RECOVERABLE   # Ch 14: stale = recoverable
            if not recoverable or result.replays >= max_replays:
                raise
            if health is not None and not health.wait_all_up(deadline - time.monotonic()):
                raise DeadlineExceeded(f"{request_id}: workers not back before the deadline") from e
            try:                    # Ch 18 fix: the supervisor's channel sees the new worker first; OUR channel
                pipe.wait_ready(min(5.0, max(0.0, deadline - time.monotonic())))   # may still be in reconnect
            except RuntimeError:    # backoff and would fail fast, burning a replay. Not ready in time: try anyway.
                pass
            result.replays += 1
        finally:
            pipe.release(wid)
        attempt += 1
    log.end(request_id, "completed")
    result.new_ids = log.get(request_id).tokens
    return result


def _finished(log: TokenLog, rid: str, eos_id: int | None) -> bool:
    rec = log.get(rid)
    return (len(rec.tokens) >= rec.max_new
            or (bool(rec.tokens) and rec.tokens[-1] == eos_id)
            or len(rec.prompt) + len(rec.tokens) >= N_CTX)


def _run(pipe: Pipeline, log: TokenLog, wid: str, rid: str, eos_id: int | None, deadline: float,
         result: LoggedResult) -> None:
    """Rebuild the caches from the log (one prefill), then decode until finished. Raises on worker errors."""
    rec = log.get(rid)
    prefix = rec.prompt + rec.tokens[:-1]          # everything the caches must hold before the next step
    tok, _ = pipe.step(wid, 0, prefix, 0, deadline)
    if rec.tokens:                                 # resuming: this step re-predicts the last logged token
        result.replayed_tokens += len(prefix)
        if tok != rec.tokens[-1]:
            result.mismatches += 1
        tok = rec.tokens[-1]                       # committed tokens never change: the log wins
    else:
        log.append_token(rid, tok)
    start_pos, step = len(prefix), 1
    while not _finished(log, rid, eos_id):
        tok, _ = pipe.step(wid, step, [tok], start_pos, deadline)
        log.append_token(rid, tok)                 # commit first ...
        start_pos, step = start_pos + 1, step + 1  # ... then move on
