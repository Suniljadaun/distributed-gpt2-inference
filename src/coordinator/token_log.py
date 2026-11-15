"""
token_log.py - Chapter 13
An append-only, crash-safe log of every request's progress: one line per record,
    <json>\t<crc32 of the json, 8 hex digits>\n
Records:  {"lsn": n, "type": "start", "rid": ..., "prompt": [...], "max_new": k}
          {"lsn": n, "type": "token", "rid": ..., "i": index, "tok": id, "epoch": e}
          {"lsn": n, "type": "end",   "rid": ..., "status": "completed"}
          {"lsn": n, "type": "layout", "epoch": e, "ranges": [[0, 3], [3, 6]]}
"epoch" on a token is the layout epoch the step ran under (0 = no rebalancer: the layout never changes).
A layout record is written at start-up and after every layer move, so the log says which layout produced
every token, and a restarted coordinator can start its workers in the last committed layout (latest_layout).
A token is COMMITTED once its record is in the file. On open, the file is read back to rebuild an
in-memory index; a torn last line (crash mid-write) is dropped, damage anywhere else is an error.
"""
import json
import os
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path


class LogCorrupted(Exception):
    """A record in the middle of the log is damaged: the log cannot be trusted."""


@dataclass
class RequestRecord:
    prompt: list[int]
    max_new: int
    tokens: list[int] = field(default_factory=list)
    status: str | None = None                 # None = still running (or interrupted)
    epochs: list[int] = field(default_factory=list)   # layout epoch of each token


class TokenLog:
    def __init__(self, path: str | Path, sync: str = "flush") -> None:
        """sync = "flush": hand each record to the OS (survives a process crash)
                  "fsync": also force it to disk (survives a power cut, much slower)"""
        if sync not in ("flush", "fsync"):
            raise ValueError("sync must be 'flush' or 'fsync'")
        self.path = Path(path)
        self.sync = sync
        self.lock = threading.Lock()
        self.requests: dict[str, RequestRecord] = {}
        self.next_lsn = 0                     # log sequence number of the next record
        self.dropped_torn_tail = False
        self.latest_layout: tuple[int, list[tuple[int, int]]] | None = None   # (epoch, ranges) last logged
        self._recover()
        self.file = open(self.path, "a", encoding="utf-8", newline="\n")

    # ---------------- writing ----------------

    def start(self, rid: str, prompt: list[int], max_new: int) -> None:
        with self.lock:
            if rid in self.requests:
                raise ValueError(f"request {rid!r} is already in the log")
            self.requests[rid] = RequestRecord(list(prompt), max_new)
            self._write({"type": "start", "rid": rid, "prompt": list(prompt), "max_new": max_new})

    def append_token(self, rid: str, tok: int, epoch: int = 0) -> int:
        """Commit the request's next token, computed under layout `epoch`. Returns its index."""
        with self.lock:
            rec = self.requests[rid]
            if rec.status is not None:
                raise ValueError(f"request {rid!r} already ended")
            i = len(rec.tokens)
            self._write({"type": "token", "rid": rid, "i": i, "tok": int(tok), "epoch": int(epoch)})
            rec.tokens.append(int(tok))       # in memory only AFTER it is in the file
            rec.epochs.append(int(epoch))
            return i

    def layout(self, epoch: int, ranges) -> None:
        """Record that layout `epoch` (one (first, last) layer range per stage) is now committed.
        Not written again if it is already the latest one (e.g. a coordinator restart in the same layout)."""
        ranges = [(int(a), int(b)) for a, b in ranges]
        with self.lock:
            if self.latest_layout == (epoch, ranges):
                return
            if self.latest_layout is not None and epoch < self.latest_layout[0]:
                raise ValueError(f"layout epoch {epoch} is older than the logged epoch {self.latest_layout[0]}")
            self._write({"type": "layout", "epoch": int(epoch), "ranges": [list(r) for r in ranges]})
            self.latest_layout = (int(epoch), ranges)

    def end(self, rid: str, status: str) -> None:
        with self.lock:
            self._write({"type": "end", "rid": rid, "status": status})
            self.requests[rid].status = status

    def close(self) -> None:
        with self.lock:
            self.file.close()

    def _write(self, record: dict) -> None:
        record = {"lsn": self.next_lsn, **record}
        body = json.dumps(record, separators=(",", ":"))
        self.file.write(f"{body}\t{zlib.crc32(body.encode()):08x}\n")
        self.file.flush()                                  # Python's buffer -> the OS
        if self.sync == "fsync":
            os.fsync(self.file.fileno())                   # the OS's cache -> the disk
        self.next_lsn += 1

    # ---------------- reading ----------------

    def has(self, rid: str) -> bool:
        with self.lock:
            return rid in self.requests

    def get(self, rid: str) -> RequestRecord:
        with self.lock:
            rec = self.requests[rid]
            return RequestRecord(list(rec.prompt), rec.max_new, list(rec.tokens), rec.status, list(rec.epochs))

    def incomplete(self) -> list[str]:
        """Requests that started but never ended: what a restarted coordinator must resume."""
        with self.lock:
            return [rid for rid, rec in self.requests.items() if rec.status is None]

    def _recover(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return
        with open(self.path, "rb") as f:
            raw_lines = f.read().split(b"\n")
        complete, tail = raw_lines[:-1], raw_lines[-1]     # the part after the last \n was never finished
        good_bytes = 0
        for n, raw in enumerate(complete):
            record = self._parse(raw)
            if record is None:
                if n == len(complete) - 1 and not tail:    # damaged LAST line: a torn write, drop it
                    self.dropped_torn_tail = True
                    break
                raise LogCorrupted(f"{self.path}: record {n} is damaged")
            self._apply(record)
            good_bytes += len(raw) + 1
        if tail:
            self.dropped_torn_tail = True                  # bytes without a newline: never committed
        if self.dropped_torn_tail:
            with open(self.path, "r+b") as f:              # cut the torn bytes off, so appends start clean
                f.truncate(good_bytes)

    @staticmethod
    def _parse(raw: bytes) -> dict | None:
        try:
            body, crc = raw.decode("utf-8").rsplit("\t", 1)
            if int(crc, 16) != zlib.crc32(body.encode()):
                return None
            return json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None

    def _apply(self, r: dict) -> None:
        if r["lsn"] != self.next_lsn:
            raise LogCorrupted(f"{self.path}: expected lsn {self.next_lsn}, found {r['lsn']}")
        if r["type"] == "start":
            self.requests[r["rid"]] = RequestRecord(r["prompt"], r["max_new"])
        elif r["type"] == "token":
            rec = self.requests[r["rid"]]
            if r["i"] != len(rec.tokens):
                raise LogCorrupted(f"{self.path}: token {r['i']} of {r['rid']!r} out of order")
            rec.tokens.append(r["tok"])
            rec.epochs.append(r.get("epoch", 0))               # logs written before epochs were recorded: 0
        elif r["type"] == "end":
            self.requests[r["rid"]].status = r["status"]
        elif r["type"] == "layout":
            self.latest_layout = (r["epoch"], [tuple(x) for x in r["ranges"]])
        self.next_lsn += 1
