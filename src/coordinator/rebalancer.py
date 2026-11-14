"""
rebalancer.py - Chapter 15
Dynamic stage rebalancing. move() shifts ONE boundary layer between two neighbouring stages:
  phase 1 (system keeps running): the receiver loads the layer's weights          PrepareLayer
  phase 2 (all steps paused by the layout gate):
      the sender exports every request's K,V for that layer, the receiver imports  ExportKV / ImportKV
      the receiver commits (adds the layer), then the sender commits (drops it)    CommitLayout
      the pipeline switches to the new epoch, so calls under the old layout are fenced
If anything fails before the receiver commits, both sides abort and the old layout stays.
The monitor (start()) models each stage as  time = fixed + blocks x time-per-block,  measured from
recent decode steps, predicts the slowest stage after every possible one-layer move, and makes
the best move only if it lowers that bottleneck by more than `margin`.
"""
import statistics
import threading
import time
from dataclasses import dataclass

import grpc

from src.coordinator.pipeline import Pipeline
from src.proto import inference_pb2 as pb

RPC_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class MoveReport:
    layer: int
    src: int
    dst: int
    epoch: int
    prepare_ms: float              # loading weights, while serving continued
    pause_ms: float                # steps were blocked for this long
    kv_requests: int               # requests whose cache slice was migrated
    kv_bytes: int
    layout: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StageModel:
    """What the monitor knows about the stages, from recent decode steps (medians)."""
    stage_ms: tuple[float, ...]    # compute per step
    per_block_ms: tuple[float, ...]
    fixed_ms: tuple[float, ...]    # embeddings / output layer: cannot move

    @property
    def bottleneck_ms(self) -> float:
        return max(self.stage_ms)


class Rebalancer:
    def __init__(self, pipe: Pipeline, n_layer: int = 6, supervisor=None, period_s: float = 1.0,
                 window_s: float = 3.0, margin: float = 0.05, cooldown_s: float = 3.0, min_samples: int = 15,
                 undo_margin: float = 0.10) -> None:
        self.pipe, self.n_layer, self.supervisor = pipe, n_layer, supervisor
        self.period_s, self.window_s, self.margin = period_s, window_s, margin
        self.undo_margin = undo_margin                             # Ch 18: higher bar for reversing the last move
        self.cooldown_s, self.min_samples = cooldown_s, min_samples
        infos = pipe.check_layout(n_layer)
        epochs = {i.epoch for i in infos}
        if len(epochs) != 1:
            raise RuntimeError(f"workers disagree about the layout epoch: {sorted(epochs)}")
        self.epoch = epochs.pop()
        self.layout = [(i.first_layer, i.last_layer) for i in infos]
        pipe.epoch = self.epoch                                    # every call now carries the epoch
        self.moves: list[MoveReport] = []
        self.events: list[tuple[float, str]] = []
        self.move_lock = threading.Lock()                          # one layout change at a time
        self.last_change = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    # ---------------- the protocol ----------------

    def move(self, src: int, dst: int) -> MoveReport:
        """Move the boundary layer between neighbouring stages src and dst from src to dst."""
        with self.move_lock:
            if abs(src - dst) != 1:
                raise ValueError("layers can only move between neighbouring stages")
            (sf, sl), (df, dl) = self.layout[src], self.layout[dst]
            if dst == src + 1:
                layer, new_src, new_dst = sl - 1, (sf, sl - 1), (sl - 1, dl)
            else:
                layer, new_src, new_dst = sf, (sf + 1, sl), (df, sf + 1)
            if new_src[0] >= new_src[1]:
                raise ValueError(f"stage {src} would be left with no layers")
            epoch = self.epoch + 1
            sender, receiver = self.pipe.stubs[src], self.pipe.stubs[dst]

            start = time.perf_counter()                            # phase 1: no pause
            try:
                receiver.PrepareLayer(pb.PrepareLayerRequest(epoch=epoch, layer=layer), timeout=RPC_TIMEOUT_S)
            except grpc.RpcError:
                self._abort(receiver)
                raise
            prepare_ms = (time.perf_counter() - start) * 1000

            with self.pipe.gate.exclusive():                       # phase 2: every step waits
                pause_start = time.perf_counter()
                try:
                    exported = sender.ExportKV(pb.ExportKVRequest(layer=layer), timeout=RPC_TIMEOUT_S)
                    receiver.ImportKV(pb.ImportKVRequest(epoch=epoch, layer=layer, slices=exported.slices),
                                      timeout=RPC_TIMEOUT_S)
                    receiver.CommitLayout(pb.CommitLayoutRequest(epoch=epoch, first_layer=new_dst[0],
                                                                 last_layer=new_dst[1]), timeout=RPC_TIMEOUT_S)
                except grpc.RpcError:
                    self._abort(receiver)                          # nothing changed anywhere: old layout stays
                    raise
                # the receiver has the layer now; from here on we must finish, not abort
                for k, stub in enumerate(self.pipe.stubs):
                    if k == dst:
                        continue
                    first, last = new_src if k == src else self.layout[k]
                    self._commit_with_retries(stub, epoch, first, last)
                self.layout[src], self.layout[dst] = new_src, new_dst
                self.epoch = self.pipe.epoch = epoch
                pause_ms = (time.perf_counter() - pause_start) * 1000

            if self.supervisor is not None:                        # a restarted worker must come back like this
                for k, (first, last) in enumerate(self.layout):
                    self.supervisor.update_spec(k, first=first, last=last, epoch=epoch)
            kv_bytes = sum(len(s.k.data) + len(s.v.data) for s in exported.slices)
            report = MoveReport(layer, src, dst, epoch, prepare_ms, pause_ms, len(exported.slices), kv_bytes,
                                tuple(self.layout))
            self.moves.append(report)
            self.last_change = time.monotonic()
            self._event(f"moved layer {layer} from stage {src} to stage {dst}: layout {list(self.layout)}, "
                        f"epoch {epoch}, paused {pause_ms:.0f} ms, {len(exported.slices)} KV slices")
            return report

    @staticmethod
    def _abort(stub) -> None:
        try:
            stub.AbortLayout(pb.AbortLayoutRequest(), timeout=RPC_TIMEOUT_S)
        except grpc.RpcError:
            pass                                                   # a dead worker has nothing prepared

    @staticmethod
    def _commit_with_retries(stub, epoch: int, first: int, last: int) -> None:
        for attempt in range(3):
            try:
                stub.CommitLayout(pb.CommitLayoutRequest(epoch=epoch, first_layer=first, last_layer=last),
                                  timeout=RPC_TIMEOUT_S)
                return
            except grpc.RpcError:
                if attempt == 2:
                    raise
                time.sleep(0.2)

    # ---------------- the monitor ----------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def model(self) -> StageModel | None:
        """Medians over the decode steps since the last change (and within the window)."""
        now = time.monotonic()
        samples = self.pipe.recent(max(now - self.window_s, self.last_change + 0.5))
        if len(samples) < self.min_samples:
            return None
        n_stages = len(self.layout)
        stage = tuple(statistics.median(s[2][k] for s in samples) for k in range(n_stages))
        blocks = tuple(statistics.median(s[3][k] for s in samples) for k in range(n_stages))
        per_block = tuple(blocks[k] / (last - first) for k, (first, last) in enumerate(self.layout))
        fixed = tuple(max(0.0, stage[k] - blocks[k]) for k in range(n_stages))
        return StageModel(stage, per_block, fixed)

    def best_move(self, m: StageModel) -> tuple[float, int, int] | None:
        """(predicted bottleneck ms, src, dst) of the best one-layer move, if it beats `margin`."""
        best = None
        sizes = [last - first for first, last in self.layout]
        for src in range(len(self.layout)):
            for dst in (src - 1, src + 1):
                if not 0 <= dst < len(self.layout) or sizes[src] <= 1:
                    continue
                after = list(m.stage_ms)
                after[src] = m.fixed_ms[src] + (sizes[src] - 1) * m.per_block_ms[src]
                after[dst] = m.fixed_ms[dst] + (sizes[dst] + 1) * m.per_block_ms[dst]   # at dst's speed
                margin = self.undo_margin if self._undoes_last_move(src, dst) else self.margin
                if max(after) < m.bottleneck_ms * (1 - margin) and (best is None or max(after) < best[0]):
                    best = (max(after), src, dst)
        return best

    def _undoes_last_move(self, src: int, dst: int) -> bool:
        """Ch 18: hysteresis. Two moves that each predict ~5-7% are inside the noise; without a higher bar
        for going BACK, the demo rehearsal moved layer 4 back and forth four times in 12 s."""
        return bool(self.moves) and (self.moves[-1].src, self.moves[-1].dst) == (dst, src)

    def _event(self, text: str) -> None:
        self.events.append((time.monotonic(), text))
        print(f"    [rebalancer] {text}", flush=True)

    def _watch(self) -> None:
        while not self._stop.wait(self.period_s):
            if time.monotonic() < self.last_change + self.cooldown_s:
                continue
            m = self.model()
            if m is None:
                continue
            best = self.best_move(m)
            if best is None:
                continue
            predicted, src, dst = best
            self._event(f"stage ms {[round(x, 1) for x in m.stage_ms]}, per block "
                        f"{[round(x, 1) for x in m.per_block_ms]}: moving a layer {src} -> {dst} "
                        f"should cut the bottleneck {m.bottleneck_ms:.1f} -> {predicted:.1f} ms")
            try:
                self.move(src, dst)
            except (grpc.RpcError, ValueError) as e:
                self._event(f"move failed, layout unchanged: {e}")
                self.last_change = time.monotonic()
