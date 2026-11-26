"""
cluster.py - Chapter 8 (Ch 12: generation numbers, supervised_cluster)
Start the worker processes for a layout, wait until they are ready, check the layout,
hand back a Pipeline, and always stop the workers afterwards.
Ch 12: supervised_cluster does the same through a Supervisor that also restarts dead workers.
"""
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from src.coordinator.pipeline import Pipeline
from src.model.config import config_for


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    port: int
    first: int
    last: int
    embed: bool = False
    head: bool = False
    extra_args: tuple[str, ...] = ()       # Ch 10: e.g. ("--fail-every", "3") for fault injection
    first_gen_args: tuple[str, ...] = ()   # Ch 12: only for generation 1, e.g. ("--hang-after-s", "3")
    epoch: int = 1                         # Ch 15: layout epoch (the rebalancer updates first/last/epoch)
    model: str = "distilgpt2"              # Ch 18: or "gpt2" (12 layers)

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.port}"

    def command(self, generation: int = 1) -> list[str]:
        cmd = [sys.executable, "-m", "src.worker.server", "--worker-id", self.worker_id,
               "--port", str(self.port), "--first", str(self.first), "--last", str(self.last),
               "--generation", str(generation), "--epoch", str(self.epoch), "--model", self.model]
        cmd += (["--embed"] if self.embed else []) + (["--head"] if self.head else [])
        cmd += list(self.extra_args)
        return cmd + (list(self.first_gen_args) if generation == 1 else [])   # a restart is a clean worker


def two_stage_layout(base_port: int = 50061, split: int = 3, n_layer: int = 6) -> list[WorkerSpec]:
    return [
        WorkerSpec("A", base_port, 0, split, embed=True),
        WorkerSpec("B", base_port + 1, split, n_layer, head=True),
    ]


def n_stage_layout(n_stages: int, base_port: int = 50061, model: str = "distilgpt2") -> list[WorkerSpec]:
    """Ch 18: n workers A, B, C, ... with the layers split as evenly as possible (distilgpt2, 3 stages:
    0-1 | 2-3 | 4-5; gpt2: 0-3 | 4-7 | 8-11). Nothing else depends on the number of stages: Pipeline.step,
    check_layout, the supervisor and the rebalancer all loop over the stages."""
    n_layer = config_for(model).n_layer
    if not 1 <= n_stages <= n_layer:
        raise ValueError(f"need 1..{n_layer} stages, got {n_stages}")
    bounds = [k * n_layer // n_stages for k in range(n_stages + 1)]
    return layout_from_ranges(list(zip(bounds, bounds[1:])), base_port=base_port, model=model)


def layout_from_ranges(ranges, epoch: int = 1, base_port: int = 50061, model: str = "distilgpt2") -> list[WorkerSpec]:
    """Workers A, B, ... for the given (first, last) layer ranges, e.g. the latest layout in the token log."""
    return [WorkerSpec(chr(ord("A") + k), base_port + k, first, last, embed=k == 0, head=k == len(ranges) - 1,
                       epoch=epoch, model=model) for k, (first, last) in enumerate(ranges)]


@contextmanager
def running_cluster(layout: list[WorkerSpec], ready_timeout_s: float = 60.0, **pipeline_options) -> Iterator[Pipeline]:
    procs = [subprocess.Popen(spec.command()) for spec in layout]
    pipe = Pipeline([spec.address for spec in layout], **pipeline_options)
    try:
        pipe.wait_ready(ready_timeout_s)
        pipe.check_layout(n_layer=layout[-1].last)
        yield pipe
    finally:
        pipe.close()
        for p in procs:
            p.kill()
        for p in procs:
            p.wait()


@contextmanager
def supervised_cluster(layout: list[WorkerSpec], **options) -> Iterator[tuple[Pipeline, "Supervisor"]]:
    """Ch 12: like running_cluster, but a Supervisor owns the processes and restarts dead workers.
    `options` go to the Supervisor (interval_s, misses_allowed, ping_timeout_s, ...)."""
    from src.coordinator.supervisor import Supervisor       # imported here: supervisor.py imports this module
    supervisor = Supervisor(layout, **options)
    supervisor.start()
    pipe = Pipeline([spec.address for spec in layout], timeout_s=2.0)
    for i, s in enumerate(supervisor.status()):                  # Ch 14: fence every call to the current
        pipe.set_expected(i, s["generation"])                    #        generation, and follow restarts
    supervisor.listeners.append(pipe.set_expected)
    try:
        pipe.check_layout(n_layer=layout[-1].last)
        yield pipe, supervisor
    finally:
        pipe.close()
        supervisor.stop()
