"""
cluster.py - Chapter 8
Start the worker processes for a layout, wait until they are ready, check the layout,
hand back a Pipeline, and always stop the workers afterwards.
"""
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from src.coordinator.pipeline import Pipeline


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    port: int
    first: int
    last: int
    embed: bool = False
    head: bool = False
    extra_args: tuple[str, ...] = ()       # Ch 10: e.g. ("--fail-every", "3") for fault injection

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        cmd = [sys.executable, "-m", "src.worker.server", "--worker-id", self.worker_id,
               "--port", str(self.port), "--first", str(self.first), "--last", str(self.last)]
        cmd += (["--embed"] if self.embed else []) + (["--head"] if self.head else [])
        return cmd + list(self.extra_args)


def two_stage_layout(base_port: int = 50061, split: int = 3, n_layer: int = 6) -> list[WorkerSpec]:
    return [
        WorkerSpec("A", base_port, 0, split, embed=True),
        WorkerSpec("B", base_port + 1, split, n_layer, head=True),
    ]


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
