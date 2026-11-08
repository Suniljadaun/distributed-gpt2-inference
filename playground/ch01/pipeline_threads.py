"""
pipeline_threads.py - Chapter 1, demo 3
A two-stage pipeline built from threads and queues: a miniature of the whole project.
Experiment 1: sequential vs pipelined timing.  Experiment 2: a bounded queue rejecting a flood.
Run:  python playground/ch01/pipeline_threads.py
"""
import queue
import threading
import time

STAGE_TIME = 0.05   # pretend each stage needs 50 ms of compute per item
STOP = None         # "sentinel": a special item that means "no more work, shut down"


def stage_a(x: int) -> int:
    time.sleep(STAGE_TIME)
    return x * 10


def stage_b(x: int) -> int:
    time.sleep(STAGE_TIME)
    return x + 1


def run_sequential(items: list[int]) -> list[int]:
    return [stage_b(stage_a(x)) for x in items]


def stage_worker(name: str, fn, inbox: queue.Queue, outbox: queue.Queue) -> None:
    """Loop forever: take an item, process it, pass it on. Forward STOP and exit."""
    while True:
        item = inbox.get()          # blocks until something arrives
        if item is STOP:
            outbox.put(STOP)
            return
        request_id, value = item
        outbox.put((request_id, fn(value)))


def run_pipelined(items: list[int], queue_size: int = 4) -> list[int]:
    q_in: queue.Queue = queue.Queue(maxsize=queue_size)   # bounded!
    q_mid: queue.Queue = queue.Queue(maxsize=queue_size)
    q_out: queue.Queue = queue.Queue()

    threads = [
        threading.Thread(target=stage_worker, args=("A", stage_a, q_in, q_mid)),
        threading.Thread(target=stage_worker, args=("B", stage_b, q_mid, q_out)),
    ]
    for t in threads:
        t.start()

    for request_id, x in enumerate(items):
        q_in.put((request_id, x))   # blocks if the queue is full: backpressure on the producer
    q_in.put(STOP)

    results: dict[int, int] = {}
    while (item := q_out.get()) is not STOP:
        request_id, value = item
        results[request_id] = value
    for t in threads:
        t.join()
    return [results[i] for i in range(len(items))]


def flood(n_requests: int, queue_size: int) -> None:
    """Submit requests faster than stage A can take them, using put_nowait (reject, don't wait)."""
    inbox: queue.Queue = queue.Queue(maxsize=queue_size)
    accepted, rejected = 0, 0

    def slow_consumer() -> None:
        while (item := inbox.get()) is not STOP:
            stage_a(item)

    consumer = threading.Thread(target=slow_consumer)
    consumer.start()
    for i in range(n_requests):
        try:
            inbox.put_nowait(i)
            accepted += 1
        except queue.Full:
            rejected += 1          # in the real system: reply RESOURCE_EXHAUSTED to the client
        time.sleep(0.01)           # requests arrive every 10 ms, but A needs 50 ms each
    inbox.put(STOP)
    consumer.join()
    print(f"  {n_requests} requests, queue size {queue_size}: accepted {accepted}, rejected {rejected}")


def main() -> None:
    items = list(range(10))

    print("Experiment 1: sequential vs pipelined")
    t0 = time.perf_counter()
    seq = run_sequential(items)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    pipe = run_pipelined(items)
    t_pipe = time.perf_counter() - t0

    print(f"  sequential : {t_seq:.2f} s   results {seq}")
    print(f"  pipelined  : {t_pipe:.2f} s   results {pipe}")
    print(f"  same results? {seq == pipe}    speed-up: {t_seq / t_pipe:.2f}x")

    print("\nExperiment 2: flooding a bounded queue")
    flood(n_requests=30, queue_size=4)


if __name__ == "__main__":
    main()
