"""
race.py - Chapter 1, demo 2
A race condition (lost updates) between threads, and how a Lock fixes it.
Run:  python playground/ch01/race.py
"""
import threading
import time

THREADS = 4
INCREMENTS = 200


class Counter:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()

    def unsafe_increment(self) -> None:
        current = self.value      # 1. read
        time.sleep(0)             # 2. give another thread a chance to run (widens the race window)
        self.value = current + 1  # 3. write: may overwrite another thread's update

    def safe_increment(self) -> None:
        with self.lock:           # only one thread at a time inside this block
            current = self.value
            time.sleep(0)
            self.value = current + 1


def run(method_name: str) -> int:
    counter = Counter()
    method = getattr(counter, method_name)

    def work() -> None:
        for _ in range(INCREMENTS):
            method()

    threads = [threading.Thread(target=work) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()  # wait until each thread has finished
    return counter.value


def main() -> None:
    expected = THREADS * INCREMENTS
    print(f"expected        : {expected}")
    print(f"without a lock  : {run('unsafe_increment')}")
    print(f"with a lock     : {run('safe_increment')}")


if __name__ == "__main__":
    main()
