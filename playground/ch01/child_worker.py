"""
child_worker.py - Chapter 1, demo 4 (part 1 of 2)
A pretend worker process: prints a heartbeat twice a second until it is killed.
Normally started by supervisor.py, but you can run it alone:
    python playground/ch01/child_worker.py --generation 1
"""
import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    args = parser.parse_args()

    beat = 0
    while True:
        print(f"    [worker pid={os.getpid()} gen={args.generation}] heartbeat {beat}", flush=True)
        beat += 1
        time.sleep(0.5)


if __name__ == "__main__":
    main()
