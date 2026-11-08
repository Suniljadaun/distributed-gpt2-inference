"""
supervisor.py - Chapter 1, demo 4 (part 2 of 2)
Starts child_worker.py as a separate OS process, "crashes" it, notices, and restarts it
with a higher generation number. This is the skeleton of Chapter 12's failure handling.
Run:  python playground/ch01/supervisor.py
"""
import subprocess
import sys
import time
from pathlib import Path

WORKER_SCRIPT = Path(__file__).with_name("child_worker.py")


def start_worker(generation: int) -> subprocess.Popen:
    # sys.executable = the python.exe running this script, so the child uses the same .venv
    proc = subprocess.Popen([sys.executable, str(WORKER_SCRIPT), "--generation", str(generation)])
    print(f"[supervisor] started worker pid={proc.pid} generation={generation}", flush=True)
    return proc


def main() -> None:
    generation = 1
    proc = start_worker(generation)
    started = time.perf_counter()
    crashed_once = False

    while time.perf_counter() - started < 5.0:          # supervise for 5 seconds
        time.sleep(0.2)
        elapsed = time.perf_counter() - started

        if not crashed_once and elapsed > 1.6:
            print("[supervisor] *** injecting a crash: killing the worker ***", flush=True)
            proc.kill()                                  # like the process dying suddenly
            crashed_once = True

        exit_code = proc.poll()                          # None = still running
        if exit_code is not None:
            print(f"[supervisor] worker pid={proc.pid} is dead (exit code {exit_code}); restarting", flush=True)
            generation += 1
            proc = start_worker(generation)

    print("[supervisor] done; stopping the worker politely", flush=True)
    proc.terminate()
    proc.wait(timeout=5)                                 # always reap child processes
    print(f"[supervisor] final generation = {generation}", flush=True)


if __name__ == "__main__":
    main()
