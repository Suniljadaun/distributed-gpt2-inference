"""
checks that every packae this project needs is installed , 
and measures how fast your cpu does matrix multiplication ( this is the core operation of LLM).
run from the project root: python src/check_env.py
 """

import importlib
import platform 
import sys
import time

REQUIRED  = [
    ("torch", "tensors and the GPT-2 math"),
    ("grpc","RPC between coordinator and workers"),
    ("grpc_tools", "turns .proto files into python code"),
    ("transformers","download GPT-2 weights + reference model"),
    ("numpy","number crunching for measurement"),
    ("pytest","running tests"),
]


def check_packages()->bool:
    all_ok = True
    for name, purpose in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")

            print(f"[OK] {name:<13} {version:<14} {purpose}")
        except ImportError:
            print(f"[MISSING] {name:<13} {'-':<14} {purpose}")
            all_ok  = False

    return all_ok



def cpu_benchmark() ->None:
    import torch

    n = 1024
    a = torch.randn(n,n)
    b = torch.randn(n,n)

    torch.matmul(a,b)  # warm-up here .. just check first clal pays one time setup
    #cost

    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
        torch.matmul(a,b)


    seconds = (time.perf_counter() - start)  / runs

    flops = 2 * n**3 # each of n * n outputs needs n multiplies + n adds

    print(f" torch threads : {torch.get_num_threads()}")

    print(f"cuda gpu found : {torch.cuda.is_available()}")

    print(f"{n} X {n} matmul : {seconds * 1000:.1f} ms -> {flops / seconds / 1e9:.1f} GFLPS/s")



    def main() -> None:
        print("python")
        print(f"version : {platform.python_version()}")
        print(f"interpreter : {sys.executable}")

        in_venv = sys.prefix != sys.base_prefix

        print(f"inside a venv :  {in_venv}") 

        print("\nPackages")
        ok = check_packages()

        if ok:
            print("\nCPU speed")
            cpu_benchmark()

            print("\n result: ", "environment ready " if ok and in_venv else "fix the items above")

            if __name__ == "__main__":
                main()