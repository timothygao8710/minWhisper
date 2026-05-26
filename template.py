import time
from icecream import IceCreamDebugger
import sys
from pathlib import Path

_STDIN = sys.__stdin__
_STDOUT = sys.__stdout__

_in_file = None
_out_file = None

PROGRAM_START = time.perf_counter()

def elapsed_prefix() -> str:
    return f"{time.perf_counter() - PROGRAM_START:8.3f}s |> "

dbg = IceCreamDebugger(prefix=elapsed_prefix)

def bern_ci(n_correct: int, n_total: int, z: float = 1.96) -> str:
    """Format a proportion with its 95% Wilson score confidence interval."""
    
    import math
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return f"{centre:.2%} ± {margin:.2%}"

def setio(
    filepath: str | Path | None = None,
    *,
    stdin: str | Path | None = None,
    stdout: str | Path | None = None,
    append: bool = False,
) -> None:
    """
    setio("out.txt")                 -> print() writes to out.txt
    setio(stdin="in.txt")            -> input() reads from in.txt
    setio(stdin="in.txt", stdout="out.txt")
    setio()                          -> restore standard stdin/stdout
    """

    global _in_file, _out_file

    # Close previous redirected files, if any.
    if _in_file:
        _in_file.close()
        _in_file = None

    if _out_file:
        _out_file.close()
        _out_file = None

    # Restore stdio first.
    sys.stdin = _STDIN
    sys.stdout = _STDOUT

    # Convenience: setio("file.txt") means redirect stdout.
    if filepath is not None:
        if stdout is not None:
            raise ValueError("Use either filepath or stdout, not both.")
        stdout = filepath

    if stdin is not None:
        _in_file = open(stdin, "r", encoding="utf-8")
        sys.stdin = _in_file

    if stdout is not None:
        mode = "a" if append else "w"
        _out_file = open(stdout, mode, encoding="utf-8")
        sys.stdout = _out_file


def unsetio() -> None:
    setio()


def plot_hist(data, save_name='example.jpeg'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    if any(isinstance(item, list) for item in data):
        for i, sublist in enumerate(data):
            ax.hist(sublist, alpha=0.5, label=f'Histogram {i + 1}')
        ax.legend()
    else:
        ax.hist(data)

    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    fig.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    
    for i in range(1, 10000):
        if i % 200 == 0:
            print(i, bern_ci(i * 0.6, i))