import cudaq
import collections

from cudaqmbsimulation import cudaq_simulation, noisy_cudaq_simulation
from cudaqbenchmark import CudaQBenchmark
from cudaqmb import MerminBell_CUDAQ

# ------------------------------------------------------------
# Helper: normalize CUDA-Q counts -> standard bitstring dict
# ------------------------------------------------------------
def normalize_counts(counts, n):
    out = collections.Counter()

    for key, c in counts.items():
        if isinstance(key, int):
            bitstr = format(key, f"0{n}b")
        else:
            bitstr = str(key).replace(" ", "").zfill(n)
        out[bitstr] = c

    return dict(out)


# ============================================================
# Unified benchmarking driver for Mermin-Bell benchmarks
# ============================================================

def run_benchmark(benchmark: CudaQBenchmark, noise_levels=[0.0, 0.005, 0.02]):
    """Runs ideal + noisy simulations for a CUDA-Q Mermin-Bell benchmark."""

    # Important: kernel() must be called before kernel_args(),
    # since kernel() extracts and stores the ordered measurement circuit data.
    kernel = benchmark.kernel()
    kernel_args = benchmark.kernel_args()
    n = benchmark.num_qubits

    print(f"\n=== Running Benchmark: {benchmark.__class__.__name__} ===")
    print(f"Number of qubits: {n}")
    print(f"Number of Mermin terms: {len(benchmark.mermin_operator)}")
    print("----------------------------------------------")

    # Draw circuit
    # print(cudaq.draw(kernel, *kernel_args))

    # -------------------------
    # Ideal simulation
    # -------------------------
    ideal_counts_raw = cudaq_simulation(kernel, *kernel_args, shots=1000)
    ideal_counts = normalize_counts(ideal_counts_raw, n)

    ideal_score = benchmark.score(ideal_counts)
    ideal_score_reversed = benchmark.score({k[::-1]: v for k, v in ideal_counts.items()})

    print("Ideal measurement distribution:", ideal_counts)
    print(f"Ideal score: {ideal_score:.4f}")

    # -------------------------
    # Noisy simulations
    # -------------------------
    for p in noise_levels:
        print(f"\nNoise p = {p}")

        noisy_counts_raw = noisy_cudaq_simulation(kernel, *kernel_args, p=p, shots=1000)
        noisy_counts = normalize_counts(noisy_counts_raw, n)

        noisy_score = benchmark.score(noisy_counts)

        print("    counts:", noisy_counts)
        print(f"    score: {noisy_score:.4f}")

    print("\n==============================================\n")


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    cudaq.set_target("density-matrix-cpu")

    # Choose Mermin-Bell parameters
    for num_qubits in [3, 4, 5]:

        benchmark = MerminBell_CUDAQ(num_qubits)
        print("Benchmark configuration:", benchmark)

        noise_probs = [0.0, 0.005, 0.02]

        run_benchmark(benchmark, noise_probs)