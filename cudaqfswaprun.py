import cudaq
import collections

from cudaqfswapsimulation import cudaq_simulation, noisy_cudaq_simulation
from cudaqbenchmark import CudaQBenchmark
from cudaqfswap import QAOAFermionicSwapProxy_CUDAQ


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
# Unified benchmarking driver for QAOA Fermionic Swap Proxy benchmarks
# ============================================================

def run_benchmark(benchmark: CudaQBenchmark, noise_levels=[0.0, 0.005, 0.02]):
    """Runs ideal + noisy simulations for a CUDA-Q QAOA Fermionic Swap Proxy benchmark."""

    kernel = benchmark.kernel()
    gamma, beta = benchmark.params
    kernel_args = (benchmark.num_qubits, float(gamma), float(beta))
    n = benchmark.num_qubits

    print(f"\n=== Running Benchmark: {benchmark.__class__.__name__} ===")
    print(f"Number of qubits: {n}")
    print(f"Number of Hamiltonian terms: {len(benchmark.hamiltonian)}")
    print(f"Gamma: {float(gamma)}")
    print(f"Beta: {float(beta)}")
    print("----------------------------------------------")

    # Draw circuit
    # print(cudaq.draw(kernel, *kernel_args))

    # -------------------------
    # Ideal simulation
    # -------------------------
    ideal_counts_raw = cudaq_simulation(kernel, *kernel_args, shots=1000)
    ideal_counts = normalize_counts(ideal_counts_raw, n)

    ideal_score = benchmark.score(ideal_counts)

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

    for num_qubits in [3, 4, 5]:
        benchmark = QAOAFermionicSwapProxy_CUDAQ(num_qubits)
        print("Benchmark configuration:", benchmark)

        noise_probs = [0.0, 0.005, 0.02]

        run_benchmark(benchmark, noise_probs)