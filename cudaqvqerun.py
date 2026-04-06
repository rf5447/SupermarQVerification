import cudaq
import collections

from cudaqvqesimulation import cudaq_simulation, noisy_cudaq_simulation
from cudaqbenchmark import CudaQBenchmark
from cudaqvqe import VQEProxy_CUDAQ


# ------------------------------------------------------------
# Helper: normalize CUDA-Q counts -> standard bitstring dict
# ------------------------------------------------------------
def normalize_counts(counts, n):
    out = collections.Counter()

    for key, c in counts.items():
        if isinstance(key, int):
            bitstr = format(key, f"0{n}b")[::-1]
        else:
            bitstr = str(key).replace(" ", "").zfill(n)[::-1]
        out[bitstr] = c

    return dict(out)


# ============================================================
# Unified benchmarking driver for VQEProxy_CUDAQ
# ============================================================
def run_benchmark(benchmark: CudaQBenchmark, noise_levels=[0.0, 0.005, 0.02]):
    """Runs ideal + noisy simulations for a CUDA-Q VQE benchmark."""

    kernel = benchmark.kernel()
    n = benchmark.num_qubits

    print(f"\n=== Running Benchmark: {benchmark.__class__.__name__} ===")
    print(f"Number of qubits: {n}")
    print(f"Number of layers: {benchmark.num_layers}")
    print("----------------------------------------------")

    # -------------------------
    # Ideal simulation
    # -------------------------
    z_args = benchmark.kernel_args(0)
    x_args = benchmark.kernel_args(1)

    ideal_counts_z_raw = cudaq_simulation(kernel, *z_args, shots=1000)
    ideal_counts_x_raw = cudaq_simulation(kernel, *x_args, shots=1000)

    ideal_counts_z = normalize_counts(ideal_counts_z_raw, n)
    ideal_counts_x = normalize_counts(ideal_counts_x_raw, n)

    ideal_score = benchmark.score([ideal_counts_z, ideal_counts_x])

    print("Ideal Z-basis measurement distribution:", ideal_counts_z)
    print("Ideal X-basis measurement distribution:", ideal_counts_x)
    print(f"Ideal score: {ideal_score:.4f}")

    # -------------------------
    # Noisy simulations
    # -------------------------
    for p in noise_levels:
        print(f"\nNoise p = {p}")

        noisy_counts_z_raw = noisy_cudaq_simulation(kernel, *z_args, p=p, shots=1000)
        noisy_counts_x_raw = noisy_cudaq_simulation(kernel, *x_args, p=p, shots=1000)

        noisy_counts_z = normalize_counts(noisy_counts_z_raw, n)
        noisy_counts_x = normalize_counts(noisy_counts_x_raw, n)

        noisy_score = benchmark.score([noisy_counts_z, noisy_counts_x])

        print("    Z counts:", noisy_counts_z)
        print("    X counts:", noisy_counts_x)
        print(f"    score: {noisy_score:.4f}")

    print("\n==============================================\n")


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    cudaq.set_target("density-matrix-cpu")

    for num_qubits in [3, 4, 5]:
        benchmark = VQEProxy_CUDAQ(num_qubits=num_qubits, num_layers=1)
        print("Benchmark configuration:", benchmark)

        noise_probs = [0.0, 0.005, 0.02]
        run_benchmark(benchmark, noise_probs)