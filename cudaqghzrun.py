import cudaq
import collections
from cudaqsimulation import cudaq_simulation, noisy_cudaq_simulation
from cudaqfidelity import hellinger_fidelity
from cudaqbenchmark import CudaQBenchmark
from cudaqghz import GHZ_CUDAQ

# ------------------------------------------------------------
# Run GHZ benchmarks
# ------------------------------------------------------------
def run_benchmark(benchmark: CudaQBenchmark, noise_levels=[0.0, 0.005, 0.02]):
    """Runs ideal + noisy simulations for a CUDA-Q GHZ benchmark."""
    
    kernel = benchmark.kernel()
    n = benchmark.n

    print(f"\n=== Running Benchmark: {benchmark.__class__.__name__} ===")
    print(f"Method: {benchmark.method}")
    print(f"Number of qubits: {n}")
    print("----------------------------------------------")

    # Draw circuit
    print(cudaq.draw(kernel, n))

    # -------------------------
    # Ideal simulation
    # -------------------------
    ideal_counts = cudaq_simulation(kernel, n, shots=1000)
    ideal_score = benchmark.score(ideal_counts)

    print("Ideal measurement distribution:", ideal_counts)
    print(f"Ideal score: {ideal_score:.4f}\n")

    # -------------------------
    # Noisy simulations
    # -------------------------
    for p in noise_levels:
        print(f"\nNoise p = {p}")

        noisy_counts = noisy_cudaq_simulation(kernel, n, p=p, shots=1000)
        noisy_score = benchmark.score(noisy_counts)

        # Convert CUDA-Q keys to bitstrings
        clean_counts = {
            (format(k, f"0{n}b") if isinstance(k, int) else str(k).zfill(n)): noisy_counts[k]
            for k in noisy_counts
        }

        print("    counts:", clean_counts)
        print(f"    score: {noisy_score:.4f}")

    print("\n==============================================\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    cudaq.set_target("density-matrix-cpu")

    # Choose GHZ parameters
    num_qubits = 5  # You can increase this as needed
    methods = ["ladder", "star", "logdepth"]
    noise_probs = [0.0, 0.005, 0.02]

    for method in methods:
        print(f"\n{'#'*60}")
        print(f"STARTING EXPERIMENT: {method.upper()}")
        print(f"{'#'*60}")
        
        # Initialize the benchmark with the current method in the loop
        benchmark = GHZ_CUDAQ(num_qubits, method=method)
        print("Benchmark configuration:", benchmark)

        # Run the ideal and noisy simulations for this specific method
        run_benchmark(benchmark, noise_probs)
