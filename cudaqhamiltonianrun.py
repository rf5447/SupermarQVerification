import cudaq
import collections

from cudaqhamiltoniansimulation import cudaq_simulation, noisy_cudaq_simulation
from cudaqbenchmark import CudaQBenchmark
from cudaqhamiltonian import HamiltonianSimulation_CUDAQ   # ← your CUDA-Q class


# ------------------------------------------------------------
# Helper: normalize CUDA-Q SampleResult → bitstring Counter
# ------------------------------------------------------------
# def normalize_counts(counts, n):
#     """Convert CUDA-Q integer keys → standard bitstrings."""
#     out = collections.Counter()

#     for key, c in counts.items():
#         if isinstance(key, int):
#             bitstr = format(key, f"0{n}b")
#         else:
#             bitstr = str(key).zfill(n)
#         out[bitstr] = c

#     return out

def normalize_counts(counts, n):
    """Convert CUDA-Q integer keys to standard bitstrings."""
    out = collections.Counter()

    for key, c in counts.items():
        if isinstance(key, int):
            bitstr = format(key, f"0{n}b")[::-1]
        else:
            bitstr = str(key).zfill(n)[::-1]
        out[bitstr] = c

    return out


# ============================================================
# Unified benchmarking driver for HamiltonianSimulation
# ============================================================
def run_benchmark(benchmark: CudaQBenchmark, noise_levels=[0.0, 0.005, 0.02]):
    kernel = benchmark.kernel()
    # print(kernel)

    n = benchmark.num_qubits
    print(n)

    print(f"\n=== Running Benchmark: {benchmark.__class__.__name__} ===")
    
    # Draw circuit
    print(cudaq.draw(kernel, n, benchmark.time_step, benchmark.total_time))

    print(f"Num qubits        : {benchmark.num_qubits}")
    print(f"Time step (dt)    : {benchmark.time_step}")
    print(f"Total time (T)    : {benchmark.total_time}")
    print("----------------------------------------------")

    # ------------------------------------------------------------
    # Ideal simulation
    # ------------------------------------------------------------
    ideal_raw = cudaq_simulation(kernel,
                                 benchmark.num_qubits, benchmark.time_step, benchmark.total_time,
                                 shots=2000)

    ideal_counts = normalize_counts(ideal_raw,
                                    benchmark.num_qubits)

    print("Ideal measurement distribution:", ideal_counts)

    ideal_score = benchmark.score(ideal_counts)
    print(f"Ideal score: {ideal_score:.4f}\n")

    # ------------------------------------------------------------
    # Noisy simulations
    # ------------------------------------------------------------
    for p in noise_levels:
        print(f"Noise p = {p}")

        noisy_raw = noisy_cudaq_simulation(kernel,
                                           benchmark.num_qubits, benchmark.time_step, benchmark.total_time,
                                           p=p, shots=2000)

        noisy_counts = normalize_counts(noisy_raw,
                                        benchmark.num_qubits)

        print("    counts:", noisy_counts)

        noisy_score = benchmark.score(noisy_counts)
        print(f"    score:  {noisy_score:.4f}\n")

    print("==============================================\n")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    cudaq.set_target("density-matrix-cpu")

    for nq in [4, 7]:
        for steps in [1, 3]:
            ts = steps #1
            tt = 1 * ts

            benchmark = HamiltonianSimulation_CUDAQ(num_qubits=nq, time_step=ts, total_time=tt)

            print("Loaded benchmark:", benchmark)

            noise_probs = [0.0, 0.005, 0.02]
            run_benchmark(benchmark, noise_probs)
