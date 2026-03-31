import cudaq
import collections

from cudaqphasecodesimulation import cudaq_simulation, noisy_cudaq_simulation
from cudaqbenchmark import CudaQBenchmark
from cudaqphasecode import PhaseCode_CUDAQ


# ------------------------------------------------------------
# Helper: normalize CUDA-Q SampleResult → bitstring Counter
# ------------------------------------------------------------
def normalize_counts(counts, n):
    """Convert CUDA-Q integer keys → standard bitstrings."""
    out = collections.Counter()

    for key, c in counts.items():
        if isinstance(key, int):
            bitstr = format(key, f"0{n}b")
        else:
            bitstr = str(key).zfill(n)
        out[bitstr] = c

    return out


# ============================================================
# Unified benchmarking driver for BitCode
# ============================================================

def run_benchmark(benchmark: CudaQBenchmark, noise_levels=[0.0, 0.005, 0.02]):
    print("hi")
    kernel = benchmark.kernel()
    # Number of physical qubits in the full circuit:
    num_total_qubits = 2 * benchmark.num_data_qubits - 1
    print("hi2")
    # Draw circuit
    print(cudaq.draw(kernel, benchmark.num_data_qubits, benchmark.num_rounds, benchmark.phase_state))
    print(f"\n=== Running Benchmark: {benchmark.__class__.__name__} ===")
    print(f"Data qubits      : {benchmark.num_data_qubits}")
    print(f"Phase state      : {benchmark.phase_state}")
    print(f"Measurement rounds: {benchmark.num_rounds}")
    print("----------------------------------------------")

    # ------------------------------------------------------------
    # Ideal simulation
    # ------------------------------------------------------------
    # Ideal simulation
    ideal_counts_raw = cudaq_simulation(kernel, benchmark.num_data_qubits, benchmark.num_rounds, benchmark.phase_state, shots=100)#2000)
    ideal_counts = normalize_counts(ideal_counts_raw, benchmark.num_data_qubits * 2 - 1)
    
    #print("Ideal measurement distribution:", ideal_counts)
    #print("ideal raw type:", type(ideal_counts_raw))
    #print("ideal raw:", ideal_counts_raw)

    #print("Measurement distribution:" + str(ideal_counts_raw))

    # ideal_score = benchmark.score(ideal_counts)
    ideal_score = benchmark.score(ideal_counts_raw)
    print(f"Ideal score: {ideal_score:.4f}\n")

    # ------------------------------------------------------------
    # Noisy simulations
    # ------------------------------------------------------------
    for p in noise_levels:
        print(f"Noise p = {p}")

        noisy_counts_raw = noisy_cudaq_simulation(kernel, benchmark.num_data_qubits, benchmark.num_rounds, benchmark.phase_state, p=p, shots=100)#2000)
        noisy_counts = normalize_counts(noisy_counts_raw, benchmark.num_data_qubits * 2 - 1)
        

        #print("noisy raw type:", type(noisy_counts_raw))
        #print("noisy raw:", noisy_counts_raw)
        #print("    counts:", noisy_counts)

        # noisy_score = benchmark.score(noisy_counts)
        noisy_score = benchmark.score(noisy_counts_raw)
        print(f"    score:  {noisy_score:.4f}\n")

    print("==============================================\n")


# ============================================================
# Main
# ============================================================

# if __name__ == "__main__":
#     cudaq.set_target("density-matrix-cpu")

#     benchmark = BitCode_CUDAQ(num_data_qubits=3,
#                               num_rounds=1,
#                               phase_state=[0, 1, 0])

#     print("Loaded benchmark:", benchmark)

#     noise_probs = [0.0, 0.005, 0.02]
#     run_benchmark(benchmark, noise_probs)

if __name__ == "__main__":
    cudaq.set_target("density-matrix-cpu")

    noise_probs = [0.0, 0.005, 0.02]

    for num_data_qubits in [3, 4]:
        phase_state = [i % 2 for i in range(num_data_qubits)]
        print(f"Running benchmarks for {num_data_qubits} data qubits with phase state {phase_state}...")

        for num_rounds in [1, 2]:
            print(f"\n--- Benchmark with {num_rounds} measurement rounds ---")
            benchmark = PhaseCode_CUDAQ(num_data_qubits=num_data_qubits,
                                         num_rounds=num_rounds,
                                         phase_state=phase_state)

            print("Loaded benchmark:", benchmark)
            run_benchmark(benchmark, noise_probs)