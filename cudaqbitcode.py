import collections
import cudaq
from cudaqbenchmark import CudaQBenchmark
from cudaqfidelity import hellinger_fidelity

class BitCode_CUDAQ(CudaQBenchmark):
    """Creates a CUDA-Q kernel for syndrome measurement in a bit-flip error correcting code.

    Args:
        num_data_qubits: The number of data qubits.
        num_rounds: The number of measurement rounds.
        bit_state: A list denoting the state to initialize each data qubit to.

    Raises:
        ValueError: If `bit_state` is longer than `num_data_qubits`.
        TypeError: If `bit_state` is not a `list`.
        ValueError: If `bit_state` contains values not in {0,1}.
    """

    def __init__(self, num_data_qubits: int, num_rounds: int, bit_state: list[int]) -> None:
        if len(bit_state) != num_data_qubits:
            raise ValueError("The length of `bit_state` must match the number of data qubits.")
        if not isinstance(bit_state, list):
            raise TypeError("`bit_state` must be a `list[int]`.")
        if not set(bit_state).issubset({0, 1}):
            raise ValueError("Entries of `bit_state` must be 0, 1 integers.")

        self.num_data_qubits = num_data_qubits
        self.num_rounds = num_rounds
        self.bit_state = bit_state
        
    def kernel(self):
        @cudaq.kernel
        def k(num_data_qubits: int, num_rounds: int, bit_state: list[int]):
            num_qubits = 2 * num_data_qubits - 1
            q = cudaq.qvector(num_qubits)
            num_ancillas = num_data_qubits - 1

            # Initialize data qubits (indices 0, 2, 4...)
            for i in range(num_data_qubits):
                if bit_state[i] == 1:
                    x(q[2 * i])

            for _ in range(num_rounds):
                # Cycle through ancillas (indices 1, 3, 5...)
                for i in range(num_ancillas):
                    idx = 2 * i + 1
                    # Parity check
                    x.ctrl(q[idx - 1], q[idx])
                    x.ctrl(q[idx + 1], q[idx])
                    
                    # Mid-circuit measurement
                    result = mz(q[idx])

                    # Manual Reset: If qubit is 1, flip to 0. 
                    # This is more stable than reset() in current QPP versions.
                    if result:
                        x(q[idx])

            # Final measurement of everything
            # Important: Measure one by one to avoid register shape mismatches
            for i in range(num_qubits):
                mz(q[i])

        return k

    def _get_ideal_dist(self) -> dict[str, float]:
        """Return the ideal probability distribution of the benchmark."""
        ancilla_state, final_state = "", ""
        for i in range(self.num_data_qubits - 1):
            ancilla_state += str((self.bit_state[i] + self.bit_state[i + 1]) % 2)
            final_state += str(self.bit_state[i]) + "0"
        final_state += str(self.bit_state[-1])
        #print(f"Ancilla state: {ancilla_state}, Final state: {final_state}")
        ideal_bitstring = [ancilla_state] * self.num_rounds + [final_state]
        #print(f"Ideal bitstring: {ideal_bitstring}")
        return {"".join(ideal_bitstring): 1.0}
    # def _get_ideal_dist(self) -> dict[str, float]:
    #     """Return the ideal probability distribution of the benchmark.
    #        Different from the original version in that it ignores the ancilla states. 
    #        This is because CUDA-Q's mid-circuit measurement results are not currently 
    #        included in the output counts, so we cannot condition on them for scoring."""
    #     final_state = ""
    #     for i in range(self.num_data_qubits - 1):
    #         final_state += str(self.bit_state[i]) + "0"
    #     final_state += str(self.bit_state[-1])
    #     return {final_state: 1.0}

    # def score(self, counts: collections.Counter) -> float:
    #     """Compute benchmark score."""
    #     ideal_dist = self._get_ideal_dist()
    #     total_shots = sum(counts.values())
    #     experimental_dist = {}
    #     # print(f"Ideal distribution: {ideal_dist}")
    #     # print(f"Raw counts from CUDA-Q: {counts}")
    #     for bitstr, shots in counts.items():
    #         s = str(bitstr).replace(" ", "")
    #         experimental_dist[s] = shots / total_shots
            
    #     # print(f"Experimental distribution: {experimental_dist}")

    #     return hellinger_fidelity(ideal_dist, experimental_dist)

    def score(self, sample_result) -> float:
        """Compute benchmark score from a CUDA-Q SampleResult."""
        ideal_dist = self._get_ideal_dist()
        experimental_dist = {}

        final_counts = sample_result.get_register_counts("__global__")
        ancilla_counts = sample_result.get_register_counts("result")

        #print(final_counts)
        #print(ancilla_counts)

        total_shots = sum(shots for _, shots in final_counts.items())
        #print(f"Total shots (from final counts): {total_shots}")
        for ancilla_bitstr, ancilla_shots in ancilla_counts.items():
            ancilla_s = str(ancilla_bitstr).replace(" ", "")

            for final_bitstr, final_shots in final_counts.items():
                final_s = str(final_bitstr).replace(" ", "")

                combined_bitstr = ancilla_s + final_s
                # Best available approximation from separate register histograms
                shots = min(ancilla_shots, final_shots)
                #print(f"Ancilla bitstring: {ancilla_s}, Final bitstring: {final_s}, Combined: {combined_bitstr}, Shots: {final_shots}, Ancilla shots: {ancilla_shots}, Shots used for combined: {shots}")

                if combined_bitstr not in experimental_dist:
                    experimental_dist[combined_bitstr] = 0.0
                experimental_dist[combined_bitstr] += shots / total_shots

        return hellinger_fidelity(ideal_dist, experimental_dist)