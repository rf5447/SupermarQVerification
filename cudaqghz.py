import cudaq
import collections
from cudaqbenchmark import CudaQBenchmark
from cudaqfidelity import hellinger_fidelity


# ------------------------------------------------------------
# Helper: log-depth fanout pairs (same as Cirq version)
# ------------------------------------------------------------
def _fanout(indices):
    """
    Recursively generate log-depth fanout pairs, matching SupermarQ GHZ logic.
    """
    if len(indices) < 2:
        return
    mid = len(indices) // 2
    yield (indices[0], indices[mid])
    yield from _fanout(indices[:mid])
    yield from _fanout(indices[mid:])


# ------------------------------------------------------------
# CUDA-Q GHZ Benchmark
# ------------------------------------------------------------
class GHZ_CUDAQ(CudaQBenchmark):

    """Initialize a `GHZ` object.

    Args:
        num_qubits: Number of qubits in GHZ circuit.
        method: Circuit construction method to use. Must be "ladder", "star", or "logdepth". The
            "ladder" method uses a linear-depth CNOT ladder, appropriate for nearest-neighbor
            architectures. The "star" method is also linear depth, but with all CNOTs sharing
            the same control qubit. The "logdepth" method uses a log-depth CNOT fanout circuit.
    """
    def __init__(self, num_qubits: int, method: str = "ladder"):
        if method not in ("ladder", "star", "logdepth"):
            raise ValueError(
                f"Invalid GHZ method '{method}'. Must be 'ladder', 'star', or 'logdepth'."
            )
        self.n = num_qubits
        self.method = method

    # --------------------------------------------------------
    # Generate CUDA-Q GHZ kernel
    # --------------------------------------------------------
    def kernel(self):
        n = self.n

        # -----------------------------
        # LADDER KERNEL
        # -----------------------------
        if self.method == "ladder":

            @cudaq.kernel
            def k(n: int):
                q = cudaq.qvector(n)
                h(q[0])
                for i in range(1, n):
                    x.ctrl(q[i-1], q[i])
                mz(q)

            return k

        # -----------------------------
        # STAR KERNEL
        # -----------------------------
        elif self.method == "star":

            @cudaq.kernel
            def k(n: int):
                q = cudaq.qvector(n)
                h(q[0])
                for i in range(1, n):
                    x.ctrl(q[0], q[i])
                mz(q)

            return k

        # -----------------------------
        # LOG-DEPTH KERNEL
        # -----------------------------
        else:  # logdepth

            # Computer pairs using helper
            indices = list(range(n))
            pair_list = list(_fanout(indices))

            # Split the pairs into two static lists (controls and targets), for Just-In-Time (JIT) compilation
            pair_c = [c for (c, _) in pair_list]
            pair_t = [t for (_, t) in pair_list]
            num_pairs = len(pair_c)

            @cudaq.kernel
            def k(n: int):
                q = cudaq.qvector(n)
                h(q[0])
                for i in range(num_pairs):
                    x.ctrl(q[pair_c[i]], q[pair_t[i]])
                mz(q)

            return k

    # --------------------------------------------------------
    # Score function: identical to SupermarQ GHZ
    # --------------------------------------------------------
    def score(self, counts: collections.Counter) -> float:
        """
        Ideal GHZ distribution:
            '000...0' : 0.5
            '111...1' : 0.5
        """

        zero = "0" * self.n
        one = "1" * self.n

        ideal = {zero: 0.5, one: 0.5}

        total = sum(counts.values())
        device_dist = {}

        # CUDA-Q returns integer keys → convert to bitstrings
        for key, c in counts.items():
            if isinstance(key, int):
                bitstr = format(key, f"0{self.n}b")
            else:
                bitstr = str(key).zfill(self.n)
            device_dist[bitstr] = c / total

        return hellinger_fidelity(ideal, device_dist)
