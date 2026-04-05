import cudaq
import numpy as np
import collections
from cudaqbenchmark import CudaQBenchmark


class HamiltonianSimulation_CUDAQ(CudaQBenchmark):

    def __init__(self, num_qubits: int, time_step: int = 1, total_time: int = 1):
        self.num_qubits = num_qubits
        self.time_step = time_step
        self.total_time = total_time

    # -------------------------------
    # Helper function to convert keys to bitstrings
    # -------------------------------
    def _to_bitstring(self, key):
        nq = self.num_qubits

        if isinstance(key, int):
            return format(key, f"0{nq}b")[::-1]

        if isinstance(key, str):
            return key.zfill(nq)[::-1]

        if isinstance(key, tuple):
            bits = "".join(str(b) for b in key)
            return bits.zfill(nq)[::-1]

        raise TypeError(f"Unsupported key type: {type(key)}")


    # -------------------------------
    # CUDA-Q kernel
    # -------------------------------
    def kernel(self):
        nq = self.num_qubits
        dt = self.time_step
        T = self.total_time

        hbar = 0.658212
        jz = hbar * np.pi / 4
        freq = 0.0048
        w_ph = 2 * np.pi * freq
        e_ph = 3 * np.pi * hbar / (8 * np.cos(np.pi * freq))

        num_steps = int(T / dt)

        @cudaq.kernel
        def k(nq: int, dt: int, T: int):
            q = cudaq.qvector(nq)

            for step in range(num_steps):
                t = (step + 0.5) * dt

                psi = -2.0 * e_ph * np.cos(w_ph * t) * dt / hbar
                for i in range(nq):
                    h(q[i])
                    rz(psi, q[i])
                    h(q[i])

                psi2 = -2.0 * jz * dt / hbar
                for i in range(nq - 1):
                    x.ctrl(q[i], q[i + 1])
                    rz(psi2, q[i + 1])
                    x.ctrl(q[i], q[i + 1])

            mz(q)

        return k

    # -------------------------------
    # Ideal counts / probabilities
    # -------------------------------
    def _ideal_counts(self):
        kernel = self.kernel()
        state = np.array(
            cudaq.get_state(kernel, self.num_qubits, self.time_step, self.total_time)
        ).flatten()

        out = collections.Counter()
        probs = np.abs(state) ** 2

        for idx, p in enumerate(probs):
            if p > 0:
                out[self._to_bitstring(idx)] = float(p)

        return out

    # -------------------------------
    # Magnetization
    # -------------------------------
    def _average_magnetization(self, counts, shots):
        mag = 0.0
        for bitstr, c in counts.items():
            spin_vals = [1 - 2 * int(b) for b in bitstr]
            mag += (sum(spin_vals) / len(spin_vals)) * c
        return mag / shots

    # -------------------------------
    # Score
    # -------------------------------
    def score(self, counts):
        total_shots = sum(counts.values())
        ideal = self._ideal_counts()

        mag_ideal = self._average_magnetization(ideal, shots=1)
        mag_exp = self._average_magnetization(counts, total_shots)

        return 1 - abs(mag_ideal - mag_exp) / 2