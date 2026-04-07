import collections
from collections.abc import Mapping

import cudaq
import numpy as np
import scipy

from cudaqbenchmark import CudaQBenchmark

from cudaqvanilla import QAOAVanillaProxy_CUDAQ


class QAOAFermionicSwapProxy_CUDAQ(QAOAVanillaProxy_CUDAQ):
    """CUDA-Q translation of the SupermarQ QAOA Fermionic SWAP Proxy benchmark."""

    def _get_energy_for_bitstring(self, bitstring: str) -> float:
        # Reverse bitstring ordering due to SWAP network
        return super()._get_energy_for_bitstring(bitstring[::-1])

    def _final_virtual_map(self) -> list[int]:
        cover_a = [(idx - 1, idx) for idx in range(1, self.num_qubits, 2)]
        cover_b = [(idx - 1, idx) for idx in range(2, self.num_qubits, 2)]

        virtual_map = list(range(self.num_qubits))

        for layer in range(self.num_qubits):
            cover = cover_a if layer % 2 == 0 else cover_b
            for i, j in cover:
                virtual_map[j], virtual_map[i] = virtual_map[i], virtual_map[j]

        return virtual_map

    def _hamiltonian_operator(self):
        """
        Build the cost Hamiltonian for cudaq.observe().

        IMPORTANT:
        The fermionic SWAP network permutes which logical qubit ends on which
        physical wire, so we must build the Hamiltonian on the final PHYSICAL
        locations corresponding to the original logical qubits.
        """
        final_map = self._final_virtual_map()

        inverse_map = [0 for _ in range(self.num_qubits)]
        for physical, logical in enumerate(final_map):
            inverse_map[logical] = physical

        h_op = 0.0
        for i, j, weight in self.hamiltonian:
            p_i = inverse_map[i]
            p_j = inverse_map[j]
            h_op += -float(weight) * cudaq.spin.z(p_i) * cudaq.spin.z(p_j)
        return h_op

    def _get_opt_angles(self):
        h_op = self._hamiltonian_operator()

        def f(params):
            gamma = float(params[0])
            beta = float(params[1])

            exp_val = cudaq.observe(
                self.ansatz(),
                h_op,
                self.num_qubits,
                gamma,
                beta,
            ).expectation()

            return -float(exp_val)

        rng = np.random.default_rng()
        init_params = [
            float(rng.uniform() * 2 * np.pi),
            float(rng.uniform() * 2 * np.pi),
        ]

        out = scipy.optimize.minimize(f, init_params, method="COBYLA")
        return out["x"], out["fun"]

    def _build_swap_schedule(self):
        """
        Precompute the fermionic SWAP network schedule as flat arrays.

        This is a faithful translation of the Cirq implementation:
        - alternate cover_a / cover_b by layer
        - track virtual_map
        - for each active physical pair (i, j), use the weight associated with
          the current logical qubits (v_i, v_j)
        - then swap the virtual_map entries
        """
        num_qubits_val = int(self.num_qubits)

        cover_a = [(idx - 1, idx) for idx in range(1, num_qubits_val, 2)]
        cover_b = [(idx - 1, idx) for idx in range(2, num_qubits_val, 2)]

        weight_lookup = {}
        for i, j, w in self.hamiltonian:
            i = int(i)
            j = int(j)
            w = float(w)
            weight_lookup[(i, j)] = w
            weight_lookup[(j, i)] = w

        virtual_map = list(range(num_qubits_val))

        op_i = []
        op_j = []
        op_w = []

        for layer in range(num_qubits_val):
            cover = cover_a if layer % 2 == 0 else cover_b
            for i, j in cover:
                v_i = virtual_map[i]
                v_j = virtual_map[j]
                weight = weight_lookup[(v_i, v_j)]

                op_i.append(int(i))
                op_j.append(int(j))
                op_w.append(float(weight))

                virtual_map[j], virtual_map[i] = virtual_map[i], virtual_map[j]

        return op_i, op_j, op_w

    def ansatz(self):
        op_i, op_j, op_w = self._build_swap_schedule()
        num_ops = len(op_i)

        @cudaq.kernel
        def k(num_qubits: int, gamma: float, beta: float):
            q = cudaq.qvector(num_qubits)

            for i in range(num_qubits):
                h(q[i])

            for t in range(num_ops):
                i = op_i[t]
                j = op_j[t]
                phi = gamma * op_w[t]

                # ZZ + SWAP block, matching the original Cirq implementation
                x.ctrl(q[i], q[j])
                rz(2.0 * phi, q[j])
                x.ctrl(q[j], q[i])
                x.ctrl(q[i], q[j])

            for i in range(num_qubits):
                rx(2.0 * beta, q[i])

        return k

    def kernel(self):
        op_i, op_j, op_w = self._build_swap_schedule()
        num_ops = len(op_i)

        @cudaq.kernel
        def k(num_qubits: int, gamma: float, beta: float):
            q = cudaq.qvector(num_qubits)

            for i in range(num_qubits):
                h(q[i])

            for t in range(num_ops):
                i = op_i[t]
                j = op_j[t]
                phi = gamma * op_w[t]

                # ZZ + SWAP block, matching the original Cirq implementation
                x.ctrl(q[i], q[j])
                rz(2.0 * phi, q[j])
                x.ctrl(q[j], q[i])
                x.ctrl(q[i], q[j])

            for i in range(num_qubits):
                rx(2.0 * beta, q[i])

            mz(q)

        return k

    def score(self, counts: Mapping[str, float]) -> float:
        gamma = float(self.params[0])
        beta = float(self.params[1])

        ideal_value = float(
            cudaq.observe(
                self.ansatz(),
                self._hamiltonian_operator(),
                self.num_qubits,
                gamma,
                beta,
            ).expectation()
        )

        total_shots = sum(counts.values())
        experimental_probs = {}

        for key, value in counts.items():
            if isinstance(key, int):
                bitstring = format(key, f"0{self.num_qubits}b")
            else:
                bitstring = str(key).replace(" ", "").zfill(self.num_qubits)
            experimental_probs[bitstring] = float(value) / float(total_shots)

        experimental_value = self._get_expectation_value_from_probs(experimental_probs)

        denom = max(2.0 * abs(ideal_value), 1e-12)
        return 1.0 - abs(ideal_value - experimental_value) / denom