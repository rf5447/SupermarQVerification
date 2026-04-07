import collections
from collections.abc import Mapping

import cudaq
import numpy as np
import scipy

from cudaqbenchmark import CudaQBenchmark


class QAOAVanillaProxy_CUDAQ(CudaQBenchmark):
    """CUDA-Q translation of the SupermarQ QAOA Vanilla Proxy benchmark."""

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = int(num_qubits)
        self.hamiltonian = self._gen_sk_hamiltonian()
        self.params = self._gen_angles()

    def _gen_sk_hamiltonian(self) -> list[tuple[int, int, float]]:
        """Randomly pick +1 or -1 for each edge weight."""
        hamiltonian = []
        rng = np.random.default_rng()
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                hamiltonian.append((i, j, float(rng.choice([-1, 1]))))

        rng.shuffle(hamiltonian)
        return hamiltonian

    def _get_energy_for_bitstring(self, bitstring: str) -> float:
        energy = 0.0
        for i, j, weight in self.hamiltonian:
            if bitstring[i] == bitstring[j]:
                energy -= weight
            else:
                energy += weight
        return energy

    def _get_expectation_value_from_probs(
        self, probabilities: Mapping[str, float]
    ) -> float:
        expectation_value = 0.0
        for bitstring, probability in probabilities.items():
            expectation_value += probability * self._get_energy_for_bitstring(bitstring)
        return expectation_value

    def _hamiltonian_operator(self):
        """
        Build the cost Hamiltonian for cudaq.observe().

        IMPORTANT:
        This must match _get_energy_for_bitstring():
            same bits  -> -weight
            diff bits  -> +weight

        Since Z_i Z_j = +1 for same bits and -1 for different bits,
        the matching operator is:
            H = -sum weight * Z_i Z_j
        """
        h_op = 0.0
        for i, j, weight in self.hamiltonian:
            h_op += -float(weight) * cudaq.spin.z(i) * cudaq.spin.z(j)
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

            # maximize expectation <H>, so minimize negative expectation
            return -float(exp_val)

        rng = np.random.default_rng()
        init_params = [
            float(rng.uniform() * 2 * np.pi),
            float(rng.uniform() * 2 * np.pi),
        ]
        #print("init_params:", init_params, type(init_params))

        out = scipy.optimize.minimize(f, init_params, method="COBYLA")
        return out["x"], out["fun"]

    def _gen_angles(self):
        best_params = [0.0, 0.0]
        best_cost = float("inf")

        for _ in range(5):
            params, cost = self._get_opt_angles()
            params = [float(params[0]), float(params[1])]
            cost = float(cost)

            if cost < best_cost:
                best_params = params
                best_cost = cost

        return best_params

    # -------------------------------
    # CUDA-Q ansatz kernel (NO measurement)
    # -------------------------------
    def ansatz(self):
        edge_i = [int(i) for i, _, _ in self.hamiltonian]
        edge_j = [int(j) for _, j, _ in self.hamiltonian]
        edge_w = [float(w) for _, _, w in self.hamiltonian]
        num_terms = len(edge_i)

        @cudaq.kernel
        def k(num_qubits: int, gamma: float, beta: float):
            q = cudaq.qvector(num_qubits)

            for i in range(num_qubits):
                h(q[i])

            for t in range(num_terms):
                i = edge_i[t]
                j = edge_j[t]
                phi = gamma * edge_w[t]

                x.ctrl(q[i], q[j])
                rz(2.0 * phi, q[j])
                x.ctrl(q[i], q[j])

            for i in range(num_qubits):
                rx(2.0 * beta, q[i])

        return k

    # -------------------------------
    # CUDA-Q sampling kernel (with measurement)
    # -------------------------------
    def kernel(self):
        edge_i = [int(i) for i, _, _ in self.hamiltonian]
        edge_j = [int(j) for _, j, _ in self.hamiltonian]
        edge_w = [float(w) for _, _, w in self.hamiltonian]
        num_terms = len(edge_i)

        @cudaq.kernel
        def k(num_qubits: int, gamma: float, beta: float):
            q = cudaq.qvector(num_qubits)

            for i in range(num_qubits):
                h(q[i])

            for t in range(num_terms):
                i = edge_i[t]
                j = edge_j[t]
                phi = gamma * edge_w[t]

                x.ctrl(q[i], q[j])
                rz(2.0 * phi, q[j])
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